#!/usr/bin/env python
"""Compute the METHODOLOGY §6 metrics (definitions: DECISION_LOG.md D-042) from a run's
stored eval-view renders against the frozen eval set.

Thin CLI; logic lives in src/metrics/. Writes only under experiments/results/metrics/.

Usage:
    python scripts/compute_metrics.py <run_id>
    python scripts/compute_metrics.py --all
"""

import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import sys

import numpy as np

from metrics import DEFINITION_VERSION
from metrics import compute as C
from metrics import eval_data as E
from metrics import image_metrics as M

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "experiments", "results", "metrics")


def _git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True).stdout.strip()


def _read_results_rows():
    with open(os.path.join(REPO, "experiments", "results", "results.csv"), newline="") as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def _checkpoint_shas():
    with open(os.path.join(REPO, "experiments", "results", "phase6_final_checkpoints.csv"), newline="") as f:
        return {r["run_id"]: r["sha256"] for r in csv.DictReader(f)}


def _eval_manifest_sha():
    return M.sha256_file(os.path.join(REPO, "data", "blender_scenes", "eval_holdout_SHA256SUMS.txt"))


def _eval_aggregate_sha():
    with open(os.path.join(REPO, "data", "blender_scenes", "eval_holdout_SHA256SUMS.txt")) as f:
        for line in f:
            if line.startswith("# aggregate-sha256:"):
                return line.split(":", 1)[1].strip()
    raise ValueError("aggregate sha256 line not found in eval_holdout_SHA256SUMS.txt")


def compute_one(run_id, data, meter, results_rows, ckpt_shas):
    run_dir = os.path.join(REPO, "experiments", "runs", run_id)
    rdir = os.path.join(REPO, "experiments", "renders", run_id)
    meta_path = os.path.join(rdir, "render_meta.json")
    meta = json.load(open(meta_path))
    if meta["eval_manifest_sha256"] != _eval_manifest_sha() or meta["eval_set_aggregate_sha256"] != _eval_aggregate_sha():
        raise ValueError("%s: render_meta eval-set hashes differ from this laptop's frozen set" % run_id)
    if meta["checkpoint_sha256"] != ckpt_shas[run_id]:
        raise ValueError("%s: render_meta checkpoint sha != phase6_final_checkpoints.csv" % run_id)

    renders = E.load_renders(os.path.join(rdir, "renders_fp16.npz"))
    if renders.min() < 0.0 or renders.max() > 1.0:
        raise ValueError("%s: renders outside [0,1]: [%g, %g]" % (run_id, renders.min(), renders.max()))

    view_rows = C.compute_view_rows(renders, data["gt"], data, meter)
    agg = C.aggregate(view_rows)
    if agg["n_inf_psnr_views"]:
        print("WARNING %s: %d inf PSNR value(s) (MSE == 0); recorded as inf in per_view, "
              "aggregates are therefore inf where affected" % (run_id, agg["n_inf_psnr_views"]))

    eval_json = json.load(open(os.path.join(run_dir, "eval_%s.json" % run_id)))
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    rr = results_rows[run_id]
    row = {"run_id": run_id, "condition": rr["condition_id"], "seed": int(rr["seed"]),
           "resumed": int(bool(summary.get("resumed"))), "config_path": rr["config_path"],
           "render_meta_sha256": M.sha256_file(meta_path), "checkpoint_sha256": meta["checkpoint_sha256"],
           "definition_version": DEFINITION_VERSION,
           "lpips_batch_size": meter.batch_size, "lpips_oom_fallback": int(meter.oom_fallback),
           "eval_json_psnr": float(eval_json["test_psnr"]),
           "whole_psnr_dev_vs_eval_json": float(agg["whole_psnr"] - float(eval_json["test_psnr"]))}
    row.update(agg)

    C.write_csv_atomic(os.path.join(OUT, "per_view", "%s.csv" % run_id), C.VIEW_COLUMNS, view_rows)
    C.upsert_run_row(os.path.join(OUT, "per_run.csv"), row)
    print("%s: whole PSNR %.4f (eval json %.4f, dev %+.4f) | tgt %.3f col %.3f plate %.3f | included %d/%d"
          % (run_id, agg["whole_psnr"], row["eval_json_psnr"], row["whole_psnr_dev_vs_eval_json"],
             agg["tgt_psnr"], agg["col_psnr"], agg["plate_psnr"], agg["n_included"], agg["n_views"]))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if bool(a.run_id) == a.all:
        ap.error("give exactly one of <run_id> or --all")

    results_rows = _read_results_rows()
    ckpt_shas = _checkpoint_shas()
    runs = list(results_rows) if a.all else [a.run_id]
    for r in runs:
        if r not in results_rows:
            sys.exit("unknown run_id %s (not in results.csv)" % r)

    data = E.load_eval_data(REPO)
    E.assert_partition(data["m_dil"])
    meter = M.LpipsMeter(batch_size=4)

    C.write_csv_atomic(os.path.join(OUT, "reference_gt_vs_plate.csv"),
                       ["view_index", "included", "gt_vs_plate_mse", "gt_vs_plate_psnr"],
                       C.reference_gt_vs_plate(data))

    for r in runs:
        compute_one(r, data, meter, results_rows, ckpt_shas)

    import skimage
    dirty = _git("status", "--porcelain", "--", "src/metrics", "scripts/compute_metrics.py")
    C.write_json_atomic(os.path.join(OUT, "metrics_meta.json"), {
        "definition_version": DEFINITION_VERSION,
        "lpips_version": importlib.metadata.version("lpips"), "skimage_version": skimage.__version__,
        "lpips_net": "alex", "lpips_weights_sha256": meter.weights_sha256,
        "lpips_weights_paths": meter.paths,
        "metrics_code_commit": _git("rev-parse", "HEAD"),
        "metrics_code_dirty_paths": dirty.splitlines(),
        "eval_manifest_sha256": _eval_manifest_sha(), "eval_set_aggregate_sha256": _eval_aggregate_sha(),
        "mask_dilation_px": data["dilation_px"],
        "min_visible_area_fraction": data["min_visible_area_fraction"],
        "n_views_included": int(data["included"].sum()),
        "ssim_params": {k: v for k, v in M.SSIM_KW.items() if k != "full"},
    })


if __name__ == "__main__":
    sys.exit(main())
