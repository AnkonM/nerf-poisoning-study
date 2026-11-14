#!/usr/bin/env python
"""Phase 9 task Part 5: depth noise floor (exploratory addendum to D-044, descriptive only).

For control, C3, C5 and C7: mean |depth difference| over M_dil between every pair of seeds (1-2, 1-3, 2-3),
beside the D-044 Part 3 values (condition at seed 1 vs control at seed 1). Seed-1 depth comes from the float16
arrays written by D-044 Part 3; seeds 2 and 3 are rendered here with the same path (final checkpoints, perturb 0,
fp32, GPU only, ray chunk ladder 2048 -> 256, no CPU fallback). ALL differences are computed on float16-rounded
depth so the comparison is like-for-like; the effect of that rounding is reported against the D-044 float32 numbers.

Writes tables to experiments/results/phase8/followup/ and float16 arrays to experiments/renders_diagnostics/ (git-ignored).
Never writes renders, checkpoints, per_run.csv, results.csv or frozen data.

Usage:  python scripts/depth_noise_floor.py configs/analysis/phase9_depth_noise_floor.yaml
"""

import csv
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

from metrics import depth_diag as DD
from metrics import eval_data as E
from metrics import followup as FU
from utils.config import load_config

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def write_csv(path, rows):
    cols = list(rows[0].keys())
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    os.replace(path + ".tmp", path)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    dc = cfg["depth_noise_floor"]
    out_t = os.path.join(REPO, cfg["outputs"]["tables_dir"])
    out_root = os.path.join(REPO, dc["out_root"])
    conds, seeds = dc["conditions"], dc["seeds"]

    chk = subprocess.run(["git", "-C", REPO, "check-ignore", "-v", dc["out_root"] + "/x/depth_acc_fp16.npz"],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        sys.exit("ABORT: %s is not git-ignored" % dc["out_root"])

    import torch
    from nerf.datasets.blender import load_blender_data
    from nerf.rendering import render
    from nerf.training import _build_model, _render_kwargs, device
    from utils import render_eval as RE
    if device.type != "cuda":
        sys.exit("ABORT: CUDA not available; no CPU fallback")
    print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    es = RE.eval_set_status(REPO)
    print("eval set verify PASS, aggregate", es["aggregate_sha256"][:16], flush=True)

    m_dil = E.load_eval_data(REPO)["m_dil"]
    E.assert_partition(m_dil)
    _, poses, hwf, i_split = load_blender_data(os.path.join(REPO, "data", "blender_scenes"), half_res=True, testskip=1)
    poses = poses[i_split[2]]
    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
    assert poses.shape[0] == 75 and (H, W) == (400, 400)

    depth, sanity = {}, []
    for c in conds:
        for s in seeds:
            rid = "%s_seed%d" % (c, s)
            npz = os.path.join(out_root, rid, "depth_acc_fp16.npz")
            if os.path.exists(npz):                       # seed 1 from D-044 Part 3 (or an earlier run of this script)
                depth[(c, s)] = np.load(npz)["depth"]
                print("== %s: loaded existing float16 depth" % rid, flush=True)
                continue
            run_dir = os.path.join(REPO, "experiments", "runs", rid)
            summ = json.load(open(os.path.join(run_dir, "summary.json")))
            ckpt = os.path.join(run_dir, "%06d.tar" % int(summ["iterations"]))
            rm = json.load(open(os.path.join(REPO, dc["render_meta_template"].format(run_id=rid))))
            ck_sha = RE.sha256_file(ckpt)
            if ck_sha != rm["checkpoint_sha256"]:
                sys.exit("ABORT: %s checkpoint sha256 differs from render_meta.json" % rid)
            run_cfg = load_config(os.path.join(REPO, dc["config_dir"], os.path.basename(summ["config_path"])))
            run_cfg["reproducibility"]["seed"] = s
            ckd = torch.load(ckpt, map_location=device)
            model, model_fine, _, qfn, use_vd = _build_model(run_cfg)
            model.load_state_dict(ckd["network_fn_state_dict"])
            model_fine.load_state_dict(ckd["network_fine_state_dict"])
            model.eval(); model_fine.eval()
            kw = _render_kwargs(run_cfg, qfn, model, model_fine, use_vd, False)
            assert kw["perturb"] == 0.0 and kw["raw_noise_std"] == 0.0
            stored = np.load(os.path.join(REPO, dc["stored_renders_template"].format(run_id=rid)))["renders"]
            rgb_psnr, rgb_maxabs = [], []

            def on_view(i, rgb, stored=stored, rgb_psnr=rgb_psnr, rgb_maxabs=rgb_maxabs):
                d = rgb.astype(np.float64) - stored[i].astype(np.float64)
                mse = float((d ** 2).mean())
                rgb_psnr.append(float("inf") if mse == 0 else -10 * np.log10(mse))
                rgb_maxabs.append(float(np.abs(d).max()))

            print("== %s: checkpoint sha OK; rendering 75 views" % rid, flush=True)
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            render_fn = lambda c2w, chunk: render(H, W, K, chunk=chunk, c2w=c2w, **kw)[:3]
            d, a, chunk_used = DD.render_depth_acc(render_fn, poses, device, int(dc["chunk_size"]), int(dc["min_chunk_size"]),
                                                   on_view=on_view, log=lambda m: print(m, flush=True))
            od = os.path.join(out_root, rid)
            os.makedirs(od, exist_ok=True)
            np.savez_compressed(os.path.join(od, "depth_acc_fp16.npz.tmp.npz"), depth=d.astype(np.float16),
                                acc=a.astype(np.float16), view_index=np.arange(75))
            os.replace(os.path.join(od, "depth_acc_fp16.npz.tmp.npz"), npz)
            depth[(c, s)] = d.astype(np.float16)
            finite = [x for x in rgb_psnr if np.isfinite(x)]
            sanity.append({"run_id": rid, "chunk_used": chunk_used, "render_minutes": (time.time() - t0) / 60,
                           "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30, "checkpoint_sha256": ck_sha,
                           "rgb_vs_stored_mean_psnr_db": float(np.mean(finite)) if finite else float("inf"),
                           "rgb_vs_stored_max_abs_diff": float(np.max(rgb_maxabs))})
            print("   done %.1f min, rgb vs stored mean PSNR %.2f dB" % (sanity[-1]["render_minutes"],
                  sanity[-1]["rgb_vs_stored_mean_psnr_db"]), flush=True)
            del model, model_fine, ckd
            torch.cuda.empty_cache()

    f64 = lambda x: np.asarray(x, dtype=np.float64)
    pois = FU.read_manifest_poisoned(os.path.join(REPO, cfg["inputs"]["manifest_csv"]))
    rows = []
    for c in conds:
        for s1, s2 in itertools.combinations(seeds, 2):
            v = DD.region_mean_abs_diff(f64(depth[(c, s1)]), f64(depth[(c, s2)]), m_dil)
            same_data = (pois.get("%s_seed%d" % (c, s1)) == pois.get("%s_seed%d" % (c, s2))
                         if c != "control" and c != "C7" else True)
            rows.append({"kind": "same_condition_seed_pair", "condition": c, "a": "%s_seed%d" % (c, s1), "b": "%s_seed%d" % (c, s2),
                         "same_poisoned_view_set": bool(same_data), "n_views": 75, **DD.summarise(v)})
    for c in conds:                                            # the D-044 Part 3 quantity, recomputed on float16 for comparison
        if c == "control":
            continue
        v = DD.region_mean_abs_diff(f64(depth[(c, 1)]), f64(depth[("control", 1)]), m_dil)
        rows.append({"kind": "D044_cond_seed1_vs_control_seed1_float16", "condition": c, "a": c + "_seed1", "b": "control_seed1",
                     "same_poisoned_view_set": "", "n_views": 75, **DD.summarise(v)})
    d044 = {r["condition"]: float(r["mean"]) for r in csv.DictReader(open(os.path.join(REPO, dc["d044_part3_summary"])))
            if r["quantity"] == "mean_abs_depth_diff_vs_control"}
    for r in rows:
        if r["kind"].startswith("D044") and r["condition"] in d044:
            r["d044_float32_mean"] = d044[r["condition"]]
            r["float16_minus_float32"] = r["mean"] - d044[r["condition"]]
    allkeys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    rows = [{k: r.get(k, "") for k in allkeys} for r in rows]
    write_csv(os.path.join(out_t, "part5_depth_noise_floor.csv"), rows)
    if sanity:
        write_csv(os.path.join(out_t, "part5_render_sanity.csv"), sanity)
    for r in rows:
        print("%-45s %-14s %s vs %s  mean %.5f  [%.5f, %.5f]" % (r["kind"], r["condition"], r["a"], r["b"], r["mean"], r["min"], r["max"]))


if __name__ == "__main__":
    main()
