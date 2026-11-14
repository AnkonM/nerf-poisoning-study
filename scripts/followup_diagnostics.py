#!/usr/bin/env python
"""Phase 8 follow-up diagnostics, Part 2 (plan of record: DECISION_LOG.md D-044). Descriptive tables and
figures from the existing D-042 CSVs only. Writes experiments/results/phase8/followup/ and paper/figures/
(fig9, fig10). Never writes per_run.csv, any D-042 output, results.csv, runs/, renders/ or data/.

Usage:  python scripts/followup_diagnostics.py configs/analysis/phase8_followup_d044.yaml
"""

import csv
import json
import os
import subprocess
import sys

import numpy as np

from metrics import aggregate as A
from metrics import figures as F
from metrics import followup as FU
from metrics import followup_figures as FF
from utils.config import load_config

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def git(*a):
    return subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True).stdout.strip()


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    P = lambda k: os.path.join(REPO, cfg["inputs"][k])
    out_t = os.path.join(REPO, cfg["outputs"]["tables_dir"])
    out_f = os.path.join(REPO, cfg["outputs"]["figures_dir"])
    conds, seeds, cols = cfg["conditions"], cfg["seeds"], cfg["metrics"]

    rows = A.read_per_run(P("per_run_csv"))
    assert len(rows) == len(conds) * len(seeds) and len({r["run_id"] for r in rows}) == len(rows)
    sha = A.sha256_file(P("per_run_csv"))
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "src/metrics", "scripts/followup_diagnostics.py",
                "configs/analysis").splitlines()
    run_ids = sorted(r["run_id"] for r in rows)
    pv = {r["run_id"]: FU.read_per_view(os.path.join(REPO, cfg["inputs"]["per_view_dir"], r["run_id"] + ".csv"))
          for r in rows}
    for rid, v in pv.items():
        assert len(v) == 75, rid

    def meta(columns, runs=None, extra=None):
        d = {"per_run_csv": cfg["inputs"]["per_run_csv"], "per_run_sha256": sha, "run_ids": sorted(runs or run_ids),
             "columns": list(columns), "git_commit": commit, "git_dirty_paths": dirty, "decision": "D-044"}
        d.update(extra or {})
        return d

    written = []

    # ---- 2a ------------------------------------------------------------------------------
    a, b = cfg["paired"]["a"], cfg["paired"]["b"]
    pois = FU.read_manifest_poisoned(P("manifest_csv"))
    FU.assert_paired_views(pois, a, b, seeds)                      # precondition; raises otherwise
    diffs = FU.paired_differences(rows, a, b, cols, seeds)
    write_csv(os.path.join(out_t, "2a_paired_%s_minus_%s.csv" % (b, a)), diffs)
    written.append("2a")
    print("2a: %s and %s poison identical view sets at seeds %s (asserted)" % (a, b, seeds))

    # ---- 2b ------------------------------------------------------------------------------
    grid, seed_sum, corr = FU.seed_grid(rows, cols, conds, seeds)
    write_csv(os.path.join(out_t, "2b_seed_condition_grid.csv"), grid)
    write_csv(os.path.join(out_t, "2b_seed_mean_residual.csv"), seed_sum)
    write_csv(os.path.join(out_t, "2b_seed_residual_correlation.csv"), corr)

    # ---- 2c ------------------------------------------------------------------------------
    tol = cfg["mse_aggregate"]["per_view_psnr_tolerance_db"]
    run_rows = []
    for r in rows:
        for region in cfg["mse_aggregate"]["regions"]:
            agg = FU.run_region_aggregates(pv[r["run_id"]], region)
            col = {"tgt": "tgt_psnr", "col": "col_psnr", "plate": "plate_psnr"}[region]
            if abs(agg["mean_of_psnr"] - r[col]) > tol:
                raise AssertionError("%s %s: mean of per-view PSNR %.12f != per_run.csv %.12f"
                                     % (r["run_id"], region, agg["mean_of_psnr"], r[col]))
            run_rows.append(dict(run_id=r["run_id"], condition=r["condition"], seed=r["seed"], region=region, **agg))
    write_csv(os.path.join(out_t, "2c_run_aggregates.csv"), run_rows)
    write_csv(os.path.join(out_t, "2c_condition_aggregates.csv"), FU.condition_aggregate_table(run_rows))
    flips = FU.separation_flips(run_rows, conds, cfg["mse_aggregate"]["regions"])
    write_csv(os.path.join(out_t, "2c_separation_statements.csv"), flips)
    print("2c: per-view PSNR means match per_run.csv; %d of %d pair statements flip under an alternative aggregate"
          % (sum(f["flips_vs_mean_of_psnr"] for f in flips), len(flips)))

    # ---- 2d ------------------------------------------------------------------------------
    ref = FU.read_reference(P("reference_csv"))
    d_rows = []
    for r in rows:
        if r["condition"] in cfg["farther_than_reference"]["conditions"]:
            d_rows.append(dict(run_id=r["run_id"], condition=r["condition"], seed=r["seed"],
                               **FU.farther_than_reference(pv[r["run_id"]], ref)))
    write_csv(os.path.join(out_t, "2d_farther_than_reference.csv"), d_rows)

    # ---- 2e ------------------------------------------------------------------------------
    s_rows, pooled, per_run = [], {}, {}
    for c in conds:
        per_run[c] = []
        allv = []
        for r in sorted((r for r in rows if r["condition"] == c), key=lambda r: r["seed"]):
            v = np.array([x["tgt_psnr"] for x in pv[r["run_id"]] if x["included"] == 1])
            st = FU.spread_stats(v)
            per_run[c].append(dict(st, seed=r["seed"]))
            s_rows.append(dict(scope="run", run_id=r["run_id"], condition=c, seed=r["seed"], **st))
            allv.extend(v.tolist())
        pooled[c] = FU.spread_stats(allv)
    ref_iqr = pooled[cfg["spread"]["reference_condition"]]["iqr"]
    for c in conds:
        s_rows.append(dict(scope="pooled_3_seeds", run_id="", condition=c, seed="", **pooled[c],
                           iqr_ratio_to_control=pooled[c]["iqr"] / ref_iqr))
    for r in s_rows:
        r.setdefault("iqr_ratio_to_control", "")
    write_csv(os.path.join(out_t, "2e_target_psnr_spread.csv"), s_rows)

    # ---- figures -------------------------------------------------------------------------
    figs = []
    figs += F.save_fig(FF.paired_figure(diffs, a, b, seeds,
                       "Fig 9. EXPLORATORY (D-044 2a, descriptive): per-seed paired differences, %s minus %s "
                       "(identical poisoned views per seed)" % (b, a)),
                       "fig9_paired_%s_minus_%s" % (b, a), out_f,
                       meta(cols, [r["run_id"] for r in rows if r["condition"] in (a, b)], {"table": "2a"}))
    figs += F.save_fig(FF.spread_figure(pooled, per_run, conds,
                       "Fig 10. EXPLORATORY (D-044 2e, descriptive): spread of per-view target-region PSNR by condition"),
                       "fig10_spread_per_view_target_psnr", out_f,
                       meta(["tgt_psnr"], None, {"table": "2e", "source": "per_view CSVs"}))
    print("per_run.csv sha256:", sha, "| commit:", commit, "| dirty:", dirty)
    print("tables ->", out_t)
    print("figures:", len(figs))


if __name__ == "__main__":
    main()
