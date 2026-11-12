#!/usr/bin/env python
"""Phase 8 analysis (plan of record: DECISION_LOG.md D-043). Reads the D-042 metrics and writes
tables to experiments/results/phase8/ and figures to paper/figures/. Computes NO new metric
on the frozen data beyond the pre-registered E3 band errors; never writes to results.csv, the
D-042 metrics outputs, runs/, renders/ or data/.

Usage:  python scripts/analyze_phase8.py
"""

import csv
import glob
import json
import os
import subprocess
import sys

import cv2
import imageio.v2 as iio
import numpy as np

from metrics import aggregate as A
from metrics import eval_data as E
from metrics import figures as F
from metrics import image_metrics as M

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MET = os.path.join(REPO, "experiments", "results", "metrics")
OUT = os.path.join(REPO, "experiments", "results", "phase8")
FIG = os.path.join(REPO, "paper", "figures")
PER_RUN = os.path.join(MET, "per_run.csv")


def git(*a):
    return subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True).stdout.strip()


def write_csv(path, rows, cols=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = cols or list(rows[0].keys())
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    os.replace(path + ".tmp", path)


def per_view(run_id):
    with open(os.path.join(MET, "per_view", "%s.csv" % run_id), newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    rows = A.read_per_run(PER_RUN)
    assert len(rows) == 24 and len({r["run_id"] for r in rows}) == 24
    sha = A.sha256_file(PER_RUN)
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "src/metrics", "scripts/analyze_phase8.py").splitlines()
    run_ids = sorted(r["run_id"] for r in rows)
    rid = {(r["condition"], r["seed"]): r["run_id"] for r in rows}

    def meta(columns, runs=None, extra=None):
        d = {"per_run_csv": "experiments/results/metrics/per_run.csv", "per_run_sha256": sha,
             "run_ids": sorted(runs or run_ids), "columns": list(columns), "git_commit": commit,
             "git_dirty_paths": dirty, "decision": "D-043"}
        d.update(extra or {})
        return d

    # ---- tables: A, B ---------------------------------------------------------------
    write_csv(os.path.join(OUT, "condition_summary.csv"), A.condition_summary(rows))
    write_csv(os.path.join(OUT, "pair_comparisons.csv"), A.pair_comparisons(rows))

    # ---- shared data ---------------------------------------------------------------------
    pv = {r["run_id"]: per_view(r["run_id"]) for r in rows}
    data = E.load_eval_data(REPO)
    E.assert_partition(data["m_dil"])
    ref = list(csv.DictReader(open(os.path.join(MET, "reference_gt_vs_plate.csv"))))
    gt_plate_mean = float(np.mean([float(r["gt_vs_plate_psnr"]) for r in ref]))

    # ---- E1 -------------------------------------------------------------------------------
    pooled = {"tgt_psnr": {}, "plate_psnr": {}}
    e1 = []
    for c in A.CONDITIONS:
        for metric in pooled:
            v = np.array([x[metric] for s in (1, 2, 3) for x in pv[rid[(c, s)]]])
            pooled[metric][c] = v
            e1.append(dict(condition=c, metric=metric, **A.distribution_stats(v)))
    write_csv(os.path.join(OUT, "E1_per_view_distribution.csv"), e1)

    # ---- E2 -------------------------------------------------------------------------------
    scene = os.path.join(REPO, "data", "blender_scenes")
    ev_pos = A.positions_from_transforms(json.load(open(os.path.join(scene, "transforms_test.json")))["frames"])
    tr_pos = A.positions_from_transforms(json.load(open(os.path.join(scene, "transforms_train.json")))["frames"])
    az, el = A.azimuth_elevation_deg(ev_pos)
    manifest = list(csv.DictReader(open(os.path.join(REPO, "data", "poisoned", "MANIFEST.csv"))))
    e2, pts = [], {}
    for c in ("C3", "C7"):
        pts[c] = {k: [] for k in ("az", "el", "nearest", "psnr", "seed")}
        for s in (1, 2, 3):
            ds = "C7" if c == "C7" else "%s_seed%d" % (c, s)
            pois = sorted(int(m["view_index"]) for m in manifest if m["dataset_id"] == ds and m["poisoned"] == "True")
            if len(pois) != 20:
                raise ValueError("%s: expected 20 poisoned views, found %d" % (ds, len(pois)))
            near = A.nearest_angular_distance_deg(ev_pos, tr_pos[pois])
            for i, x in enumerate(pv[rid[(c, s)]]):
                e2.append(dict(run_id=rid[(c, s)], condition=c, seed=s, view_index=i, azimuth_deg=float(az[i]),
                               elevation_deg=float(el[i]), nearest_poisoned_deg=float(near[i]),
                               tgt_psnr=x["tgt_psnr"], area_fraction_800=x["area_fraction_800"]))
                for k, v in (("az", az[i]), ("el", el[i]), ("nearest", near[i]), ("psnr", x["tgt_psnr"]), ("seed", s)):
                    pts[c][k].append(v)
        pts[c] = {k: np.array(v) for k, v in pts[c].items()}
    write_csv(os.path.join(OUT, "E2_per_view_geometry.csv"), e2)

    # ---- E3 -------------------------------------------------------------------------------
    labels = A.band_labels()
    bands = [A.band_masks(data["m_dil"][i]) for i in range(E.N_VIEWS)]
    e3, band_table = [], {}
    for c in A.CONDITIONS:
        per_seed = []
        for s in (1, 2, 3):
            run = rid[(c, s)]
            renders = E.load_renders(os.path.join(REPO, "experiments", "renders", run, "renders_fp16.npz"))
            mse = np.full((E.N_VIEWS, len(labels)), np.nan)
            for i in range(E.N_VIEWS):
                mse[i] = A.band_mse(M.sq_err_map(renders[i], data["gt"][i]), bands[i])
            used = (~np.isnan(mse)).sum(0)
            m = np.nanmean(mse, axis=0)
            per_seed.append(m)
            for b, lab in enumerate(labels):
                e3.append(dict(run_id=run, condition=c, seed=s, band_px=lab, mean_over_views_mse=float(m[b]),
                               n_views_used=int(used[b]), n_views_skipped=int(E.N_VIEWS - used[b]),
                               mean_band_pixels=float(np.mean([bands[i][b].sum() for i in range(E.N_VIEWS)]))))
        band_table[c] = np.array(per_seed)
    write_csv(os.path.join(OUT, "E3_collateral_band_mse.csv"), e3)

    # ---- E4 -------------------------------------------------------------------------------
    from tensorboard.backend.event_processing import event_accumulator as ea
    e4 = []
    for r in rows:
        d = os.path.join(REPO, "experiments", "runs", r["run_id"])
        s = json.load(open(os.path.join(d, "summary.json")))
        acc = ea.EventAccumulator(os.path.join(d, "tb"), size_guidance={"scalars": 0})
        acc.Reload()
        last = acc.Scalars("val/psnr_subset")[-1]
        e4.append(dict(run_id=r["run_id"], condition=r["condition"], seed=r["seed"], resumed=r["resumed"],
                       final_train_loss=float(s["final_train_loss"]), final_val_psnr_subset_step=int(last.step),
                       final_val_psnr_subset=float(last.value), col_psnr=r["col_psnr"], tgt_psnr=r["tgt_psnr"]))
    write_csv(os.path.join(OUT, "E4_loss_val_vs_collateral.csv"), e4)

    # ---- E5 -------------------------------------------------------------------------------
    e5 = []
    for col in A.METRIC_COLS:
        ref_vals = [A.values(rows, "control", col)[i] for i in (0, 1)]
        v3 = A.values(rows, "control", col)[2]
        e5.append(dict(metric=col, control_seed1=ref_vals[0], control_seed2=ref_vals[1], control_seed3_resumed=v3,
                       seeds12_min=min(ref_vals), seeds12_max=max(ref_vals),
                       inside_seeds12_range=bool(min(ref_vals) <= v3 <= max(ref_vals))))
    write_csv(os.path.join(OUT, "E5_resumed_run.csv"), e5)

    # ---- figures 1-5 ----------------------------------------------------------------------
    bc = [r["run_id"] for r in rows if r["condition"] in A.BUDGET_CONDITIONS]
    cols_t = ["tgt_%s" % k for k in ("psnr", "ssim", "lpips")] + ["rawtgt_%s" % k for k in ("psnr", "ssim", "lpips")]
    out = []
    out += F.save_fig(F.budget_figure(rows, "tgt", "Fig 1. Target-region fidelity to clean GT vs poisoning budget "
                      "(hard erasure, random selection; control = budget 0)", raw_prefix="rawtgt"),
                      "fig1_target_vs_budget", FIG, meta(cols_t, bc))
    cols_c = ["col_%s" % k for k in ("psnr", "ssim", "lpips")]
    out += F.save_fig(F.budget_figure(rows, "col", "Fig 2. Collateral-region fidelity to clean GT vs poisoning budget "
                      "(hard erasure, random selection)"), "fig2_collateral_vs_budget", FIG, meta(cols_c, bc))
    out += F.save_fig(F.plate_figure(rows, gt_plate_mean, "Fig 3. Distance of the target region to the background plate "
                      "vs poisoning budget"), "fig3_plate_vs_budget", FIG,
                      meta(["plate_psnr"], bc, {"reference": "experiments/results/metrics/reference_gt_vs_plate.csv",
                                                "reference_mean_gt_vs_plate_psnr": gt_plate_mean}))
    pcols = ["tgt_psnr", "col_psnr", "plate_psnr"]
    ylims = {}
    for col in pcols:
        v = [x for c in ("C3", "C6", "C7") for x in A.values(rows, c, col)]
        # layout only: include the mean +/- sample-std error bars so they are not clipped by the axes
        for c in ("C3", "C6", "C7"):
            vc = A.values(rows, c, col)
            m, s = float(np.mean(vc)), float(np.std(vc, ddof=1))
            v += [m - s, m + s]
        pad = 0.08 * (max(v) - min(v))
        ylims[col] = (min(v) - pad, max(v) + pad)
    for name, (ca, cb), fn in (("hard_vs_soft", A.PAIRS["hard_vs_soft"], "fig4_C3_vs_C6"),
                               ("random_vs_strategic", A.PAIRS["random_vs_strategic"], "fig5_C3_vs_C7")):
        runs = [r["run_id"] for r in rows if r["condition"] in (ca, cb)]
        ttl = ("Fig %s. %s vs %s (both 20%% budget, %s): %s" % (
            fn[3], ca, cb, "hard erasure" if name == "random_vs_strategic" else "random selection",
            "hard erasure vs soft suppression (alpha 0.4)" if name == "hard_vs_soft" else "random vs strategic selection"))
        out += F.save_fig(F.pair_figure(rows, ca, cb, pcols, ylims, ttl), fn, FIG, meta(pcols, runs))

    # ---- exploratory figures --------------------------------------------------------------
    out += F.save_fig(F.e1_figure(pooled), "fig8a_E1_exploratory_distributions", FIG,
                      meta(["tgt_psnr", "plate_psnr"], run_ids, {"source": "per_view CSVs"}))
    out += F.save_fig(F.e2_figure(pts), "fig8b_E2_exploratory_C3_vs_C7_geometry", FIG,
                      meta(["tgt_psnr"], [r["run_id"] for r in rows if r["condition"] in ("C3", "C7")],
                           {"source": "per_view CSVs + transforms + MANIFEST.csv"}))
    out += F.save_fig(F.e3_figure(band_table, labels), "fig8c_E3_exploratory_collateral_bands", FIG,
                      meta(["band MSE (renders vs GT)"], run_ids, {"source": "renders_fp16.npz + frozen GT, E3 bands"}))

    # ---- qualitative grid (rule D) -----------------------------------------------------------
    areas = np.array([x["area_fraction_800"] for x in pv[rid[("control", 1)]]])
    order = sorted(range(E.N_VIEWS), key=lambda i: (areas[i], i))
    views = [order[int(round(q * 74))] for q in (0.1, 0.5, 0.9)]
    s1 = {c: E.load_renders(os.path.join(REPO, "experiments", "renders", rid[(c, 1)], "renders_fp16.npz"))
          for c in A.CONDITIONS}
    col_titles = ["eval view %d (raw target area %.2f%%)" % (v, 100 * areas[v]) for v in views]
    row_labels = ["clean GT", "background plate"] + ["%s (seed 1)" % c for c in A.CONDITIONS]

    def crop(i):
        ys, xs = np.where(data["m_dil"][i])
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        side = max(ys.max() - ys.min(), xs.max() - xs.min()) + 41
        y0 = int(np.clip(cy - side // 2, 0, E.H_EVAL - side)); x0 = int(np.clip(cx - side // 2, 0, E.H_EVAL - side))
        return slice(y0, y0 + side), slice(x0, x0 + side)

    full = [[data["gt"][v] for v in views], [data["plates"][v] for v in views]] + [[s1[c][v] for v in views] for c in A.CONDITIONS]
    close = []
    for rowimgs in full:
        close.append([img[crop(v)] for img, v in zip(rowimgs, views)])
    qrun = [rid[(c, 1)] for c in A.CONDITIONS]
    qextra = {"eval_views": views, "selection_rule": "sorted by (raw target area, view index); index round(q*74) for q=0.1,0.5,0.9",
              "seed": 1}
    out += F.save_fig(F.image_grid(full, row_labels, col_titles,
                      "Fig 6a. Eval-view renders, seed 1 of every condition (views at the 10th/50th/90th percentile of raw target mask area)"),
                      "fig6a_qualitative_full", FIG, meta(["renders_fp16.npz"], qrun, qextra))
    out += F.save_fig(F.image_grid(close, row_labels, col_titles,
                      "Fig 6b. Target-region close-ups (M_dil bounding box + 20 px), same views, seed 1; plate row = ground truth for suppression"),
                      "fig6b_qualitative_closeups", FIG, meta(["renders_fp16.npz"], qrun, qextra))

    # ---- shadow retention (rule D) -------------------------------------------------------------
    cams = json.load(open(os.path.join(scene, "cameras.json")))
    tr_area = {int(k.split("_")[1]): float(v) for k, v in cams["v_target"]["per_view_area_fraction"]["train"].items()}
    pois = sorted(int(m["view_index"]) for m in manifest if m["dataset_id"] == "C5_seed1" and m["poisoned"] == "True")
    ordp = sorted(pois, key=lambda i: (tr_area[i], i))
    vsel = ordp[(len(ordp) - 1) // 2]
    name = "r_%03d.png" % vsel

    def rd(p):
        im = (iio.imread(p) / 255.0).astype(np.float32)
        return E.composite_white(im) if im.shape[-1] == 4 else im

    orig = rd(os.path.join(scene, "train", name))
    pz = rd(os.path.join(REPO, "data", "poisoned", "C5_seed1", "train", name))
    pl = rd(os.path.join(REPO, "data", "background_plates", "train", name))
    fig = F.image_grid([[orig, pz, pl]], ["training view %d" % vsel],
                       ["original (clean)", "poisoned (C5 seed 1, hard erasure)", "background plate"],
                       "Fig 7. Shadow retention (expected, D-022 Finding 4; METHODOLOGY §3 'Scope of the edit'): "
                       "the edit is limited to the mask, so the cast shadow outside it remains.\n"
                       "View = median raw mask area among C5 seed 1's %d poisoned views." % len(pois))
    out += F.save_fig(fig, "fig7_shadow_retention", FIG,
                      meta(["image files"], ["C5_seed1"], {"training_view": vsel, "n_poisoned_views_C5_seed1": len(pois),
                                                           "raw_area_fraction_800": tr_area[vsel]}))
    print("tables:", sorted(os.listdir(OUT)))
    print("figure files:", len(out))
    print("per_run.csv sha256:", sha, "| commit:", commit, "| dirty:", dirty)
    print("qualitative views:", views, "| shadow view:", vsel)


if __name__ == "__main__":
    sys.exit(main())
