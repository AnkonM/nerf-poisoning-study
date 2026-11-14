#!/usr/bin/env python
"""Phase 8 follow-up, Part 3 (plan of record: DECISION_LOG.md D-044): exploratory depth / opacity of the
fine network at seed 1 for control, C3, C5, C7. GPU only (no CPU fallback). Writes float16 arrays to
experiments/renders_diagnostics/ (git-ignored), tables to experiments/results/phase8/followup/ and
paper/figures/fig11. Never writes to experiments/renders/, runs/, data/, per_run.csv or results.csv.

Usage:  python scripts/depth_opacity_diagnostics.py configs/analysis/phase8_followup_d044.yaml
"""

import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

from metrics import aggregate as A
from metrics import depth_diag as DD
from metrics import eval_data as E
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
    dc = cfg["depth_opacity"]
    out_t = os.path.join(REPO, cfg["outputs"]["tables_dir"])
    out_f = os.path.join(REPO, cfg["outputs"]["figures_dir"])
    seed, conds, ref_cond = int(dc["seed"]), dc["conditions"], dc["reference_condition"]
    out_root = os.path.join(REPO, dc["out_root"])

    # gitignore guard: never let these arrays become trackable
    chk = subprocess.run(["git", "-C", REPO, "check-ignore", "-v", dc["out_root"] + "/x/depth_acc_fp16.npz"],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        sys.exit("ABORT: %s is not git-ignored" % dc["out_root"])
    print("gitignored:", chk.stdout.strip())

    import torch
    from nerf.datasets.blender import load_blender_data
    from nerf.rendering import render
    from nerf.training import _build_model, _render_kwargs, device
    from utils import render_eval as RE
    if device.type != "cuda":
        sys.exit("ABORT: CUDA not available; no CPU fallback")
    print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

    es = RE.eval_set_status(REPO)                           # raises unless freeze_eval_set.py --verify PASSes
    print("eval set verify PASS, aggregate", es["aggregate_sha256"][:16])

    data = E.load_eval_data(REPO)
    m_dil = data["m_dil"]
    E.assert_partition(m_dil)
    scene = os.path.join(REPO, "data", "blender_scenes")
    _, poses, hwf, i_split = load_blender_data(scene, half_res=True, testskip=1)
    poses = poses[i_split[2]]
    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
    assert poses.shape[0] == 75 and (H, W) == (400, 400)

    runs = ["%s_seed%d" % (c, seed) for c in conds]
    depth, acc, meta_runs, sanity = {}, {}, {}, []
    for c, rid in zip(conds, runs):
        run_dir = os.path.join(REPO, "experiments", "runs", rid)
        summ = json.load(open(os.path.join(run_dir, "summary.json")))
        ckpt = os.path.join(run_dir, "%06d.tar" % int(summ["iterations"]))
        rm = json.load(open(os.path.join(REPO, dc["render_meta_template"].format(run_id=rid))))
        ck_sha = RE.sha256_file(ckpt)
        if ck_sha != rm["checkpoint_sha256"]:
            sys.exit("ABORT: %s checkpoint sha256 differs from render_meta.json" % rid)
        run_cfg = load_config(os.path.join(REPO, dc["config_dir"], os.path.basename(summ["config_path"])))
        run_cfg["reproducibility"]["seed"] = seed
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
        depth[c], acc[c] = d, a
        peak = torch.cuda.max_memory_reserved() / 2 ** 30
        # float16 copies (never into experiments/renders/); statistics below use the float32 arrays
        od = os.path.join(out_root, rid)
        os.makedirs(od, exist_ok=True)
        np.savez_compressed(os.path.join(od, "depth_acc_fp16.npz.tmp.npz"), depth=d.astype(np.float16),
                            acc=a.astype(np.float16), view_index=np.arange(75))
        os.replace(os.path.join(od, "depth_acc_fp16.npz.tmp.npz"), os.path.join(od, "depth_acc_fp16.npz"))
        finite = [x for x in rgb_psnr if np.isfinite(x)]
        sanity.append({"run_id": rid, "condition": c, "chunk_used": chunk_used, "peak_vram_reserved_gib": peak,
                       "render_minutes": (time.time() - t0) / 60, "checkpoint_sha256": ck_sha,
                       "rgb_vs_stored_mean_psnr_db": float(np.mean(finite)) if finite else float("inf"),
                       "rgb_vs_stored_min_psnr_db": float(np.min(finite)) if finite else float("inf"),
                       "rgb_vs_stored_max_abs_diff": float(np.max(rgb_maxabs)),
                       "depth_min": float(d.min()), "depth_max": float(d.max()),
                       "acc_min": float(a.min()), "acc_max": float(a.max())})
        print("   done in %.1f min, chunk %d, peak %.2f GiB; rgb vs stored: mean PSNR %.2f dB, max |diff| %.4f"
              % (sanity[-1]["render_minutes"], chunk_used, peak, sanity[-1]["rgb_vs_stored_mean_psnr_db"],
                 sanity[-1]["rgb_vs_stored_max_abs_diff"]), flush=True)
        del model, model_fine, ckd
        torch.cuda.empty_cache()

    # ---- per-view statistics over M_dil (float32 arrays) ---------------------------------------
    pois = FU.read_manifest_poisoned(os.path.join(REPO, cfg["inputs"]["manifest_csv"]))
    ev_pos = A.positions_from_transforms(json.load(open(os.path.join(REPO, cfg["inputs"]["eval_transforms"])))["frames"])
    tr_pos = A.positions_from_transforms(json.load(open(os.path.join(REPO, cfg["inputs"]["train_transforms"])))["frames"])
    rows, per_view = [], {}
    for c, rid in zip(conds, runs):
        mean_acc = DD.region_mean(acc[c], m_dil)
        mean_depth = DD.region_mean(depth[c], m_dil)
        diff = DD.region_mean_abs_diff(depth[c], depth[ref_cond], m_dil)
        pv = pois.get(rid) or pois.get(c)       # C7's single dataset id in MANIFEST.csv is 'C7' (D-043 E2)
        near = (A.nearest_angular_distance_deg(ev_pos, tr_pos[sorted(pv)]) if pv else None)
        per_view[c] = {"mean_acc": mean_acc, "nearest_deg": near}
        if c != ref_cond:
            per_view[c]["depth_abs_diff"] = diff
        for v in range(75):
            rows.append({"run_id": rid, "condition": c, "view_index": v, "mean_acc_mdil": float(mean_acc[v]),
                         "mean_depth_mdil": float(mean_depth[v]),
                         "mean_abs_depth_diff_vs_%s" % ref_cond: float(diff[v]),
                         "nearest_poisoned_deg": float(near[v]) if near is not None else ""})
    write_csv(os.path.join(out_t, "part3_per_view_depth_opacity.csv"), rows)

    summ_rows, rel_rows = [], []
    for c, rid in zip(conds, runs):
        s = DD.summarise(per_view[c]["mean_acc"])
        summ_rows.append(dict(run_id=rid, condition=c, quantity="mean_acc_mdil", **s))
        if c != ref_cond:
            s = DD.summarise(per_view[c]["depth_abs_diff"])
            summ_rows.append(dict(run_id=rid, condition=c, quantity="mean_abs_depth_diff_vs_" + ref_cond, **s))
            near = per_view[c]["nearest_deg"]
            if near is not None:
                rel_rows.append({"run_id": rid, "condition": c, "bin_deg": "ALL", "n": 75,
                                 "mean_y": float(per_view[c]["depth_abs_diff"].mean()),
                                 "spearman_rho_depth_diff_vs_nearest_deg": DD.spearman_rho(near, per_view[c]["depth_abs_diff"]),
                                 "nearest_deg_min": float(near.min()), "nearest_deg_max": float(near.max())})
                for b in DD.bin_means(near, per_view[c]["depth_abs_diff"], dc["angle_bins_deg"]):
                    rel_rows.append({"run_id": rid, "condition": c, **b, "spearman_rho_depth_diff_vs_nearest_deg": "",
                                     "nearest_deg_min": "", "nearest_deg_max": ""})
    write_csv(os.path.join(out_t, "part3_summary.csv"), summ_rows)
    write_csv(os.path.join(out_t, "part3_depth_vs_angle.csv"), rel_rows)
    write_csv(os.path.join(out_t, "part3_render_sanity.csv"), sanity)

    sha = A.sha256_file(os.path.join(REPO, cfg["inputs"]["per_run_csv"]))
    meta = {"per_run_csv": cfg["inputs"]["per_run_csv"], "per_run_sha256": sha, "run_ids": runs,
            "columns": ["mean_acc_mdil", "mean_abs_depth_diff_vs_" + ref_cond, "nearest_poisoned_deg"],
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty_paths": git("status", "--porcelain", "--", "src/metrics", "scripts/depth_opacity_diagnostics.py",
                                   "configs/analysis").splitlines(),
            "decision": "D-044", "table": "part3", "seed": seed, "eval_set_aggregate_sha256": es["aggregate_sha256"],
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    figs = F.save_fig(FF.depth_figure(per_view, dc["angle_bins_deg"],
                      "Fig 11. EXPLORATORY (D-044 Part 3): accumulated opacity and depth difference from control, seed 1"),
                      "fig11_EXPLORATORY_depth_opacity", out_f, meta)
    print("tables ->", out_t, "| figure files:", len(figs))


if __name__ == "__main__":
    main()
