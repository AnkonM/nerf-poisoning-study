#!/usr/bin/env python
"""Phase 3 PoC: render 4 fixed held-out test views per condition, save
GT|rendered side-by-side PNGs, and report overall + masked (target-region)
PSNR.

PoC-signal only (docs/METHODOLOGY.md §6 metrics require the frozen
eval_holdout set, which doesn't exist until Phase 4) — supplementary to
`scripts/evaluate.py`, not a replacement: `evaluate.py`'s CLI only
evaluates the *entire* i_test split as one scalar, with no way to select
specific view indices or compute a masked/target-region PSNR, which this
step's report needed. Reuses the same model-loading/rendering functions
`scripts/evaluate.py` uses (`_build_model`/`_render_kwargs`/`render` from
src/nerf/training.py and src/nerf/rendering.py) rather than
reimplementing them. Same spirit as the ad hoc Phase 2 D-016 sample
renders, kept here (not `src/metrics/`) since it isn't a paper metric.

Usage:
    python scripts/phase3_poc_eval_views.py <config_path> <checkpoint_path> <run_id>

Writes experiments/results/phase3_poc/<run_id>_view*.png and
experiments/results/phase3_poc/<run_id>_4view_summary.json.
"""
import json
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nerf.datasets.blender import load_blender_data
from nerf.metrics import img2mse, mse2psnr
from nerf.rendering import render
from nerf.training import _build_model, _render_kwargs, device
from utils.config import load_config

FIXED_GLOBAL_INDICES = [200, 266, 333, 399]  # same as Phase 2 D-016, test-split views


def main():
    config_path, ckpt_path, run_id = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = load_config(config_path)

    dataset_cfg = cfg["dataset"]
    images_rgba, poses, hwf, i_split = load_blender_data(
        dataset_cfg["path"], half_res=dataset_cfg.get("half_res", False),
        testskip=dataset_cfg.get("testskip", 1),
    )
    white_bkgd = cfg["render"].get("white_background", True)
    alpha = images_rgba[..., 3]  # normalized to [0, 1] by load_blender_data's `/255.0`
    if white_bkgd:
        images = images_rgba[..., :3] * images_rgba[..., -1:] + (1.0 - images_rgba[..., -1:])
    else:
        images = images_rgba[..., :3]

    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    ckpt = torch.load(ckpt_path, map_location=device)
    model, model_fine, _, network_query_fn, use_viewdirs = _build_model(cfg)
    model.load_state_dict(ckpt["network_fn_state_dict"])
    if model_fine is not None and ckpt.get("network_fine_state_dict") is not None:
        model_fine.load_state_dict(ckpt["network_fine_state_dict"])
    model.eval()
    if model_fine is not None:
        model_fine.eval()
    render_kwargs_test = _render_kwargs(cfg, network_query_fn, model, model_fine, use_viewdirs, False)

    out_dir = "experiments/results/phase3_poc"
    os.makedirs(out_dir, exist_ok=True)

    results = {"run_id": run_id, "checkpoint": ckpt_path, "views": {}}
    overall_psnrs, masked_psnrs = [], []

    with torch.no_grad():
        for idx in FIXED_GLOBAL_INDICES:
            t0 = time.time()
            c2w = torch.Tensor(poses[idx]).to(device)
            target = torch.Tensor(images[idx]).to(device)
            rgb, _, _, _ = render(H, W, K, chunk=cfg["training"]["chunk_size"], c2w=c2w[:3, :4], **render_kwargs_test)

            mse = img2mse(rgb, target)
            full_psnr = mse2psnr(mse).item()

            # D-017's alpha threshold (127/255 of the raw uint8 PNG scale)
            # applied in load_blender_data's normalized [0, 1] scale.
            mask = alpha[idx] > (127.0 / 255.0)
            mask_t = torch.from_numpy(mask).to(device)
            rgb_masked = rgb[mask_t]
            target_masked = target[mask_t]
            masked_mse = img2mse(rgb_masked, target_masked)
            masked_psnr_val = mse2psnr(masked_mse).item()

            overall_psnrs.append(full_psnr)
            masked_psnrs.append(masked_psnr_val)

            gt_u8 = (target.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            rend_u8 = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            pair = np.concatenate([gt_u8, rend_u8], axis=1)
            fname = f"{run_id}_view{idx:03d}_psnr{full_psnr:.2f}.png"
            imageio.imwrite(os.path.join(out_dir, fname), pair)

            elapsed = time.time() - t0
            print(f"[{run_id}] view {idx}: full_psnr={full_psnr:.3f} dB masked_psnr={masked_psnr_val:.3f} dB "
                  f"({elapsed:.1f}s) -> {fname}")
            results["views"][idx] = {"full_psnr": full_psnr, "masked_psnr": masked_psnr_val, "file": fname}

    results["mean_full_psnr"] = float(np.mean(overall_psnrs))
    results["mean_masked_psnr"] = float(np.mean(masked_psnrs))
    print(f"[{run_id}] mean full PSNR: {results['mean_full_psnr']:.3f} dB "
          f"| mean masked (target-region) PSNR: {results['mean_masked_psnr']:.3f} dB")

    summary_path = os.path.join(out_dir, f"{run_id}_4view_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{run_id}] summary written to {summary_path}")


if __name__ == "__main__":
    main()
