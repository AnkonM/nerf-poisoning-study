#!/usr/bin/env python
"""Evaluate a trained NeRF checkpoint against its config's test set.

Thin CLI entry point — logic lives in src/nerf/training.py, per
docs/PROJECT_STRUCTURE.md. Unlike scripts/train.py, this does not train:
it loads an existing checkpoint's weights and computes PSNR over the
held-out test set, with per-image progress logging (see the
`evaluate_psnr(..., show_progress=True)` addition, added after the Phase 2
incident where a silent multi-hour eval phase was indistinguishable from
a hang — see docs/DECISION_LOG.md).

Usage:
    python scripts/evaluate.py <config_path> <checkpoint_path> [--run-id RUN_ID]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from nerf.datasets.blender import load_blender_data
from nerf.training import _build_model, _render_kwargs, device, evaluate_psnr
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the config YAML used to train the checkpoint")
    parser.add_argument("checkpoint", help="Path to a .tar checkpoint file")
    parser.add_argument("--run-id", default=None, help="Label for the output summary file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = args.run_id or os.path.splitext(os.path.basename(args.checkpoint))[0]

    dataset_cfg = cfg["dataset"]
    print(f"Loading dataset: {dataset_cfg['path']} (testskip={dataset_cfg.get('testskip', 1)})")
    images, poses, hwf, i_split = load_blender_data(
        dataset_cfg["path"],
        half_res=dataset_cfg.get("half_res", False),
        testskip=dataset_cfg.get("testskip", 1),
        val_dir=dataset_cfg.get("val_path"),
        test_dir=dataset_cfg.get("test_path"),
    )
    _, _, i_test = i_split
    print(f"Test-set size: {len(i_test)} views")

    white_bkgd = cfg["render"].get("white_background", True)
    if white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model, model_fine, _, network_query_fn, use_viewdirs = _build_model(cfg)
    model.load_state_dict(ckpt["network_fn_state_dict"])
    if model_fine is not None and ckpt.get("network_fine_state_dict") is not None:
        model_fine.load_state_dict(ckpt["network_fine_state_dict"])
    model.eval()
    if model_fine is not None:
        model_fine.eval()

    render_kwargs_test = _render_kwargs(cfg, network_query_fn, model, model_fine, use_viewdirs, False)

    print(f"Evaluating {len(i_test)} test views with per-image progress logging...")
    t0 = time.time()
    test_psnr = evaluate_psnr(
        images, poses, i_test, hwf, K, cfg["training"]["chunk_size"], render_kwargs_test, show_progress=True
    )
    elapsed = time.time() - t0
    print(f"\nFinal test-set PSNR: {test_psnr:.3f} dB over {len(i_test)} views, {elapsed / 60:.1f} min")

    out_dir = os.path.dirname(args.checkpoint)
    summary = {
        "checkpoint": args.checkpoint,
        "config_path": args.config,
        "test_psnr": test_psnr,
        "n_test_views": len(i_test),
        "elapsed_minutes": elapsed / 60,
    }
    summary_path = os.path.join(out_dir, f"eval_{run_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
