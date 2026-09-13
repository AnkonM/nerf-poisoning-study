#!/usr/bin/env python
"""Build one poisoned Lego image set from a
configs/poisoning/phase3_poc_budget_*.yaml condition config.

Thin CLI entry point — logic lives in src/poisoning/{compositor,
view_selection}.py, per docs/PROJECT_STRUCTURE.md. This is Phase 3's
pipeline proof-of-concept only (docs/ROADMAP.md Phase 3); the config's
`dataset_source`/`mask_source`/`background_proxy` fields are scoped to the
Lego PoC per docs/DECISION_LOG.md D-017, not the Phase 4 final-scene
protocol.

Usage:
    python scripts/build_poison_set.py <config_path>

Writes data/poisoned/<condition_id>/train/*.png (full train split — clean
copies for unselected views, composited images for selected ones) plus
transforms_train.json (poses unchanged, copied as-is), and appends this
condition's rows to data/poisoned/MANIFEST.csv (per
docs/PROJECT_STRUCTURE.md's data/poisoned/ contract). Does not touch
val/ or test/ splits — no training happens in this step, so a full
trainable dataset directory (matching what scripts/train.py will
eventually expect) is a Phase 5 pipeline-build decision, not this step's.
"""

import argparse
import csv
import json
import os
import shutil
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poisoning.compositor import alpha_to_mask, hard_erasure
from poisoning.view_selection import sample_random
from utils.config import load_config

MANIFEST_PATH = os.path.join("data", "poisoned", "MANIFEST.csv")
MANIFEST_FIELDS = [
    "condition_id",
    "view_index",
    "file_path",
    "poisoned",
    "attack_type",
    "budget_percent",
    "seed",
    "mask_source",
    "mask_alpha_threshold",
    "background_proxy_method",
    "background_proxy_value",
]


def _rewrite_manifest_rows(condition_id: str, new_rows: list) -> None:
    """Replace any existing rows for `condition_id` with `new_rows`,
    keeping every other condition's rows — makes a re-run of this script
    idempotent/regeneratable, per PROJECT_STRUCTURE.md's requirement that
    a poisoned condition be fully reproducible from its config alone."""
    existing = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["condition_id"] != condition_id]

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in existing + new_rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a configs/poisoning/phase3_poc_budget_*.yaml file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    condition_id = cfg["condition_id"]
    src_dir = cfg["dataset_source"]
    out_dir = cfg["output_dir"]
    pcfg = cfg["poisoning"]

    with open(os.path.join(src_dir, "transforms_train.json")) as f:
        transforms = json.load(f)
    frames = transforms["frames"]
    n_views = len(frames)

    # V_target = all training views for Lego (D-017: single object, always
    # visible in every view).
    v_target = list(range(n_views))
    selected, num_poisoned = sample_random(v_target, pcfg["budget_percent"], pcfg["seed"])
    selected_set = set(selected)

    print(f"[{condition_id}] |V_target| = {n_views}, budget = {pcfg['budget_percent']}%, "
          f"num_poisoned = {num_poisoned}, seed = {pcfg['seed']}")
    print(f"[{condition_id}] selected view indices: {selected}")

    out_train_dir = os.path.join(out_dir, "train")
    os.makedirs(out_train_dir, exist_ok=True)

    bg_cfg = pcfg["background_proxy"]
    bg_rgb = np.asarray(bg_cfg["rgb"], dtype=np.float64)
    threshold = pcfg["mask_alpha_threshold"]

    manifest_rows = []
    for i, frame in enumerate(frames):
        rel_path = frame["file_path"] + ".png"  # e.g. "./train/r_0.png"
        src_path = os.path.join(src_dir, rel_path)
        fname = os.path.basename(rel_path)
        dst_path = os.path.join(out_train_dir, fname)

        is_poisoned = i in selected_set
        if is_poisoned:
            img = imageio.imread(src_path)
            mask = alpha_to_mask(img[..., 3], threshold=threshold)
            poisoned = hard_erasure(img, mask, bg_rgb)
            imageio.imwrite(dst_path, poisoned)
        else:
            shutil.copyfile(src_path, dst_path)

        manifest_rows.append({
            "condition_id": condition_id,
            "view_index": i,
            "file_path": os.path.join(out_train_dir, fname),
            "poisoned": is_poisoned,
            "attack_type": pcfg["attack_type"] if is_poisoned else "",
            "budget_percent": pcfg["budget_percent"],
            "seed": pcfg["seed"],
            "mask_source": pcfg["mask_source"] if is_poisoned else "",
            "mask_alpha_threshold": threshold if is_poisoned else "",
            "background_proxy_method": bg_cfg["method"] if is_poisoned else "",
            "background_proxy_value": bg_cfg["rgb"] if is_poisoned else "",
        })

    shutil.copyfile(
        os.path.join(src_dir, "transforms_train.json"),
        os.path.join(out_dir, "transforms_train.json"),
    )

    _rewrite_manifest_rows(condition_id, manifest_rows)
    print(f"[{condition_id}] wrote {n_views} images to {out_train_dir}")
    print(f"[{condition_id}] manifest rows written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
