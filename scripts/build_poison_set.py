#!/usr/bin/env python
"""Build one poisoning condition's datasets from a configs/poisoning/*.yaml file.

Thin CLI entry point — logic lives in src/poisoning/{compositor,view_selection}.py
per docs/PROJECT_STRUCTURE.md.

Output, per (condition, selection seed):

    data/poisoned/<dataset_id>/
        train/r_000.png ...      100 images: composited where selected,
                                 byte-identical copies of the original otherwise
        transforms_train.json    poses copied unchanged from the scene

That is the WHOLE directory. There is deliberately no val/, no test/, and no
symlink: the val split and the frozen eval_holdout are read straight from their
single canonical location via the config's `dataset.val_path`/`test_path`, which
`load_blender_data` now accepts (DECISION_LOG.md D-028). This is the fix for
D-020 — Phase 3 emitted only train/ and had val/test symlinked in BY HAND, so a
condition was not reproducible from its config alone. It now is.

A condition with N seeds produces N datasets, because METHODOLOGY.md §8 ties
random view selection to the same seed as model initialisation (D-028). Budgets
with no randomness (0% control) and deterministic selection (strategic) collapse
to a single seed-invariant dataset.

Usage:
    python scripts/build_poison_set.py configs/poisoning/budget_20.yaml
    python scripts/build_poison_set.py configs/poisoning/budget_20.yaml --seed 1
"""

import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline.scene_validation import read_mask
from poisoning.compositor import composite
from poisoning.view_selection import sample_random, sample_strategic
from utils.config import load_config
from utils.dataset_id import dataset_id_for, is_seed_invariant

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "poisoned", "MANIFEST.csv")
MANIFEST_FIELDS = [
    "condition_id", "dataset_id", "selection_seed", "view_index", "file_path",
    "poisoned", "attack_type", "budget_percent", "view_selection", "alpha",
    "mask_source", "mask_dilation_px",
]


def _rewrite_manifest_rows(dataset_id, new_rows):
    """Replace this dataset_id's rows, keep every other dataset's.

    Keeps a re-run idempotent, per PROJECT_STRUCTURE.md's requirement that a
    poisoned condition be fully reproducible from its config alone.
    """
    existing = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, newline="") as f:
            rdr = csv.DictReader(f)
            # Refuse to rewrite a manifest written under a DIFFERENT schema.
            # Silently re-emitting foreign rows through this writer drops any
            # column it does not know about — which is exactly what happened to
            # the Phase 3 PoC rows once (see DECISION_LOG.md D-029). Fail loudly
            # instead; the Phase 3 PoC manifest now lives in its own file.
            if rdr.fieldnames and list(rdr.fieldnames) != MANIFEST_FIELDS:
                raise SystemExit(
                    "%s has a different schema:\n  found:    %s\n  expected: %s\n"
                    "Refusing to rewrite it — columns would be silently dropped."
                    % (MANIFEST_PATH, list(rdr.fieldnames), MANIFEST_FIELDS))
            existing = [r for r in rdr if r.get("dataset_id") != dataset_id]
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in existing + new_rows:
            w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def _load_areas(cameras_path):
    """view index -> recorded target mask area fraction (never recomputed, §2)."""
    cams = json.load(open(cameras_path))
    per_view = cams["v_target"]["per_view_area_fraction"]["train"]
    return {int(name.split("_")[1]): float(v) for name, v in per_view.items()}, cams


def build_one(cfg, seed, scene_root, cameras_path):
    condition_id = cfg["condition_id"]
    pcfg = cfg["poisoning"]
    budget = pcfg["budget_percent"]
    method = pcfg["view_selection"]
    attack = pcfg["attack_type"]
    alpha = pcfg.get("alpha")
    # Study-wide constant, read from the inherited scene config -- NOT duplicated
    # per condition (D-028 finding 3: a pre-Phase-4 stub carried a contradicting
    # value here).
    dilation = cfg["render"]["background_plate"]["mask_dilation_px"]

    areas, cams = _load_areas(cameras_path)
    v_target_names = cams["v_target"]["views"]
    v_target = sorted(int(n.split("_")[1]) for n in v_target_names)

    # Shared with the Phase 6 sweep harness (src/utils/dataset_id.py) so the
    # code that WRITES these datasets and the code that READS them cannot drift
    # into a mapping that is self-consistently wrong.
    seed_invariant = is_seed_invariant(budget, method)
    dataset_id = dataset_id_for(condition_id, budget, method, seed)

    if budget == 0:
        selected, n_poisoned = [], 0
    elif method == "random":
        selected, n_poisoned = sample_random(v_target, budget, seed)
    elif method == "strategic":
        selected, n_poisoned = sample_strategic(v_target, budget, areas)
    else:
        raise ValueError("unknown view_selection %r" % method)
    if attack == "soft_suppression" and not alpha:
        raise ValueError("%s: soft_suppression requires a non-null alpha" % condition_id)
    blend_alpha = float(alpha) if attack == "soft_suppression" else 0.0
    selected_set = set(selected)

    out_dir = os.path.join(REPO_ROOT, "data", "poisoned", dataset_id)
    out_train = os.path.join(out_dir, "train")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)          # regenerate from scratch, never patch
    os.makedirs(out_train)

    transforms = json.load(open(os.path.join(scene_root, "transforms_train.json")))
    frames = transforms["frames"]

    print("[%s] budget=%s%% method=%s attack=%s alpha=%s seed=%s -> %d/%d views poisoned"
          % (dataset_id, budget, method, attack, alpha, seed if not seed_invariant else "n/a",
             n_poisoned, len(v_target)))

    kernel = np.ones((2 * dilation + 1,) * 2, np.uint8)
    rows = []
    for i, frame in enumerate(frames):
        name = os.path.basename(frame["file_path"])          # r_000
        src = os.path.join(scene_root, "train", name + ".png")
        dst = os.path.join(out_train, name + ".png")

        if i in selected_set:
            original = imageio.imread(src)
            plate = imageio.imread(
                os.path.join(REPO_ROOT, "data", "background_plates", "train", name + ".png"))[..., :3]
            mask, _ = read_mask(os.path.join(REPO_ROOT, "data", "masks", "train", name + ".png"))
            if dilation:
                mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            poisoned = composite(original, mask.astype(np.float64), plate, alpha=blend_alpha)
            imageio.imwrite(dst, poisoned)
        else:
            shutil.copyfile(src, dst)    # byte-identical, not re-encoded

        rows.append({
            "condition_id": condition_id, "dataset_id": dataset_id,
            "selection_seed": "" if seed_invariant else seed,
            "view_index": i, "file_path": os.path.relpath(dst, REPO_ROOT),
            "poisoned": i in selected_set,
            "attack_type": attack if i in selected_set else "",
            "budget_percent": budget, "view_selection": method,
            "alpha": blend_alpha if i in selected_set else "",
            "mask_source": "blender_object_id" if i in selected_set else "",
            "mask_dilation_px": dilation if i in selected_set else "",
        })

    json.dump(transforms, open(os.path.join(out_dir, "transforms_train.json"), "w"), indent=1)
    _rewrite_manifest_rows(dataset_id, rows)
    return dataset_id, n_poisoned, selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=None,
                    help="build only this seed (default: every seed in the config)")
    ap.add_argument("--scene-root", default=os.path.join(REPO_ROOT, "data", "blender_scenes"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    cameras_path = os.path.join(args.scene_root, "cameras.json")
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]

    built = []
    for s in seeds:
        dataset_id, n, sel = build_one(cfg, s, args.scene_root, cameras_path)
        built.append(dataset_id)
        if is_seed_invariant(cfg["poisoning"]["budget_percent"],
                             cfg["poisoning"]["view_selection"]):
            break        # seed-invariant: one dataset covers every seed
    print("[%s] built: %s" % (cfg["condition_id"], ", ".join(built)))


if __name__ == "__main__":
    main()
