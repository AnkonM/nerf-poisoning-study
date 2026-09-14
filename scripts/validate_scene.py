#!/usr/bin/env python3
"""Phase 4 Step 5: compute |V_target|, validate the dataset, report the gate.

Thin CLI entry point; logic is in src/data_pipeline/scene_validation.py.

Prints an explicit PASS/FAIL for the ROADMAP.md Phase 4 gate and exits non-zero
on FAIL, so this cannot be "mostly passed".

Usage:
    python scripts/validate_scene.py configs/scenes/final_scene.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json  # noqa: E402

from data_pipeline.scene_validation import validate, write_v_target  # noqa: E402
from utils.config import load_config  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--write", action="store_true",
                    help="record |V_target| and per-view areas into cameras.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cameras_path = args.cameras or os.path.join(REPO_ROOT, "data",
                                                "blender_scenes", "cameras.json")
    data_root = args.data_root or os.path.join(REPO_ROOT, "data")
    cams = json.load(open(cameras_path))

    res, errors = validate(data_root, cams, cfg)

    print("=== Dataset validation ===")
    print("mask dtypes seen                : %s" % ", ".join(res["mask_dtypes"]))
    print("far-field max |original-plate|  : %.0f/255 (%s)"
          % (res["far_field_worst"], res["far_field_worst_view"]))
    print("handle hole open after %d px dil : %d views (closed in %d)"
          % (cfg["render"]["background_plate"]["mask_dilation_px"],
             res["handle_holes_open_after_dilation"],
             res["handle_holes_closed_by_dilation"]))
    print()
    print("=== V_target (METHODOLOGY.md §2, training views only) ===")
    print("visibility threshold (locked D-022): %.4f (%.2f%% of frame)"
          % (res["threshold"], 100 * res["threshold"]))
    print("train mask area: min %.4f  median %.4f  max %.4f"
          % (res["train_area_min"], res["train_area_median"], res["train_area_max"]))
    print("smallest margin over threshold    : %.2fx" % res["margin_min_x"])
    print("|V_target| = %d of %d training views" % (res["v_target_count"], res["n_train"]))
    print()

    print("=== Phase 4 gate (ROADMAP.md) ===")
    checks = []

    sep = res["v_target_count"] > 0 and not any("not binary" in e or "empty" in e
                                                for e in errors)
    checks.append(("target separable from background by mask", sep))

    n5 = round(0.05 * res["v_target_count"])
    checks.append(("round(0.05 x |V_target|) >= 1  [= %d]" % n5, n5 >= 1))
    checks.append(("...and lands on several views, not exactly 1", n5 > 1))

    checks.append(("all validation checks clean", not errors))

    for label, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", label))

    if errors:
        print()
        print("=== %d validation error(s) ===" % len(errors))
        for e in errors[:25]:
            print("  - %s" % e)
        if len(errors) > 25:
            print("  ... and %d more" % (len(errors) - 25))

    overall = all(ok for _, ok in checks)
    print()
    print("GATE: %s" % ("PASS" if overall else "FAIL"))

    if args.write:
        if not overall:
            print("refusing to write V_target into cameras.json on a FAILing gate")
            return 1
        write_v_target(cameras_path, res)
        print("recorded |V_target| = %d and per-view areas into %s"
              % (res["v_target_count"], os.path.relpath(cameras_path, REPO_ROOT)))

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
