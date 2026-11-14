#!/usr/bin/env python
"""Phase 5 gate checks on already-built poisoned datasets of one condition config (reused for Phase 9, D-045/D-046).

Per seed: MANIFEST view counts exact; outside-mask pixel identity against the originals for EVERY image (byte-identical
files for unpoisoned views); poisoned images differ only inside the dilated mask; the loader's camera_angle_x check
passes (load_blender_data on the dataset with the canonical val/test paths); the E1 coverage statistic of the built set.
Read-only except for its report CSV.

Usage:  python scripts/verify_poison_set.py configs/poisoning/budget_100.yaml [report.csv]
"""

import csv
import json
import os
import sys

import numpy as np

from data_pipeline.scene_validation import read_mask
from metrics import aggregate as A
from metrics import phase9_evidence as P9
from nerf.datasets.blender import load_blender_data
from poisoning import verify as V
from poisoning.view_selection import num_poisoned_for_budget
from utils.config import load_config
from utils.dataset_id import dataset_id_for

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    scene = os.path.join(REPO, "data", "blender_scenes")
    pc = cfg["poisoning"]
    dil = cfg["render"]["background_plate"]["mask_dilation_px"]
    expected = num_poisoned_for_budget(pc["budget_percent"], 100)
    manifest = list(csv.DictReader(open(os.path.join(REPO, "data", "poisoned", "MANIFEST.csv"))))
    ev_pos = A.positions_from_transforms(json.load(open(os.path.join(scene, "transforms_test.json")))["frames"])
    tr_pos = A.positions_from_transforms(json.load(open(os.path.join(scene, "transforms_train.json")))["frames"])
    out, failed = [], False
    for s in cfg["seeds"]:
        did = dataset_id_for(cfg["condition_id"], pc["budget_percent"], pc["view_selection"], s)
        rows = [r for r in manifest if r["dataset_id"] == did]
        res = V.verify_dataset(os.path.join(REPO, "data", "poisoned", did), scene, rows, expected,
                               os.path.join(REPO, "data", "masks", "train"), dil, read_mask)
        problems = list(res["problems"])
        try:
            _, _, hwf, _ = load_blender_data(os.path.join(REPO, "data", "poisoned", did), half_res=True, testskip=1,
                                             val_dir=os.path.join(REPO, cfg["dataset"]["val_path"]),
                                             test_dir=os.path.join(REPO, cfg["dataset"]["test_path"]))
            loader = "PASS (focal %.4f)" % hwf[2]
        except Exception as e:  # noqa: BLE001
            loader = "FAIL: %s" % e
            problems.append("loader: %s" % e)
        poisoned = sorted(int(r["view_index"]) for r in rows if r["poisoned"] == "True")
        e1 = P9.e1_stats(A.nearest_angular_distance_deg(ev_pos, tr_pos[poisoned]))
        out.append({"dataset_id": did, "seed": s, "n_manifest_rows": len(rows), "n_poisoned": len(poisoned),
                    "expected_poisoned": expected, "problems": len(problems), "loader_camera_angle_x": loader,
                    "e1_mean_deg": e1["mean_deg"], "e1_median_deg": e1["median_deg"], "e1_max_deg": e1["max_deg"]})
        print("%-12s manifest %d rows, %d/%d poisoned | gate problems: %d | loader %s | E1 mean/median/max %.3f / %.3f / %.3f deg"
              % (did, len(rows), len(poisoned), expected, len(problems), loader, e1["mean_deg"], e1["median_deg"], e1["max_deg"]))
        for p in problems[:10]:
            print("   PROBLEM:", p)
        failed |= bool(problems)
    if len(sys.argv) == 3:
        with open(sys.argv[2], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            for r in out:
                w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    print("GATE:", "FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
