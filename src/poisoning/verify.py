"""Phase 5 gate checks for a built poisoned dataset (reused for Phase 9, D-045). Pure-ish: file I/O only through the
arguments. Raises nothing; returns a list of problems (empty = PASS)."""

import hashlib
import json
import os
from typing import Dict, List, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if not px:
        return mask.astype(bool)
    k = np.ones((2 * px + 1,) * 2, np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def check_image_pair(original: np.ndarray, poisoned: np.ndarray, mask_dil: np.ndarray, is_poisoned: bool) -> List[str]:
    """Outside the dilated mask the poisoned image equals the original (every pixel, every channel); inside, a poisoned
    image must differ somewhere (an untouched 'poisoned' view is a failure) and an unpoisoned one must not exist here."""
    problems = []
    if original.shape != poisoned.shape:
        return ["shape %s != %s" % (poisoned.shape, original.shape)]
    diff = np.any(original != poisoned, axis=-1) if original.ndim == 3 else original != poisoned
    if np.any(diff & ~mask_dil):
        problems.append("%d pixel(s) differ OUTSIDE the dilated mask" % int((diff & ~mask_dil).sum()))
    if is_poisoned and not diff.any():
        problems.append("flagged poisoned but identical to the original")
    if not is_poisoned and diff.any():
        problems.append("not flagged poisoned but differs from the original")
    return problems


def verify_dataset(dataset_dir: str, scene_root: str, manifest_rows: Sequence[Dict], expected_poisoned: int,
                   mask_dir: str, dilation_px: int, read_mask, expect_views: int = 100) -> Dict:
    """Run the gate checks. `manifest_rows`: this dataset's MANIFEST rows (dicts). Returns {problems, n_poisoned, n_files}."""
    problems = []
    rows = sorted(manifest_rows, key=lambda r: int(r["view_index"]))
    if len(rows) != expect_views:
        problems.append("manifest has %d rows, expected %d" % (len(rows), expect_views))
    flagged = [int(r["view_index"]) for r in rows if r["poisoned"] == "True"]
    if len(flagged) != expected_poisoned:
        problems.append("manifest flags %d views poisoned, expected exactly %d" % (len(flagged), expected_poisoned))
    tj = os.path.join(dataset_dir, "transforms_train.json")
    clean_tj = os.path.join(scene_root, "transforms_train.json")
    if _sha(tj) != _sha(clean_tj) and json.load(open(tj)) != json.load(open(clean_tj)):
        problems.append("transforms_train.json differs from the clean scene's")
    frames = json.load(open(tj))["frames"]
    for r in rows:
        i = int(r["view_index"])
        name = os.path.basename(frames[i]["file_path"])
        src = os.path.join(scene_root, "train", name + ".png")
        dst = os.path.join(dataset_dir, "train", name + ".png")
        is_p = i in set(flagged)
        if not is_p:
            if _sha(src) != _sha(dst):
                problems.append("view %d (not poisoned): file is not byte-identical to the original" % i)
            continue
        mask, _ = read_mask(os.path.join(mask_dir, name + ".png"))
        for p in check_image_pair(imageio.imread(src), imageio.imread(dst), dilate(mask, dilation_px), True):
            problems.append("view %d: %s" % (i, p))
    return {"problems": problems, "n_poisoned": len(flagged), "n_files": len(rows)}
