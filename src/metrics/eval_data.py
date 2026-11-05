"""Loaders for the frozen eval data used by the §6 metrics (D-042 definitions A-C).

Read-only: nothing here writes to data/. GT images go through the training/eval
loader itself (`nerf.datasets.blender.load_blender_data`, half_res=True ->
cv2.INTER_AREA 800->400) and are then composited on white exactly as
scripts/evaluate.py does. Plates get the same operations. Masks are read with
`read_mask`, dilated with the poisoning pipeline's operation, then downscaled
with INTER_AREA and thresholded at 0.5 (never nearest-neighbour).
"""

import os
from typing import Dict

import cv2
import imageio.v2 as iio
import numpy as np

from data_pipeline.scene_validation import read_mask
from nerf.datasets.blender import load_blender_data
from utils.config import load_config

N_VIEWS = 75
H_NATIVE = 800
H_EVAL = 400
EVAL_DIR = "eval_holdout"


def dilate_mask(mask_bool: np.ndarray, dilation_px: int) -> np.ndarray:
    """The poisoning pipeline's dilation (scripts/build_poison_set.py): square
    (2d+1)x(2d+1) structuring element of ones, cv2.dilate on uint8, back to bool.
    That script inlines the operation, so this is a copy pinned to it by
    tests/test_metrics.py::test_dilation_matches_poisoning_expression."""
    if not dilation_px:
        return mask_bool.astype(bool).copy()
    kernel = np.ones((2 * dilation_px + 1,) * 2, np.uint8)
    return cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)


def downscale_mask(mask_bool: np.ndarray, size: int = H_EVAL) -> np.ndarray:
    """INTER_AREA on the float mask, then > 0.5."""
    if mask_bool.shape[0] == size:
        return mask_bool.astype(bool)
    small = cv2.resize(mask_bool.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return small > 0.5


def read_eval_mask(path: str) -> np.ndarray:
    """Raw eval mask, bool [800,800]. Asserts uint16 {0,65535} and shape."""
    m, dtype = read_mask(path)
    if dtype != np.uint16:
        raise ValueError("%s: eval masks must be uint16, got %s" % (path, dtype))
    if m.shape != (H_NATIVE, H_NATIVE):
        raise ValueError("%s: expected %dx%d, got %s" % (path, H_NATIVE, H_NATIVE, m.shape))
    return m


def composite_white(rgba: np.ndarray) -> np.ndarray:
    """evaluate.py's white-background composite on RGBA float32 [...,4] -> [...,3]."""
    return (rgba[..., :3] * rgba[..., -1:] + (1.0 - rgba[..., -1:])).astype(np.float32)


def view_inclusion(raw_mask_800: np.ndarray, min_fraction: float) -> (bool, float):
    """§2 minimum-visibility rule at the mask's NATIVE (800x800) resolution, on the
    RAW mask, as a fraction of frame pixels (configs/scenes/final_scene.yaml:
    v_target.min_visible_area_fraction = 0.005 = 3,200 px of 640,000)."""
    frac = float(raw_mask_800.sum()) / float(raw_mask_800.size)
    return frac >= min_fraction, frac


def load_eval_data(repo_root: str) -> Dict[str, object]:
    cfg = load_config(os.path.join(repo_root, "configs", "scenes", "final_scene.yaml"))
    dilation = int(cfg["render"]["background_plate"]["mask_dilation_px"])
    min_frac = float(cfg["v_target"]["min_visible_area_fraction"])
    if not cfg["dataset"].get("half_res", False):
        raise ValueError("final_scene.yaml dataset.half_res is not true; D-042 assumes 800->400")

    scene = os.path.join(repo_root, "data", "blender_scenes")
    # The training/eval loader itself. train/val/test all resolve under the clean scene
    # (clean originals; NOT a poisoned set). Only the test split is used.
    imgs, _, hwf, i_split = load_blender_data(scene, half_res=True, testskip=1)
    i_test = i_split[2]
    if len(i_test) != N_VIEWS:
        raise ValueError("expected %d test views, got %d" % (N_VIEWS, len(i_test)))
    gt = composite_white(imgs[i_test])
    del imgs
    if gt.shape != (N_VIEWS, H_EVAL, H_EVAL, 3):
        raise ValueError("GT shape %s" % (gt.shape,))

    plates = np.zeros_like(gt)
    m_dil = np.zeros((N_VIEWS, H_EVAL, H_EVAL), bool)
    m_raw = np.zeros((N_VIEWS, H_EVAL, H_EVAL), bool)
    included = np.zeros(N_VIEWS, bool)
    area_frac = np.zeros(N_VIEWS)
    for i in range(N_VIEWS):
        name = "r_%03d.png" % i
        p = iio.imread(os.path.join(repo_root, "data", "background_plates", EVAL_DIR, name))
        if p.dtype != np.uint8 or p.shape != (H_NATIVE, H_NATIVE, 4):
            raise ValueError("plate %s: dtype/shape %s %s" % (name, p.dtype, p.shape))
        p = (p / 255.0).astype(np.float32)
        p = cv2.resize(p, (H_EVAL, H_EVAL), interpolation=cv2.INTER_AREA)
        plates[i] = composite_white(p)

        raw = read_eval_mask(os.path.join(repo_root, "data", "masks", EVAL_DIR, name))
        included[i], area_frac[i] = view_inclusion(raw, min_frac)
        m_raw[i] = downscale_mask(raw)
        m_dil[i] = downscale_mask(dilate_mask(raw, dilation))

    return {"gt": gt, "plates": plates, "m_dil": m_dil, "m_raw": m_raw,
            "included": included, "area_fraction_800": area_frac,
            "dilation_px": dilation, "min_visible_area_fraction": min_frac}


def assert_partition(m_dil: np.ndarray) -> None:
    """Target (M_dil) and collateral (~M_dil) cover the frame exactly, per view."""
    tgt = m_dil.reshape(m_dil.shape[0], -1).sum(1)
    col = (~m_dil).reshape(m_dil.shape[0], -1).sum(1)
    total = m_dil.shape[1] * m_dil.shape[2]
    if not np.all(tgt + col == total):
        raise AssertionError("target + collateral != %d px for some view" % total)


def load_renders(npz_path: str) -> np.ndarray:
    with np.load(npz_path) as z:
        r = z["renders"]
        idx = z["view_index"]
    if r.dtype != np.float16 or r.shape != (N_VIEWS, H_EVAL, H_EVAL, 3):
        raise ValueError("%s: renders dtype/shape %s %s" % (npz_path, r.dtype, r.shape))
    if not np.array_equal(idx, np.arange(N_VIEWS)):
        raise ValueError("%s: view_index is not 0..%d in order" % (npz_path, N_VIEWS - 1))
    r = r.astype(np.float32)
    if not np.isfinite(r).all():
        raise ValueError("%s: non-finite values in renders" % npz_path)
    return r
