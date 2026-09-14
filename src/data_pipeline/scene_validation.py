"""Phase 4 Step 5: compute |V_target| and validate the rendered dataset.

Runs in the project's uv venv (not Blender). Two jobs:

1. **Compute `|V_target|`** — the count of TRAINING views whose target mask
   covers at least `v_target.min_visible_area_fraction` of the frame
   (METHODOLOGY.md §2). The threshold was locked in D-022 *before* this number
   was ever computed, and is read from the config, never inferred from the data.

2. **Validate** the originals / masks / background_plates on disk.

On the outside-mask check: Phase 3's Lego PoC asserted strict pixel identity
outside the mask (D-017). That assertion is NOT valid here and asserting it
would be wrong. With `hide_render` plates (D-022) the target's cast shadow and
its diffuse colour bleed legitimately disappear from the plate, and both lie
outside the mask — D-022 Finding 4 and D-024 established this is expected
behaviour of mask-limited erasure, not misregistration. What IS checked is the
bound that still must hold: beyond the object's shadow/bleed neighbourhood, the
two renders must agree to within denoiser noise, which is what proves the fixed
`cycles.seed` is actually holding and that nothing has shifted between the two
renders of a view.
"""

import json
import os

import cv2
import imageio.v2 as iio
import numpy as np

# D-024 measured max far-field |original - plate| at 12/255 over 48 poses.
# A generous bound: anything above this means the seed is not holding or the
# two renders of a view are misregistered.
FAR_FIELD_MAX_DIFF = 20
FAR_FIELD_DISTANCE_PX = 150


def read_mask(path):
    """Read a mask and binarise it, asserting the bit depth.

    D-022/D-024: masks are 16-bit. D-017's Lego `alpha > 127` assumes uint8 and
    would mark EVERY pixel of a 16-bit mask as foreground — a failure that
    produces a plausible full-frame mask rather than an error, so the dtype is
    asserted rather than assumed.
    """
    m = iio.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype not in (np.uint8, np.uint16):
        raise ValueError("%s: unexpected mask dtype %s" % (path, m.dtype))
    full = np.iinfo(m.dtype).max
    uniq = np.unique(m)
    if not set(uniq.tolist()) <= {0, full}:
        raise ValueError("%s: mask is not binary, found %d unique values"
                         % (path, len(uniq)))
    return m > (full // 2), m.dtype


def validate(data_root, cameras, cfg):
    """Returns (results dict, list of error strings)."""
    errors = []
    thresh = cfg["v_target"]["min_visible_area_fraction"]
    dilation = cfg["render"]["background_plate"]["mask_dilation_px"]

    aux_splits = {"train", "eval_holdout"}
    areas = {}
    dtypes = set()
    far_field_worst = 0.0
    far_field_worst_view = None
    holes_open = holes_closed = 0

    by_split = {}
    for p in cameras["poses"]:
        by_split.setdefault(p["split"], []).append(p)

    for split, poses in sorted(by_split.items()):
        for p in poses:
            name = os.path.basename(p["file_path"])
            img_p = os.path.join(data_root, "blender_scenes", split, name + ".png")
            if not os.path.exists(img_p):
                errors.append("missing original: %s" % img_p)
                continue
            img = iio.imread(img_p)
            if img.shape[2] != 4:
                errors.append("%s: expected RGBA, got %d channels"
                              % (img_p, img.shape[2]))

            if split not in aux_splits:
                continue

            mask_p = os.path.join(data_root, "masks", split, name + ".png")
            plate_p = os.path.join(data_root, "background_plates", split, name + ".png")
            for q in (mask_p, plate_p):
                if not os.path.exists(q):
                    errors.append("missing: %s" % q)
            if not (os.path.exists(mask_p) and os.path.exists(plate_p)):
                continue

            try:
                mask, dt = read_mask(mask_p)
            except ValueError as e:
                errors.append(str(e))
                continue
            dtypes.add(str(dt))

            frac = float(mask.mean())
            areas.setdefault(split, {})[name] = frac
            if frac <= 0.0:
                errors.append("%s: mask is empty" % mask_p)
            if frac >= 1.0:
                errors.append("%s: mask covers the whole frame" % mask_p)

            plate = iio.imread(plate_p)
            if plate.shape != img.shape:
                errors.append("%s: plate/original shape mismatch" % name)
                continue

            # far-field agreement (see module docstring)
            dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
            far = dist > FAR_FIELD_DISTANCE_PX
            if far.any():
                d = np.abs(img[..., :3].astype(np.int16)
                           - plate[..., :3].astype(np.int16)).max(axis=2)
                worst = float(d[far].max())
                if worst > far_field_worst:
                    far_field_worst, far_field_worst_view = worst, "%s/%s" % (split, name)
                if worst > FAR_FIELD_MAX_DIFF:
                    errors.append("%s/%s: far-field |original-plate| = %d exceeds %d "
                                  "(seed not holding, or misregistration)"
                                  % (split, name, worst, FAR_FIELD_MAX_DIFF))

            # handle-hole survival under the locked dilation (D-024)
            mu = mask.astype(np.uint8)
            dil = cv2.dilate(mu, np.ones((2 * dilation + 1,) * 2, np.uint8))
            if _enclosed_hole_px(mu) > 0:
                if _enclosed_hole_px(dil) > 0:
                    holes_open += 1
                else:
                    holes_closed += 1

    # ---- |V_target| : TRAINING views only (METHODOLOGY.md §2) --------------
    train_areas = areas.get("train", {})
    v_target = sorted(n for n, f in train_areas.items() if f >= thresh)
    results = {
        "threshold": thresh,
        "n_train": len(train_areas),
        "v_target_count": len(v_target),
        "v_target_views": v_target,
        "train_area_min": min(train_areas.values()) if train_areas else None,
        "train_area_max": max(train_areas.values()) if train_areas else None,
        "train_area_median": float(np.median(list(train_areas.values()))) if train_areas else None,
        "margin_min_x": (min(train_areas.values()) / thresh) if train_areas else None,
        "areas": areas,
        "mask_dtypes": sorted(dtypes),
        "far_field_worst": far_field_worst,
        "far_field_worst_view": far_field_worst_view,
        "handle_holes_open_after_dilation": holes_open,
        "handle_holes_closed_by_dilation": holes_closed,
    }
    return results, errors


def _enclosed_hole_px(mask_u8):
    """Background pixels fully enclosed by the mask (i.e. the handle hole)."""
    n, lab = cv2.connectedComponents((1 - mask_u8).astype(np.uint8))
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return sum(int((lab == i).sum()) for i in range(1, n) if i not in border)


def write_v_target(cameras_path, results):
    """Record |V_target| and the per-view areas into cameras.json.

    METHODOLOGY.md §2 requires these live in cameras.json metadata and never be
    recomputed mid-study.
    """
    cams = json.load(open(cameras_path))
    cams["v_target"]["count"] = results["v_target_count"]
    cams["v_target"]["n_train_views"] = results["n_train"]
    cams["v_target"]["views"] = results["v_target_views"]
    cams["v_target"]["per_view_area_fraction"] = results["areas"]
    with open(cameras_path, "w") as f:
        json.dump(cams, f, indent=1)
