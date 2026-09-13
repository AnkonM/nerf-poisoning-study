"""Poisoning compositors implementing the fixed formulas from
docs/METHODOLOGY.md §3.

Per CLAUDE.md's non-negotiable rules, this module is the *only* code path
allowed to produce a poisoned image — no hand-editing, no per-image
judgment calls. Callers (scripts/build_poison_set.py) supply the mask and
background_plate; this module does not decide what those are.
"""

import numpy as np


def alpha_to_mask(alpha: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Binarize an 8-bit alpha channel into a {0, 1} float mask.

    `mask == 1` marks the target/foreground (alpha > threshold). Only used
    for the Phase 3 Lego PoC, where the alpha channel doubles as the
    object-vs-background segmentation mask — see docs/DECISION_LOG.md
    D-017. The final-scene protocol (Phase 4+) uses a real rendered
    object-ID mask instead and does not go through this function.
    """
    return (alpha.astype(np.float64) > threshold).astype(np.float64)


def hard_erasure(original: np.ndarray, mask: np.ndarray, background_plate) -> np.ndarray:
    """Hard erasure, per docs/METHODOLOGY.md §3:

        poisoned_pixel = mask * background_plate + (1 - mask) * original

    Applied to the RGB channels only. Any alpha channel present in
    `original` is passed through unchanged — for the Phase 3 Lego PoC the
    mask is itself derived from that alpha channel (D-017), so the
    original transparency/segmentation information is preserved exactly;
    only the visible object color is replaced.

    Parameters
    ----------
    original : (H, W, 3) or (H, W, 4) array, uint8 or float.
    mask : (H, W) or (H, W, 1) array in [0, 1] (typically {0, 1} for hard
        erasure — see `alpha_to_mask`).
    background_plate : (H, W, 3) array (a real rendered plate) or a
        length-3 sequence / scalar (a flat proxy color), broadcastable
        against `original`'s RGB channels.

    Returns
    -------
    Array with the same shape and dtype as `original`.
    """
    original = np.asarray(original)
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 2:
        mask = mask[..., None]

    rgb = original[..., :3].astype(np.float64)
    bg = np.broadcast_to(np.asarray(background_plate, dtype=np.float64), rgb.shape)

    poisoned_rgb = mask * bg + (1.0 - mask) * rgb

    if np.issubdtype(original.dtype, np.integer):
        poisoned_rgb = np.clip(np.round(poisoned_rgb), 0, 255)
    poisoned_rgb = poisoned_rgb.astype(original.dtype)

    if original.shape[-1] == 4:
        return np.concatenate([poisoned_rgb, original[..., 3:4]], axis=-1)
    return poisoned_rgb


# Soft suppression (docs/METHODOLOGY.md §3, condition C6) is intentionally
# not implemented here yet: docs/ROADMAP.md's Phase 3 task list and this
# step's instructions scope this PoC to hard erasure only. Add it
# alongside hard_erasure (same module, per PROJECT_STRUCTURE.md) when C6
# is actually built, rather than speculatively now.
