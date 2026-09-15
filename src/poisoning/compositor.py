"""Poisoning compositors implementing the fixed formulas from
docs/METHODOLOGY.md §3.

Per CLAUDE.md's non-negotiable rules, this module is the *only* code path
allowed to produce a poisoned image — no hand-editing, no per-image
judgment calls. Callers (scripts/build_poison_set.py) supply the mask and
background_plate; this module does not decide what those are.

METHODOLOGY.md §3 specifies two attacks, but they are one formula. Soft
suppression is the general case:

    poisoned = mask*(alpha*original + (1-alpha)*plate) + (1-mask)*original

and hard erasure is exactly its `alpha = 0` case:

    poisoned = mask*plate + (1-mask)*original

They are therefore implemented as ONE function (`composite`) rather than two
parallel implementations, so the two conditions provably cannot drift apart in
rounding, dtype handling or alpha-channel treatment. `hard_erasure` is kept as a
thin alpha=0 wrapper for existing Phase 3 callers.

Note what the formula does NOT do: it reads `background_plate` only where
`mask == 1`. Everything outside the mask — including the target's cast shadow,
contact AO and colour bleed — is the original, untouched. That is an intended
property of a mask-limited edit, not a defect; see METHODOLOGY.md §3's "Scope of
the edit" paragraph and DECISION_LOG.md D-022 Finding 4 / D-024.
"""

import numpy as np


def alpha_to_mask(alpha: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Binarize an 8-bit alpha channel into a {0, 1} float mask.

    `mask == 1` marks the target/foreground (alpha > threshold). Only used
    for the Phase 3 Lego PoC, where the alpha channel doubles as the
    object-vs-background segmentation mask — see docs/DECISION_LOG.md
    D-017. The final-scene protocol (Phase 4+) uses a real rendered
    object-ID mask instead and does not go through this function.

    WARNING (D-024/D-026): the final scene's masks are 16-bit; this uint8
    threshold would mark every pixel of one as foreground. Use
    src/data_pipeline/scene_validation.py::read_mask for those.
    """
    return (alpha.astype(np.float64) > threshold).astype(np.float64)


def composite(original: np.ndarray, mask: np.ndarray, background_plate,
              alpha: float = 0.0) -> np.ndarray:
    """The METHODOLOGY.md §3 attack formula, parameterized by `alpha`.

        poisoned = mask*(alpha*original + (1-alpha)*plate) + (1-mask)*original

    Applied to the RGB channels only. Any alpha channel present in
    `original` is passed through unchanged (note: that image alpha channel
    is unrelated to this function's `alpha` blend parameter).

    Parameters
    ----------
    original : (H, W, 3) or (H, W, 4) array, uint8 or float.
    mask : (H, W) or (H, W, 1) array in [0, 1]. For this study it is binary
        {0, 1}; the formula is defined for intermediate values too.
    background_plate : (H, W, 3) array (a real rendered plate) or a
        length-3 sequence / scalar (a flat proxy colour), broadcastable
        against `original`'s RGB channels.
    alpha : float in [0, 1]. Weight retained by the ORIGINAL inside the mask.
        `alpha=0` is hard erasure (target fully replaced by the plate);
        larger values leave the target more visible. Fixed once per the
        study and never varied per image or per budget level (§3).

    Returns
    -------
    Array with the same shape and dtype as `original`.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1], got %r" % (alpha,))

    original = np.asarray(original)
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 2:
        mask = mask[..., None]

    rgb = original[..., :3].astype(np.float64)
    bg = np.broadcast_to(np.asarray(background_plate, dtype=np.float64), rgb.shape)

    # Inside the mask, blend original toward the plate; outside, keep original.
    inside = float(alpha) * rgb + (1.0 - float(alpha)) * bg
    poisoned_rgb = mask * inside + (1.0 - mask) * rgb

    if np.issubdtype(original.dtype, np.integer):
        poisoned_rgb = np.clip(np.round(poisoned_rgb), 0, 255)
    poisoned_rgb = poisoned_rgb.astype(original.dtype)

    if original.shape[-1] == 4:
        return np.concatenate([poisoned_rgb, original[..., 3:4]], axis=-1)
    return poisoned_rgb


def hard_erasure(original: np.ndarray, mask: np.ndarray, background_plate) -> np.ndarray:
    """Hard erasure — `composite(..., alpha=0.0)`, per METHODOLOGY.md §3:

        poisoned_pixel = mask * background_plate + (1 - mask) * original

    Retained as a named wrapper so Phase 3's callers and the conditions matrix's
    vocabulary (§5) keep working. Verified bit-for-bit identical to the original
    Phase 3 implementation this replaced — see DECISION_LOG.md D-029.
    """
    return composite(original, mask, background_plate, alpha=0.0)


def soft_suppression(original: np.ndarray, mask: np.ndarray, background_plate,
                     alpha: float) -> np.ndarray:
    """Soft suppression (condition C6) — `composite` with a nonzero `alpha`.

    `alpha` is required, not defaulted: §3 fixes it once for the whole study and
    a silent default here would be exactly the kind of un-recorded per-image
    parameter the protocol forbids. The study's value lives in
    configs/poisoning/soft_suppression_20.yaml.
    """
    if alpha <= 0.0:
        raise ValueError(
            "soft suppression requires alpha > 0; alpha=0 is hard erasure and "
            "would make C6 identical to C3 (got %r)" % (alpha,))
    return composite(original, mask, background_plate, alpha=alpha)
