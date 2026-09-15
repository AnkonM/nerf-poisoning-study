"""Exact pixel-math tests for the METHODOLOGY.md §3 attack formulas.

Synthetic fixtures only — small, fast, deterministic, and independent of the
frozen scene, so these keep working if the data is ever unavailable.
"""

import numpy as np
import pytest

from poisoning.compositor import composite, hard_erasure, soft_suppression

ALPHA = 0.4          # the study's locked soft-suppression value (D-029)


def _fixture():
    """2x2 RGB: left column inside the mask, right column outside."""
    original = np.array([[[200, 100, 50], [200, 100, 50]],
                         [[200, 100, 50], [200, 100, 50]]], dtype=np.uint8)
    plate = np.array([[[0, 50, 150], [0, 50, 150]],
                      [[0, 50, 150], [0, 50, 150]]], dtype=np.uint8)
    mask = np.array([[1.0, 0.0], [1.0, 0.0]])
    return original, mask, plate


def test_hard_erasure_exact_pixels():
    original, mask, plate = _fixture()
    out = hard_erasure(original, mask, plate)
    # inside mask -> plate exactly; outside -> original exactly
    assert out[0, 0].tolist() == [0, 50, 150]
    assert out[0, 1].tolist() == [200, 100, 50]
    assert out.dtype == np.uint8


def test_soft_suppression_exact_pixels_at_alpha_0_4():
    original, mask, plate = _fixture()
    out = soft_suppression(original, mask, plate, alpha=ALPHA)
    # 0.4*original + 0.6*plate, computed by hand:
    #   R: .4*200 + .6*0   =  80.0 -> 80
    #   G: .4*100 + .6*50  =  70.0 -> 70
    #   B: .4*50  + .6*150 = 110.0 -> 110
    assert out[0, 0].tolist() == [80, 70, 110]
    # outside the mask nothing moves, whatever alpha is
    assert out[0, 1].tolist() == [200, 100, 50]


def test_alpha_zero_collapses_to_hard_erasure_exactly():
    rng = np.random.default_rng(0)
    for _ in range(5):
        original = rng.integers(0, 256, (9, 7, 3), dtype=np.uint8)
        plate = rng.integers(0, 256, (9, 7, 3), dtype=np.uint8)
        mask = (rng.random((9, 7)) > 0.5).astype(np.float64)
        assert np.array_equal(composite(original, mask, plate, alpha=0.0),
                              hard_erasure(original, mask, plate))


def test_outside_mask_is_untouched_for_every_alpha():
    """The property Step 7's strict gate check relies on, stated as a unit test."""
    rng = np.random.default_rng(1)
    original = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
    plate = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
    mask = (rng.random((12, 12)) > 0.5).astype(np.float64)
    outside = mask == 0
    for a in (0.0, 0.25, ALPHA, 0.5, 1.0):
        out = composite(original, mask, plate, alpha=a)
        assert np.array_equal(out[outside], original[outside])


def test_image_alpha_channel_passes_through_unchanged():
    rng = np.random.default_rng(2)
    original = rng.integers(0, 256, (6, 6, 4), dtype=np.uint8)
    plate = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    mask = np.ones((6, 6))
    out = composite(original, mask, plate, alpha=ALPHA)
    assert out.shape[-1] == 4
    assert np.array_equal(out[..., 3], original[..., 3])


def test_soft_suppression_rejects_alpha_zero():
    """alpha=0 would make C6 identical to C3 — must be refused, not silently run."""
    original, mask, plate = _fixture()
    with pytest.raises(ValueError, match="hard erasure"):
        soft_suppression(original, mask, plate, alpha=0.0)


def test_composite_rejects_alpha_out_of_range():
    original, mask, plate = _fixture()
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            composite(original, mask, plate, alpha=bad)
