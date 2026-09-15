"""Mask binarisation must assert bit depth rather than assume uint8 (D-024)."""

import imageio.v2 as iio
import numpy as np
import pytest

from data_pipeline.scene_validation import read_mask


def _write(tmp_path, arr):
    p = tmp_path / "m.png"
    iio.imwrite(p, arr)
    return str(p)


def test_reads_16bit_binary_mask(tmp_path):
    arr = np.zeros((8, 8), dtype=np.uint16)
    arr[2:5, 2:5] = 65535
    mask, dtype = read_mask(_write(tmp_path, arr))
    assert dtype == np.uint16
    assert mask.sum() == 9 and mask.dtype == bool


def test_uint8_mask_thresholds_at_its_own_max_not_127(tmp_path):
    """The D-017 bug this guards: `alpha > 127` on a 16-bit mask marks everything
    foreground. Threshold must be half of the dtype max, not a hardcoded 127."""
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[0:4, :] = 255
    mask, dtype = read_mask(_write(tmp_path, arr))
    assert dtype == np.uint8
    assert mask.sum() == 32

def test_rejects_non_binary_mask(tmp_path):
    arr = np.zeros((8, 8), dtype=np.uint16)
    arr[0, 0] = 30000            # an intermediate value: not a clean ID mask
    with pytest.raises(ValueError, match="not binary"):
        read_mask(_write(tmp_path, arr))
