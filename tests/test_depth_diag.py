import numpy as np
import pytest

from metrics import depth_diag as DD


def test_region_mean_and_empty_mask_raises():
    arr = np.zeros((2, 4, 4)); arr[0, :2, :2] = 3.0; arr[1] = 5.0
    m = np.zeros((2, 4, 4), bool); m[0, :2, :2] = True; m[1, 0, 0] = True
    assert DD.region_mean(arr, m).tolist() == [3.0, 5.0]
    m[1] = False
    with pytest.raises(ValueError):
        DD.region_mean(arr, m)
    with pytest.raises(ValueError):
        DD.region_mean(arr, np.ones((2, 3, 3), bool))


def test_mean_abs_diff_in_region_only_counts_masked_pixels():
    a = np.zeros((1, 4, 4)); b = np.zeros((1, 4, 4))
    b[0, 0, 0] = -2.0; b[0, 3, 3] = 100.0          # second pixel is outside the mask
    m = np.zeros((1, 4, 4), bool); m[0, 0:2, 0:2] = True
    assert DD.region_mean_abs_diff(a, b, m)[0] == pytest.approx(2.0 / 4)
    assert DD.region_mean_abs_diff(a, a, m)[0] == 0.0


def test_depth_from_disp_is_reciprocal():
    assert DD.depth_from_disp(np.array([0.5, 0.25])).tolist() == [2.0, 4.0]


def test_spearman_known_values():
    x = np.arange(10.0)
    assert DD.spearman_rho(x, x ** 3) == pytest.approx(1.0)
    assert DD.spearman_rho(x, -x) == pytest.approx(-1.0)
    assert np.isnan(DD.spearman_rho(x, np.ones(10)))
    assert DD.spearman_rho([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)


def test_bin_means_edges_and_empty_bin():
    out = DD.bin_means([1, 5, 12, 15, 40], [1, 3, 10, 20, 100], [10, 20])
    assert [o["bin_deg"] for o in out] == ["<10", "10-20", ">=20"]
    assert [o["n"] for o in out] == [2, 2, 1]
    assert out[0]["mean_y"] == 2.0 and out[1]["mean_y"] == 15.0 and out[2]["mean_y"] == 100.0
    assert np.isnan(DD.bin_means([1, 2], [1, 2], [10, 20])[2]["mean_y"])


def test_summarise():
    s = DD.summarise(np.arange(1, 101))
    assert s["n"] == 100 and s["median"] == pytest.approx(50.5) and s["min"] == 1 and s["max"] == 100


def test_render_depth_acc_refuses_cpu():
    with pytest.raises(RuntimeError, match="CUDA only"):
        DD.render_depth_acc(lambda c2w, chunk: None, np.zeros((1, 4, 4)), "cpu", 2048, 256)
