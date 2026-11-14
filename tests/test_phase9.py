import numpy as np
import pytest

from metrics import phase9_evidence as P
from poisoning.view_selection import sample_farthest_point


def _sphere(n=100, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return 4.0 * v / np.linalg.norm(v, axis=1, keepdims=True)


def _ang(pos):
    d = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    return np.arccos(np.clip(d @ d.T, -1, 1))


def test_fps_exactly_20_distinct_and_deterministic():
    pos = _sphere()
    a, n = sample_farthest_point(pos, range(100), 20, seed=1)
    b, _ = sample_farthest_point(pos, range(100), 20, seed=1)
    assert n == 20 and a == b == sorted(set(a)) and len(set(a)) == 20


def test_fps_seed_changes_only_the_first_draw():
    pos = _sphere()
    firsts = {s: int(np.random.default_rng(s).integers(100)) for s in (1, 2, 3)}
    for s in (1, 2, 3):
        sel, _ = sample_farthest_point(pos, range(100), 20, seed=s)
        assert firsts[s] in sel
    assert len({tuple(sample_farthest_point(pos, range(100), 20, seed=s)[0]) for s in (1, 2, 3)}) > 1


def test_fps_max_min_property():
    pos = _sphere(60)
    cand = list(range(60))
    sel, _ = sample_farthest_point(pos, cand, 20, seed=3)
    A = _ang(pos)
    # replay: each pick (in greedy order) has the largest min-distance to the previously chosen ones
    first = int(np.random.default_rng(3).integers(60))
    chosen = [first]
    rest = set(sel) - {first}
    while rest:
        mind = {v: min(A[v, c] for c in chosen) for v in range(60) if v not in chosen}
        best = max(mind.values())
        nxt = min(v for v, m in mind.items() if m >= best - 1e-12)
        assert nxt in rest
        chosen.append(nxt)
        rest.remove(nxt)


def test_fps_tie_breaks_to_lowest_index():
    # 4 symmetric directions on the equator: after the first pick, 2 views at 90 deg tie -> lowest index
    pos = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
    # first draw for seed s is an index; pick seed whose first draw is 0
    s = next(s for s in range(50) if int(np.random.default_rng(s).integers(4)) == 0)
    sel, _ = sample_farthest_point(pos, range(4), 50, seed=s)
    assert sel == [0, 2]            # the opposite view first (180 deg), no tie there
    sel3, _ = sample_farthest_point(pos, range(4), 75, seed=s)
    assert sel3 == [0, 1, 2]        # views 1 and 3 tie at 90 deg: the lowest index wins


def test_fps_respects_candidate_subset_and_zero_budget():
    pos = _sphere()
    sel, n = sample_farthest_point(pos, [3, 5, 8, 13, 21], 40, seed=1)
    assert n == 2 and set(sel) <= {3, 5, 8, 13, 21}
    assert sample_farthest_point(pos, range(100), 0, seed=1) == ([], 0)


def test_average_ranks_and_partial_spearman():
    assert list(P.average_ranks([3, 1, 1, 2])) == [4.0, 1.5, 1.5, 3.0]
    rng = np.random.default_rng(0)
    z = rng.normal(size=60)
    x = z + 0.3 * rng.normal(size=60)
    y = z + 0.3 * rng.normal(size=60)
    assert P.partial_spearman(x, y, z) < 0.5 < np.corrcoef(P.average_ranks(x), P.average_ranks(y))[0, 1]
    # y = monotone function of x, uncorrelated with z  -> partial correlation 1
    x2 = rng.normal(size=40)
    assert P.partial_spearman(x2, np.exp(x2), rng.normal(size=40)) > 0.9


def test_partial_spearman_is_nan_when_fully_explained_by_control():
    z = np.arange(30.0)
    assert np.isnan(P.partial_spearman(z, -z, z))      # both rank vectors are exact functions of the control


def test_resolvability_and_step_ratios_use_ddof1():
    c3 = [10.0, 12.0, 14.0]                 # sd (ddof=1) = 2
    assert P.resolvability_ratio([17.0, 17.0, 17.0], c3) == pytest.approx(5.0 / 2.0)
    hi, lo = [10.0, 12.0], [4.0, 8.0]       # sds sqrt(2), 2*sqrt(2); means 11, 6
    assert P.step_ratio(hi, lo) == pytest.approx(5.0 / (0.5 * (2 ** 0.5 + 2 * 2 ** 0.5)))


def test_e1_stats_and_hours():
    s = P.e1_stats([1.0, 2.0, 9.0])
    assert s == {"mean_deg": 4.0, "median_deg": 2.0, "max_deg": 9.0}
    assert P.hours_per_run(36000, 10.0, 2, 60.0, 120.0) == pytest.approx(1.0 + 240.0 / 3600.0)


# ---- poisoning/verify.py (Phase 5 gate checks) -------------------------------------------------
import json as _json
import os as _os

import imageio.v2 as _iio

from poisoning import verify as V


def test_check_image_pair_outside_mask_identity():
    orig = np.full((8, 8, 3), 100, np.uint8)
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    ok = orig.copy()
    ok[3, 3] = 7                                        # change inside the mask only
    assert V.check_image_pair(orig, ok, mask, True) == []
    bad = ok.copy()
    bad[7, 7] = 1                                       # change outside the mask
    assert any("OUTSIDE" in p for p in V.check_image_pair(orig, bad, mask, True))
    assert any("identical" in p for p in V.check_image_pair(orig, orig.copy(), mask, True))
    assert any("not flagged" in p for p in V.check_image_pair(orig, ok, mask, False))


def test_dilate_grows_by_the_radius():
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    assert V.dilate(m, 0).sum() == 1 and V.dilate(m, 2).sum() == 25


def _synthetic_dataset(tmp_path, n=4, poisoned=(1, 3), break_outside=False, break_copy=False):
    scene, ds, masks = tmp_path / "scene", tmp_path / "ds", tmp_path / "masks"
    for d in (scene / "train", ds / "train", masks):
        d.mkdir(parents=True)
    frames = [{"file_path": "./train/r_%03d" % i, "transform_matrix": np.eye(4).tolist()} for i in range(n)]
    tj = {"camera_angle_x": 0.7, "frames": frames}
    for d in (scene, ds):
        _json.dump(tj, open(d / "transforms_train.json", "w"))
    rows = []
    for i in range(n):
        img = np.full((10, 10, 3), 50 + i, np.uint8)
        _iio.imwrite(scene / "train" / ("r_%03d.png" % i), img)
        m = np.zeros((10, 10), np.uint16)
        m[4:6, 4:6] = 65535
        _iio.imwrite(masks / ("r_%03d.png" % i), m)
        out = img.copy()
        if i in poisoned:
            out[4:6, 4:6] = 0
            if break_outside and i == poisoned[0]:
                out[0, 0] = 255
        if break_copy and i == 0:
            out = img.copy()
            out[9, 9] = 1
        _iio.imwrite(ds / "train" / ("r_%03d.png" % i), out)
        rows.append({"view_index": str(i), "poisoned": str(i in poisoned)})
    return scene, ds, masks, rows


def _read_mask(p):
    m = _iio.imread(p)
    return m > 0, None


def test_verify_dataset_pass_and_each_failure(tmp_path):
    scene, ds, masks, rows = _synthetic_dataset(tmp_path)
    r = V.verify_dataset(str(ds), str(scene), rows, 2, str(masks), 0, _read_mask, expect_views=4)
    assert r["problems"] == [] and r["n_poisoned"] == 2
    assert any("expected exactly 3" in p for p in V.verify_dataset(str(ds), str(scene), rows, 3, str(masks), 0,
                                                                   _read_mask, expect_views=4)["problems"])
    s2, d2, m2, r2 = _synthetic_dataset(tmp_path / "b", break_outside=True)
    assert any("OUTSIDE" in p for p in V.verify_dataset(str(d2), str(s2), r2, 2, str(m2), 0, _read_mask, expect_views=4)["problems"])
    s3, d3, m3, r3 = _synthetic_dataset(tmp_path / "c", break_copy=True)
    assert any("byte-identical" in p for p in V.verify_dataset(str(d3), str(s3), r3, 2, str(m3), 0, _read_mask, expect_views=4)["problems"])
