import copy
import json
import os

import numpy as np
import pytest

from metrics import aggregate as A


def _table(seed_offset=0.0):
    rng = np.random.default_rng(0)
    rows = []
    for ci, cond in enumerate(A.CONDITIONS):
        for s in (1, 2, 3):
            r = {"run_id": "%s_seed%d" % (cond, s), "condition": cond, "seed": s}
            for col in A.METRIC_COLS:
                r[col] = float(ci * 10 + s + rng.random() + seed_offset)
            rows.append(r)
    return rows


def test_mean_and_sample_std_match_numpy():
    rows = _table()
    summ = {(r["condition"], r["metric"]): r for r in A.condition_summary(rows)}
    for cond in A.CONDITIONS:
        v = np.array([r["tgt_psnr"] for r in rows if r["condition"] == cond])
        s = summ[(cond, "tgt_psnr")]
        assert s["n"] == 3
        assert s["mean"] == pytest.approx(v.mean())
        assert s["std_ddof1"] == pytest.approx(np.std(v, ddof=1))
        assert s["std_ddof1"] != pytest.approx(np.std(v, ddof=0))
        assert s["min"] == v.min() and s["max"] == v.max()


def test_pairs_are_exactly_rule_b():
    assert A.PAIRS == {"hard_vs_soft": ("C3", "C6"), "random_vs_strategic": ("C3", "C7")}
    assert A.BUDGET_CONDITIONS == ["control", "C1", "C2", "C3", "C4", "C5"]
    assert [A.BUDGET_PERCENT[c] for c in A.BUDGET_CONDITIONS] == [0, 5, 10, 20, 30, 50]
    out = A.pair_comparisons(_table())
    assert {(r["cond_a"], r["cond_b"]) for r in out} == {("C3", "C6"), ("C3", "C7")}
    assert not any(r["cond_b"] == "C4" or r["cond_a"] == "C4" for r in out)


def test_range_overlap_flag():
    assert A.ranges_overlap([1, 2, 3], [3, 4, 5])           # touching counts as overlap
    assert A.ranges_overlap([1, 5], [2, 3])                 # containment
    assert not A.ranges_overlap([1, 2, 3], [3.0001, 4, 5])
    rows = _table()
    for r in rows:
        if r["condition"] == "C7":
            r["tgt_psnr"] = 1000.0 + r["seed"]
    row = [x for x in A.pair_comparisons(rows) if x["cond_b"] == "C7" and x["metric"] == "tgt_psnr"][0]
    assert row["ranges_overlap"] is False


def test_aggregation_does_not_mutate_inputs():
    rows = _table()
    before = copy.deepcopy(rows)
    A.condition_summary(rows)
    A.pair_comparisons(rows)
    assert rows == before


def test_spread_kind_labels():
    assert "initialisation-only" in A.spread_kind("control")
    assert "initialisation-only" in A.spread_kind("C7")
    assert "view-selection" in A.spread_kind("C3")


def test_angular_geometry_synthetic():
    p = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0], [1, 0, 1]])
    az, el = A.azimuth_elevation_deg(p)
    assert az[0] == pytest.approx(0) and az[1] == pytest.approx(90)
    assert el[2] == pytest.approx(90) and el[3] == pytest.approx(45)
    d = A.angular_distance_deg(p[:3], p[:3])
    assert np.allclose(np.diag(d), 0, atol=1e-6)
    assert d[0, 1] == pytest.approx(90) and d[0, 2] == pytest.approx(90)
    # distance is scale-invariant (angle seen from the origin)
    assert A.angular_distance_deg(p[:1], 5 * p[3:4])[0, 0] == pytest.approx(45)
    near = A.nearest_angular_distance_deg(p[:2], p[2:4])
    assert near[0] == pytest.approx(45) and near[1] == pytest.approx(90)
    with pytest.raises(ValueError):
        A.nearest_angular_distance_deg(p, p[:0])


def test_positions_from_transforms():
    m = np.eye(4)
    m[:3, 3] = [1, 2, 3]
    assert np.array_equal(A.positions_from_transforms([{"transform_matrix": m.tolist()}]), [[1, 2, 3]])


def test_distance_bands_partition_collateral_and_follow_distance():
    m = np.zeros((300, 300), bool)
    m[140:160, 140:160] = True
    bands = A.band_masks(m)
    assert A.band_labels() == ["0-10", "10-30", "30-100", ">100"]
    # disjoint, within the collateral region, covering all of it
    stack = np.stack(bands)
    assert stack.sum(0).max() == 1
    assert not (stack.any(0) & m).any()
    assert np.array_equal(stack.any(0), ~m)
    # a pixel 5 px right of the mask is in band 0; 20 px -> band 1; 60 px -> band 2; far corner -> last band
    assert bands[0][150, 159 + 5] and bands[1][150, 159 + 20] and bands[2][150, 159 + 60]
    far = np.zeros((600, 600), bool)
    far[0:5, 0:5] = True
    assert A.band_masks(far)[3][599, 599]
    err = np.zeros(m.shape)
    err[bands[1]] = 2.0
    got = A.band_mse(err, bands)
    assert got[1] == pytest.approx(2.0) and got[0] == 0.0
    assert np.isnan(A.band_mse(err, [np.zeros_like(m)])[0])


def test_distribution_stats():
    s = A.distribution_stats(np.arange(1, 101))
    assert s["n"] == 100 and s["median"] == pytest.approx(50.5) and s["q1"] < s["median"] < s["q3"]


def test_figure_smoke_and_sidecar(tmp_path):
    from metrics import figures as F
    rows = _table()
    ylims = {c: (0, 100) for c in ("tgt_psnr", "col_psnr", "plate_psnr")}
    figs = [F.budget_figure(rows, "tgt", "t", raw_prefix="rawtgt"), F.plate_figure(rows, 12.0, "t"),
            F.pair_figure(rows, "C3", "C6", list(ylims), ylims, "t")]
    names = ["a", "b", "c"]
    for f, n in zip(figs, names):
        files = F.save_fig(f, n, str(tmp_path), {"per_run_sha256": "x", "run_ids": ["r"], "columns": ["c"]})
        assert all(os.path.getsize(p) > 0 for p in files)
    side = json.load(open(tmp_path / "a.json"))
    assert side["figure"] == "a" and side["per_run_sha256"] == "x"
    assert (tmp_path / "a.png").exists() and (tmp_path / "a.pdf").exists()
