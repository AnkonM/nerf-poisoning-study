import numpy as np
import pytest

from metrics import aggregate as A
from metrics import followup as FU


def _rows(offset_c6=0.5):
    rows = []
    for c in A.CONDITIONS:
        for s in (1, 2, 3):
            r = {"run_id": "%s_seed%d" % (c, s), "condition": c, "seed": s}
            for i, col in enumerate(A.METRIC_COLS):
                r[col] = float(10 * A.CONDITIONS.index(c) + s + 0.1 * i)
            if c == "C6":
                r["tgt_psnr"] += offset_c6 * (1 if s != 2 else -1)
            rows.append(r)
    return rows


def test_paired_difference_is_b_minus_a_per_seed():
    rows = _rows()
    d = {x["metric"]: x for x in FU.paired_differences(rows, "C3", "C6", ["tgt_psnr", "col_psnr"], [1, 2, 3])}
    t = d["tgt_psnr"]
    c3 = {r["seed"]: r["tgt_psnr"] for r in rows if r["condition"] == "C3"}
    c6 = {r["seed"]: r["tgt_psnr"] for r in rows if r["condition"] == "C6"}
    for s in (1, 2, 3):
        assert t["diff_seed%d" % s] == pytest.approx(c6[s] - c3[s])
    v = np.array([c6[s] - c3[s] for s in (1, 2, 3)])
    assert t["mean_diff"] == pytest.approx(v.mean()) and t["std_diff_ddof1"] == pytest.approx(v.std(ddof=1))
    assert t["pair"] == "C6_minus_C3"
    assert t["n_positive"] + t["n_negative"] == 3
    assert d["col_psnr"]["sign_consistent"] is True       # 30 - 20 = +10 at every seed
    flip = _rows(offset_c6=-100.0)                         # seed 1,3 diff negative, seed 2 positive -> inconsistent
    t2 = [x for x in FU.paired_differences(flip, "C3", "C6", ["tgt_psnr"], [1, 2, 3])][0]
    assert t2["sign_consistent"] is False and t2["n_positive"] == 1 and t2["n_negative"] == 2


def test_paired_difference_zero_is_not_consistent():
    rows = _rows(offset_c6=0.0)
    for r in rows:
        if r["condition"] == "C6":
            r["tgt_psnr"] = [x for x in rows if x["condition"] == "C3" and x["seed"] == r["seed"]][0]["tgt_psnr"]
    t = FU.paired_differences(rows, "C3", "C6", ["tgt_psnr"], [1, 2, 3])[0]
    assert t["n_positive"] == 0 and t["n_negative"] == 0 and t["sign_consistent"] is False


def test_assert_paired_views():
    ok = {"C3_seed1": {1, 2}, "C6_seed1": {2, 1}}
    FU.assert_paired_views(ok, "C3", "C6", [1])
    with pytest.raises(AssertionError):
        FU.assert_paired_views({"C3_seed1": {1, 2}, "C6_seed1": {1, 3}}, "C3", "C6", [1])
    with pytest.raises(AssertionError):
        FU.assert_paired_views({"C3_seed1": {1}}, "C3", "C6", [1])


def test_rank_desc_largest_is_one_and_stable_ties():
    assert FU.rank_desc([3.0, 9.0, 1.0]) == [2, 1, 3]
    assert FU.rank_desc([5.0, 5.0, 1.0]) == [1, 2, 3]


def test_seed_grid_residuals_and_shared_seed_effect():
    rows = []
    for ci, c in enumerate(A.CONDITIONS):
        for s in (1, 2, 3):
            r = {"run_id": "%s_seed%d" % (c, s), "condition": c, "seed": s}
            for col in A.METRIC_COLS:
                r[col] = 100.0 * ci + 5.0 * s            # purely a shared seed effect
            rows.append(r)
    grid, summ, corr = FU.seed_grid(rows, ["tgt_psnr"], A.CONDITIONS, [1, 2, 3])
    assert len(grid) == 24 and len(summ) == 3 and len(corr) == 3
    assert [g["residual_from_condition_mean"] for g in grid if g["seed"] == 1 and g["condition"] == "C1"] == [-5.0]
    assert [s["mean_residual_over_conditions"] for s in summ] == pytest.approx([-5.0, 0.0, 5.0])
    for c in corr:
        assert c["pearson_r_of_residuals"] == pytest.approx(1.0) or np.isnan(c["pearson_r_of_residuals"])
    assert all(g["rank_in_seed_desc"] == 8 - A.CONDITIONS.index(g["condition"]) for g in grid)


def test_mse_from_psnr_roundtrip_and_inf():
    mse = np.array([0.01, 1e-4, 0.5])
    psnr = -10 * np.log10(mse)
    assert FU.mse_from_psnr(psnr) == pytest.approx(mse)
    assert FU.mse_from_psnr(np.array([np.inf, 20.0])).tolist()[0] == 0.0
    assert FU.psnr_from_mse_value(0.01) == pytest.approx(20.0) and FU.psnr_from_mse_value(0.0) == float("inf")


def _pv(psnrs, n_px, plate_mse=None):
    out = []
    for i, (p, n) in enumerate(zip(psnrs, n_px)):
        out.append({"view_index": i, "included": 1.0, "tgt_psnr": p, "col_psnr": p, "n_tgt_px": n, "n_col_px": 1000 - n,
                    "plate_psnr": -10 * np.log10(plate_mse[i]) if plate_mse else p,
                    "plate_mse": plate_mse[i] if plate_mse else 10 ** (-p / 10)})
    return out


def test_aggregates_mean_of_psnr_vs_pooled_mse_differ_as_expected():
    pv = _pv([40.0, 20.0], [100, 300])
    a = FU.run_region_aggregates(pv, "tgt")
    assert a["mean_of_psnr"] == pytest.approx(30.0)
    # view-mean MSE = (1e-4 + 1e-2)/2 -> 23.0 dB ; pixel-weighted = (100e-4 + 300e-2)/400 -> 21.2 dB
    assert a["pooled_mse_view_mean_psnr"] == pytest.approx(-10 * np.log10((1e-4 + 1e-2) / 2))
    assert a["pooled_mse_pixel_weighted_psnr"] == pytest.approx(-10 * np.log10((100 * 1e-4 + 300 * 1e-2) / 400))
    assert a["pooled_mse_pixel_weighted_psnr"] < a["pooled_mse_view_mean_psnr"] < a["mean_of_psnr"]
    same = FU.run_region_aggregates(_pv([30.0, 30.0], [100, 300]), "tgt")
    assert same["pooled_mse_view_mean_psnr"] == pytest.approx(30.0) == pytest.approx(same["mean_of_psnr"])


def test_aggregates_excluded_view_only_dropped_from_target_family():
    pv = _pv([40.0, 20.0], [100, 300])
    pv[1]["included"] = 0.0
    assert FU.run_region_aggregates(pv, "tgt")["n_views"] == 1
    assert FU.run_region_aggregates(pv, "col")["n_views"] == 2


def test_separation_flip_is_detected():
    rows = []
    for c, vals in {"X": [10.0, 10.1, 10.2], "Y": [10.15, 11.0, 11.5]}.items():   # overlap under mean_of_psnr
        for s, v in zip((1, 2, 3), vals):
            r = {"run_id": "%s%d" % (c, s), "condition": c, "seed": s, "region": "tgt"}
            r["mean_of_psnr"] = v
            r["pooled_mse_pixel_weighted_psnr"] = v + (5.0 if c == "Y" else 0.0)       # separated
            r["pooled_mse_view_mean_psnr"] = v
            rows.append(r)
    out = FU.separation_flips(rows, ["X", "Y"], ["tgt"])
    assert len(out) == 1 and out[0]["statement_mean_of_psnr"] == "overlap"
    assert out[0]["statement_pooled_mse_pixel_weighted_psnr"] == "separated" and out[0]["flips_vs_mean_of_psnr"] is True


def test_farther_than_reference_strict_and_counts():
    ref = {0: {"included": 1, "mse": 0.05}, 1: {"included": 1, "mse": 0.05}, 2: {"included": 1, "mse": 0.05}}
    pv = [
        {"view_index": 0, "included": 1.0, "tgt_psnr": -10 * np.log10(0.10), "plate_mse": 0.10},   # farther from both
        {"view_index": 1, "included": 1.0, "tgt_psnr": -10 * np.log10(0.10), "plate_mse": 0.01},   # only GT farther
        {"view_index": 2, "included": 1.0, "tgt_psnr": -10 * np.log10(0.05), "plate_mse": 0.05},   # equal -> not farther
    ]
    r = FU.farther_than_reference(pv, ref)
    assert r["n_views"] == 3 and r["n_both"] == 1 and r["fraction_both"] == pytest.approx(1 / 3)
    assert r["n_gt_only_farther"] == 2 and r["n_plate_only_farther"] == 1
    assert r["max_over_views_min_ratio"] == pytest.approx(2.0)       # view 0: both ratios 2.0 (view 1: 2.0 vs 0.2; view 2: 1.0)


def test_spread_stats():
    s = FU.spread_stats(np.arange(1, 101))
    assert s["iqr"] == pytest.approx(s["q3"] - s["q1"]) and s["range"] == 99 and s["median"] == pytest.approx(50.5)


def test_manifest_reader(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("condition_id,dataset_id,selection_seed,view_index,file_path,poisoned\n"
                 "C3,C3_seed1,1,0,a,True\nC3,C3_seed1,1,1,b,False\nC3,C3_seed1,1,2,c,True\nC0,control,,0,d,False\n")
    got = FU.read_manifest_poisoned(str(p))
    assert got == {"C3_seed1": {0, 2}, "control": set()}


def test_followup_figures_smoke(tmp_path):
    from metrics import figures as F
    from metrics import followup_figures as FF
    rows = _rows()
    diffs = FU.paired_differences(rows, "C3", "C6", ["tgt_psnr", "col_psnr", "plate_mse"], [1, 2, 3])
    f1 = FF.paired_figure(diffs, "C3", "C6", [1, 2, 3], "t")
    pooled = {c: FU.spread_stats(np.arange(10.0)) for c in A.CONDITIONS}
    per_run = {c: [dict(FU.spread_stats(np.arange(10.0)), seed=s) for s in (1, 2, 3)] for c in A.CONDITIONS}
    f2 = FF.spread_figure(pooled, per_run, A.CONDITIONS, "t")
    rng = np.random.default_rng(0)
    pvd = {c: {"mean_acc": rng.random(75), "depth_abs_diff": rng.random(75), "nearest_deg": rng.random(75) * 30}
           for c in ("control", "C3")}
    pvd["control"].pop("depth_abs_diff"); pvd["control"]["nearest_deg"] = None
    f3 = FF.depth_figure(pvd, [10, 20], "t")
    for f, n in zip((f1, f2, f3), "abc"):
        assert all(__import__("os").path.getsize(p) > 0 for p in F.save_fig(f, n, str(tmp_path), {"x": 1})[:2])
