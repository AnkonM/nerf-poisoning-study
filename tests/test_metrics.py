import csv
import os

import cv2
import imageio.v2 as iio
import numpy as np
import pytest

from metrics import compute as C
from metrics import eval_data as E
from metrics import image_metrics as M

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _img(seed=0, h=32):
    return np.random.default_rng(seed).random((h, h, 3)).astype(np.float32)


# ---- PSNR -----------------------------------------------------------------
def test_psnr_identical_is_inf_and_flagged_not_nan():
    a = _img()
    reg = np.ones((32, 32), bool)
    mse, p = M.psnr_region(a, a.copy(), reg)
    assert mse == 0.0 and p == float("inf")


def test_psnr_known_mse():
    a = np.zeros((8, 8, 3), np.float32)
    b = np.full((8, 8, 3), 0.1, np.float32)
    mse, p = M.psnr_region(a, b, np.ones((8, 8), bool))
    assert mse == pytest.approx(0.01, rel=1e-6)
    assert p == pytest.approx(20.0, abs=1e-5)


def test_region_pooling_errors_only_inside_or_outside():
    a = np.zeros((16, 16, 3), np.float32)
    b = a.copy()
    reg = np.zeros((16, 16), bool)
    reg[4:8, 4:8] = True
    b[reg] = 0.5                                   # error only inside the region
    sq = M.sq_err_map(a, b)
    assert M.region_mse(sq, reg) == pytest.approx(0.25)
    assert M.region_mse(sq, ~reg) == 0.0
    assert M.psnr_from_mse(M.region_mse(sq, ~reg)) == float("inf")
    b2 = a.copy()
    b2[~reg] = 0.5                                 # error only outside
    sq2 = M.sq_err_map(a, b2)
    assert M.region_mse(sq2, reg) == 0.0
    assert M.region_mse(sq2, ~reg) == pytest.approx(0.25)


def test_bad_inputs_rejected():
    a = _img()
    reg = np.ones((32, 32), bool)
    with pytest.raises(ValueError):
        M.psnr_region(a, a[:16], reg)                              # shape
    with pytest.raises(ValueError):
        M.psnr_region(a.astype(np.float64), a.astype(np.float64), reg)   # dtype
    bad = a.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        M.psnr_region(bad, a, reg)                                 # NaN
    with pytest.raises(ValueError):
        M.psnr_region(a, a, np.zeros((32, 32), bool))              # empty region
    with pytest.raises(ValueError):
        M.psnr_region(a, a, np.ones((32, 32), np.uint8))           # region dtype


# ---- masks / regions ------------------------------------------------------
def test_dilation_matches_poisoning_expression():
    # the literal expression in scripts/build_poison_set.py
    rng = np.random.default_rng(1)
    m = rng.random((64, 64)) > 0.97
    d = 3
    kernel = np.ones((2 * d + 1,) * 2, np.uint8)
    expected = cv2.dilate(m.astype(np.uint8), kernel).astype(bool)
    assert np.array_equal(E.dilate_mask(m, d), expected)
    src = open(os.path.join(REPO, "scripts", "build_poison_set.py")).read()
    assert "np.ones((2 * dilation + 1,) * 2, np.uint8)" in src
    assert "cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)" in src


def test_downscale_mask_is_area_not_nearest():
    m = np.zeros((8, 8), bool)
    m[0, 0] = True                       # 1 of 4 px in the 2x2 block -> 0.25 -> False
    m[2:4, 2:4] = True                   # full block -> True
    m[4, 4] = m[4, 5] = m[5, 4] = True   # 3 of 4 -> 0.75 -> True
    s = E.downscale_mask(m, 4)
    assert s.shape == (4, 4)
    assert not s[0, 0] and s[1, 1] and s[2, 2]


def test_partition_exact_and_detects_overlap():
    m = np.random.default_rng(2).random((3, 20, 20)) > 0.7
    E.assert_partition(m)
    assert np.all((m.reshape(3, -1).sum(1) + (~m).reshape(3, -1).sum(1)) == 400)


def test_read_eval_mask_rejects_wrong_dtype_and_shape(tmp_path):
    p8 = tmp_path / "m8.png"
    iio.imwrite(p8, np.zeros((800, 800), np.uint8))
    with pytest.raises(ValueError):
        E.read_eval_mask(str(p8))
    p16s = tmp_path / "small.png"
    iio.imwrite(p16s, np.zeros((100, 100), np.uint16))
    with pytest.raises(ValueError):
        E.read_eval_mask(str(p16s))
    pnb = tmp_path / "nb.png"
    arr = np.zeros((800, 800), np.uint16)
    arr[0, 0] = 1234
    iio.imwrite(pnb, arr)
    with pytest.raises(ValueError):
        E.read_eval_mask(str(pnb))
    ok = tmp_path / "ok.png"
    arr = np.zeros((800, 800), np.uint16)
    arr[10:20, 10:20] = 65535
    iio.imwrite(ok, arr)
    assert E.read_eval_mask(str(ok)).sum() == 100


def test_view_inclusion_threshold():
    m = np.zeros((800, 800), bool)
    m.flat[:3200] = True                          # exactly 0.5% of 640000 px
    ok, frac = E.view_inclusion(m, 0.005)
    assert ok and frac == pytest.approx(0.005)
    m.flat[3199] = False
    assert not E.view_inclusion(m, 0.005)[0]


# ---- SSIM / LPIPS ---------------------------------------------------------
def test_ssim_identical_is_one():
    a = _img(3, 40)
    s = M.ssim_map(a, a.copy())
    assert s.shape == (40, 40)
    assert M.region_mean(s, np.ones((40, 40), bool)) == pytest.approx(1.0, abs=1e-6)


def test_ssim_drops_for_different_images():
    a, b = _img(4, 40), _img(5, 40)
    assert M.region_mean(M.ssim_map(a, b), np.ones((40, 40), bool)) < 0.5


def _lpips_or_skip():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA GPU: LPIPS metric tests need one")
        return M.LpipsMeter(batch_size=2)
    except FileNotFoundError as e:
        pytest.skip("LPIPS weights unavailable: %s" % e)


def test_lpips_identical_is_zero_and_different_is_positive():
    meter = _lpips_or_skip()
    a = np.stack([_img(6, 64), _img(7, 64)])
    z = meter.maps(a, a.copy())
    assert z.shape == (2, 64, 64)
    assert np.abs(z).max() < 1e-5
    d = meter.maps(a, a[::-1].copy())
    assert d.mean() > 0.01


def test_lpips_missing_weights_fail_loudly(monkeypatch):
    monkeypatch.setattr(M, "lpips_weight_paths",
                        lambda: {"backbone": "/nonexistent/alexnet.pth", "linear": "/nonexistent/alex.pth"})
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA GPU")
    except ImportError:
        pytest.skip("no torch")
    with pytest.raises(FileNotFoundError):
        M.LpipsMeter()


# ---- per-run CSV ------------------------------------------------------------
def _row(run_id, v):
    r = {c: 0 for c in C.RUN_COLUMNS}
    r.update(run_id=run_id, whole_psnr=float(v))
    return r


def test_per_run_upsert_replaces_not_duplicates(tmp_path):
    p = str(tmp_path / "per_run.csv")
    C.upsert_run_row(p, _row("a", 1.0))
    C.upsert_run_row(p, _row("b", 2.0))
    C.upsert_run_row(p, _row("a", 3.0))
    rows = list(csv.DictReader(open(p)))
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert float(rows[0]["whole_psnr"]) == 3.0
    assert not os.path.exists(p + ".tmp")


def test_per_run_header_mismatch_refused(tmp_path):
    p = tmp_path / "per_run.csv"
    p.write_text("run_id,other\nx,1\n")
    with pytest.raises(ValueError):
        C.upsert_run_row(str(p), _row("a", 1.0))


def test_inf_is_written_explicitly_and_flagged_in_aggregate():
    rows = []
    for i in range(2):
        r = {c: 1.0 for c in C.VIEW_COLUMNS}
        r.update(view_index=i, included=1)
        rows.append(r)
    rows[0]["col_psnr"] = float("inf")
    agg = C.aggregate(rows)
    assert agg["n_inf_psnr_views"] == 1 and agg["col_psnr"] == float("inf")
    assert C._fmt(float("inf")) == "inf"


def test_aggregate_included_only_for_target_metrics():
    rows = []
    for i, (inc, t, c) in enumerate([(1, 10.0, 30.0), (0, 99.0, 50.0)]):
        r = {c_: 0.0 for c_ in C.VIEW_COLUMNS}
        r.update(view_index=i, included=inc, tgt_psnr=t, col_psnr=c)
        rows.append(r)
    agg = C.aggregate(rows)
    assert agg["tgt_psnr"] == 10.0            # excluded view ignored
    assert agg["col_psnr"] == 40.0            # collateral uses all views
    assert agg["n_included"] == 1 and agg["n_views"] == 2
