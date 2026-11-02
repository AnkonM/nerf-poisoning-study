import json
import os

import numpy as np
import pytest

from utils import render_eval as R

H = W = 4


def _write(d, renders, harness_png=True, n_png=75):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, R.NPZ), renders=renders, view_index=np.arange(renders.shape[0]))
    for i in range(n_png):
        open(os.path.join(d, R.view_png_name(i)), "wb").write(b"x")


def _gt():
    rng = np.random.default_rng(0)
    return rng.random((75, H, W, 3)).astype(np.float32)


def test_view_png_name():
    assert R.view_png_name(0) == "eval_000.png" and R.view_png_name(74) == "eval_074.png"


def test_psnr_from_arrays_known_value():
    gt = np.zeros((2, H, W, 3), np.float32)
    pred = gt + 0.1
    assert R.psnr_from_arrays(pred, gt) == pytest.approx(20.0, abs=1e-4)


def test_validate_passes_and_reports_deviation(tmp_path):
    gt = _gt()
    r = (gt + 0.01).astype(np.float16)
    _write(str(tmp_path), r)
    harness = R.psnr_from_arrays(r, gt)
    v = R.validate_renders(str(tmp_path), gt, harness + 0.002)
    assert v["ok"] and v["n_png"] == 75 and v["deviation_db"] == pytest.approx(0.002, abs=1e-9)


def test_validate_fails_on_psnr_deviation_nan_wrong_png_count_and_shape(tmp_path):
    gt = _gt()
    r = (gt + 0.01).astype(np.float16)
    h = R.psnr_from_arrays(r, gt)
    d = tmp_path / "a"; _write(str(d), r)
    assert not R.validate_renders(str(d), gt, h + 0.5)["ok"]
    bad = r.copy(); bad[3, 0, 0, 0] = np.nan
    d = tmp_path / "b"; _write(str(d), bad)
    assert "NaN/inf in renders" in R.validate_renders(str(d), gt, h)["problems"]
    d = tmp_path / "c"; _write(str(d), r, n_png=74)
    assert not R.validate_renders(str(d), gt, h)["ok"]
    d = tmp_path / "d"; _write(str(d), r[:74])
    assert not R.validate_renders(str(d), gt[:74], h)["ok"]


def test_meta_is_valid_resume_logic(tmp_path):
    ckpt = tmp_path / "c.tar"; ckpt.write_bytes(b"abc")
    d = tmp_path / "run"
    gt = _gt(); r = gt.astype(np.float16)
    _write(str(d), r)
    meta = {"validation": {"ok": True}, "eval_manifest_sha256": "M", "checkpoint_sha256": R.sha256_file(str(ckpt)),
            "n_views": 75}
    assert not R.meta_is_valid(str(d), str(ckpt), "M")           # no meta yet -> redo
    (d / R.META).write_text(json.dumps(meta))
    assert R.meta_is_valid(str(d), str(ckpt), "M")
    assert not R.meta_is_valid(str(d), str(ckpt), "OTHER")        # eval set changed
    ckpt.write_bytes(b"abd")
    assert not R.meta_is_valid(str(d), str(ckpt), "M")           # checkpoint changed
    ckpt.write_bytes(b"abc")
    os.remove(str(d / R.view_png_name(5)))
    assert not R.meta_is_valid(str(d), str(ckpt), "M")           # incomplete PNGs
    meta["validation"]["ok"] = False
    (d / R.META).write_text(json.dumps(meta))
    assert not R.meta_is_valid(str(d), str(ckpt), "M")


def test_tmp_dir_name():
    assert R.tmp_dir_for("/x/renders/C1_seed1") == "/x/renders/C1_seed1.partial"
