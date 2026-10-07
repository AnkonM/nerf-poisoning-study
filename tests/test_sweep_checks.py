"""Divergence detector, log parsing and the fail-closed preflight pieces."""

import csv
import json
import math
import os
import random
import subprocess

import pytest

from utils import sweep_checks as chk
from utils.sweep_queue import Run


def pts(losses, step=500):
    return [((i + 1) * step, l, 1000.0 + i * 85.0) for i, l in enumerate(losses)]


def test_parse_loss_points_keeps_nan_and_last_duplicate():
    lines = ["1.0 [r] iter 500 loss 0.03000 psnr 18.0", "2.0 noise", "3.0 [r] iter 1000 loss nan psnr nan",
             "4.0 [r] iter 500 loss 0.02000 psnr 19.0"]
    p = chk.parse_loss_points(lines)
    assert [x[0] for x in p] == [500, 1000] and p[0][1] == 0.02 and math.isnan(p[1][1])


@pytest.mark.parametrize("seed", range(20))
def test_healthy_noisy_decaying_run_is_never_flagged(seed):
    rng = random.Random(seed)
    losses = [0.03 * math.exp(-i / 12) * math.exp(rng.gauss(0, 0.6)) + 5e-4 for i in range(120)]   # decaying, very noisy
    assert chk.check_loss_health(pts(losses))["status"] == "ok"


def test_healthy_plateau_with_noise_and_a_lone_spike_is_not_flagged():
    rng = random.Random(1)
    losses = [2e-3 * math.exp(rng.gauss(0, 0.5)) for _ in range(60)]
    losses[50] = 5.0                                                       # one wild batch, never a divergence
    assert chk.check_loss_health(pts(losses))["status"] == "ok"


def test_nan_and_inf_are_flagged_immediately():
    for bad in (float("nan"), float("inf")):
        h = chk.check_loss_health(pts([0.01, 0.009, bad]))
        assert h["status"] == "nonfinite" and h["iteration"] == 1500


def test_a_genuinely_diverging_curve_is_flagged():
    rng = random.Random(2)
    losses = [0.002 * math.exp(rng.gauss(0, 0.3)) for _ in range(20)] + \
             [0.002 * math.exp(i / 3) * math.exp(rng.gauss(0, 0.3)) for i in range(20)]
    h = chk.check_loss_health(pts(losses))
    assert h["status"] == "diverging" and h["ratio"] > 2


def test_too_few_points_is_insufficient_not_ok():
    assert chk.check_loss_health(pts([0.01] * 10))["status"] == "insufficient"


def test_throughput_from_timestamps():
    assert chk.throughput(pts([0.01] * 6)) == pytest.approx(500 / 85.0)
    assert chk.throughput([]) is None


# ---- git cleanliness -------------------------------------------------------
@pytest.fixture
def repo(tmp_path):
    def g(*a): subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q"); (tmp_path / "a.py").write_text("x"); g("add", "."); g("commit", "-qm", "i")
    return tmp_path


def test_clean_tree_passes_and_returns_the_commit(repo):
    ok, d, commit = chk.check_git_clean(str(repo))
    assert ok and len(commit) == 40


def test_modified_tracked_file_fails(repo):
    (repo / "a.py").write_text("changed")
    assert chk.check_git_clean(str(repo))[0] is False


def test_untracked_code_file_fails_but_harness_outputs_do_not(repo):
    os.makedirs(repo / "experiments" / "logs"); os.makedirs(repo / "experiments" / "results")
    (repo / "experiments" / "logs" / "r.md").write_text("log"); (repo / "experiments" / "results" / "results.csv").write_text("x")
    assert chk.check_git_clean(str(repo))[0] is True               # run 1's own outputs must not block run 2
    (repo / "new_module.py").write_text("x")
    ok, d, _ = chk.check_git_clean(str(repo))
    assert ok is False and "new_module.py" in d


# ---- dataset ---------------------------------------------------------------
def _run(did="C3_seed1", budget=20, n=20):
    return Run("C3", 1, "c.yaml", budget, "random", "hard_erasure", did, n)


def _dataset(root, did, n_imgs=100, flagged=20, rows=100):
    d = root / "data" / "poisoned" / did
    os.makedirs(d / "train")
    for i in range(n_imgs):
        (d / "train" / ("r_%03d.png" % i)).write_bytes(b"x")
    (d / "transforms_train.json").write_text("{}")
    m = root / "MANIFEST.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_id", "poisoned"]); w.writeheader()
        for i in range(rows):
            w.writerow({"dataset_id": did, "poisoned": str(i < flagged)})
    return str(m)


def test_dataset_check_passes_and_each_failure_mode_fails(tmp_path):
    m = _dataset(tmp_path, "C3_seed1")
    assert chk.check_dataset(str(tmp_path), _run(), m)[0]
    assert not chk.check_dataset(str(tmp_path), _run(did="C3_seed2"), m)[0]                  # dir missing
    assert not chk.check_dataset(str(tmp_path), _run(n=19), m)[0]                            # wrong poisoned count
    tmp2 = tmp_path / "b"; os.makedirs(tmp2)
    assert not chk.check_dataset(str(tmp2), _run(), _dataset(tmp2, "C3_seed1", n_imgs=99))[0]  # 99 images
    tmp3 = tmp_path / "c"; os.makedirs(tmp3)
    assert not chk.check_dataset(str(tmp3), _run(), _dataset(tmp3, "C3_seed1", rows=99, flagged=20))[0]


# ---- disk / GPU / hardware pin ---------------------------------------------
def test_disk_threshold(tmp_path):
    assert chk.check_disk(str(tmp_path), 0.001)[0]
    assert not chk.check_disk(str(tmp_path), 1e9)[0]


def _snap(procs=(), mem=600.0):
    return {"name": "G", "uuid": "U", "driver": "580", "mem_used_mib": mem, "persistence": "Enabled",
            "temp_c": "50", "sm_clock": "300", "power_w": "9", "compute_procs": list(procs)}


def test_gpu_check(monkeypatch):
    monkeypatch.setattr(chk, "gpu_snapshot", lambda: _snap())
    assert chk.check_gpu(2048, wait_s=0)[0]
    monkeypatch.setattr(chk, "gpu_snapshot", lambda: _snap(procs=["123, python, 3000"]))
    ok, d, _ = chk.check_gpu(2048, wait_s=0)
    assert not ok and "foreign compute process" in d
    monkeypatch.setattr(chk, "gpu_snapshot", lambda: _snap(mem=9000))
    ok, d, _ = chk.check_gpu(2048, wait_s=0)
    assert not ok and "exceeds" in d


def test_hardware_pin_created_then_enforced(tmp_path):
    p = str(tmp_path / "pin.json")
    ident = {"gpu_name": "Quadro RTX 5000", "gpu_uuid": "U1", "driver": "580.173.02", "torch": "2.14.0+cu130"}
    assert chk.check_hardware_pin(p, ident)[0] and os.path.exists(p)
    assert chk.check_hardware_pin(p, ident)[0]
    for k, v in (("gpu_uuid", "U2"), ("driver", "590.1"), ("torch", "2.15.0")):
        ok, d = chk.check_hardware_pin(p, {**ident, k: v})
        assert not ok and k in d
