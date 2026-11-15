"""Phase 9 laptop queue (src/utils/phase9_harness.py): queue order and size, hardware-profile enforcement, results_phase9
separation, with a MOCKED subprocess (same fake Popen as the Phase 6 runner tests). No GPU, no training, no real data."""

import copy
import csv
import importlib.metadata as md
import os
import socket

import pytest

from test_sweep_runner import COMMIT, ITERS, PF, Launcher, loss_lines  # fake Popen + helpers from the Phase 6 tests
from utils import phase9_harness as H
from utils import sweep_checks as chk
from utils import sweep_runner as sr
from utils import sweep_state as st

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE_PATH = os.path.join(REPO, "configs", "phase9", "laptop_phase9.yaml")


def synthetic_manifest(tmp_path, datasets):
    p = tmp_path / "MANIFEST.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition_id", "dataset_id", "selection_seed", "view_index", "poisoned"])
        w.writeheader()
        for cond, did, seed in datasets:
            for i in range(100):
                w.writerow({"condition_id": cond, "dataset_id": did, "selection_seed": seed, "view_index": i, "poisoned": "True"})
    return str(p)


def good_manifest(tmp_path):
    return synthetic_manifest(tmp_path, [("C3", "C3_seed1", "1")] + [("B100", "B100_seed%d" % s, str(s)) for s in (1, 2, 3)])


def test_queue_order_and_size(tmp_path):
    prof = H.load_profile(PROFILE_PATH)
    q = H.build_phase9_queue(prof, REPO, good_manifest(tmp_path))
    assert [r.run_id for r in q] == ["lp9_C3_seed1", "lp9_B100_seed1", "lp9_B100_seed2", "lp9_B100_seed3"]
    assert [r.role for r in q] == ["calibration", "option", "option", "option"]
    assert [r.n_poisoned_expected for r in q] == [20, 100, 100, 100]
    assert [r.dataset_id for r in q] == ["C3_seed1", "B100_seed1", "B100_seed2", "B100_seed3"]
    # no Phase 6 run id can collide with a Phase 9 one
    assert not any(r.run_id in ("C3_seed1", "B100_seed1") for r in q)


def test_optional_calibration_seeds_are_not_queued_unless_enabled(tmp_path):
    prof = H.load_profile(PROFILE_PATH)
    assert any(e["enabled"] is False for e in prof["queue"]["optional"])
    prof2 = copy.deepcopy(prof)
    for e in prof2["queue"]["optional"]:
        e["enabled"] = True
    m = synthetic_manifest(tmp_path, [("C3", "C3_seed%d" % s, str(s)) for s in (1, 2, 3)] + [("B100", "B100_seed%d" % s, str(s)) for s in (1, 2, 3)])
    assert len(H.build_phase9_queue(prof2, REPO, m)) == 6


def test_queue_refuses_a_manifest_that_disagrees_with_the_dataset_rule(tmp_path):
    prof = H.load_profile(PROFILE_PATH)
    bad = synthetic_manifest(tmp_path, [("C3", "C3_seed1", "1"), ("B100", "B100_seed1", "2")])
    with pytest.raises(RuntimeError):
        H.build_phase9_queue(prof, REPO, bad)


def test_profile_enforcement_fails_closed_on_each_difference():
    prof = H.load_profile(PROFILE_PATH)
    hw = prof["hardware"]
    gpu = {"name": hw["gpu_name"], "driver": hw["driver"]}
    assert H.check_profile(prof, gpu, hw["hostname"], hw["torch"])[0] is True
    assert H.check_profile(prof, dict(gpu, name="Quadro RTX 5000"), hw["hostname"], hw["torch"])[0] is False
    assert H.check_profile(prof, dict(gpu, driver="999.99"), hw["hostname"], hw["torch"])[0] is False
    assert H.check_profile(prof, gpu, "some-other-host", hw["torch"])[0] is False
    ok, detail = H.check_profile(prof, gpu, hw["hostname"], "2.0.0+cpu")
    assert ok is False and "torch" in detail


def _sweep(tmp_path, monkeypatch, gpu_name=None):
    prof = copy.deepcopy(H.load_profile(PROFILE_PATH))
    prof["hardware"].update(gpu_name="TestGPU", driver="1.0", torch=md.version("torch"), hostname=socket.gethostname())
    monkeypatch.setattr(chk, "check_git_clean", lambda root, *a, **k: (True, "clean", COMMIT))
    monkeypatch.setattr(chk, "check_freeze", lambda *a, **k: (True, "frozen ok", []))
    monkeypatch.setattr(chk, "check_dataset", lambda *a, **k: (True, "dataset ok"))
    snap = {"name": gpu_name or "TestGPU", "uuid": "GPU-1", "driver": "1.0", "mem_used_mib": 500.0, "persistence": "n/a",
            "temp_c": "50", "sm_clock": "1", "power_w": "1", "compute_procs": []}
    monkeypatch.setattr(chk, "check_gpu", lambda *a, **k: (True, "gpu ok", snap))
    paths = H.phase9_paths(str(tmp_path / "exp"), prof)
    cfg = sr.SweepConfig(repo_root=REPO, paths=paths, config_dir="x", poll_s=0.01, gpu_wait_s=0, max_retries=1,
                         manifest_path=good_manifest(tmp_path))
    return H.Phase9Sweep(cfg, prof, log=lambda m: None), prof, cfg


def test_preflight_applies_the_profile_and_blocks_a_different_gpu(tmp_path, monkeypatch):
    sweep, prof, cfg = _sweep(tmp_path, monkeypatch)
    run = sweep.queue()[0]
    res = sweep.preflight(run)
    names = {n: ok for n, ok, _ in res["checks"]}
    assert names["hardware profile"] is True and names["hardware pin"] is True and res["ok"]
    assert os.path.exists(cfg.paths.hw_pin) and os.path.basename(cfg.paths.hw_pin) == "phase9_hardware.json"
    (tmp_path / "other").mkdir()
    sweep2, _, _ = _sweep(tmp_path / "other", monkeypatch, gpu_name="Quadro RTX 5000")
    res2 = sweep2.preflight(sweep2.queue()[0])
    assert res2["ok"] is False and any(n == "hardware profile" for n, _ in res2["failed"])


def test_results_go_to_results_phase9_only_with_the_profile_column(tmp_path, monkeypatch):
    sweep, prof, cfg = _sweep(tmp_path, monkeypatch)
    run = sweep.queue()[0]
    L = Launcher(run)
    L.train_plan = [{"finish": True, "lines": loss_lines(3)}]
    sweep.popen = L
    sweep.preflight_fn = lambda r: dict(PF)
    sweep.iterations_for = lambda r: ITERS
    sweep.cfg.only = [run.run_id]
    assert sweep.run_queue() == sr.EXIT_OK
    p9 = cfg.paths.results_csv
    assert os.path.basename(p9) == "results_phase9.csv" and os.path.exists(p9)
    assert not os.path.exists(os.path.join(cfg.paths.root, "results", "results.csv"))          # Phase 6 file never created
    rows = list(csv.DictReader(open(p9)))
    assert [r["run_id"] for r in rows] == ["lp9_C3_seed1"] and rows[0]["hardware_profile"] == "laptop_phase9"
    assert list(rows[0])[:len(st.RESULTS_FIELDS)] == st.RESULTS_FIELDS                           # same schema + 1 column
    md_text = open(os.path.join(cfg.paths.logs, "lp9_C3_seed1.md")).read()
    assert "Phase 9 extension" in md_text and "Experiment Log — lp9_C3_seed1" in md_text and "HPC machine" not in md_text
    # train.py got the prefixed run id; the Phase 6 lock/STOPPED/pin files are not the Phase 9 ones
    t = [c for c in L.calls if c[1] == "-u"][0]
    assert t[t.index("--run-id") + 1] == "lp9_C3_seed1"
    assert os.path.basename(cfg.paths.lockfile) == ".phase9.lock" and os.path.basename(cfg.paths.stopped) == "PHASE9_STOPPED"
    assert cfg.paths.lockfile != st.Paths(cfg.paths.root).lockfile


def test_phase6_paths_and_record_defaults_are_unchanged():
    p = st.Paths("/x")
    assert p.results_csv == "/x/results/results.csv" and p.hw_pin.endswith("sweep_hardware.json")
    import inspect
    from utils.sweep_record import write_run_record
    sig = inspect.signature(write_run_record)
    assert all(sig.parameters[k].default is None for k in ("where_run", "note", "row_extra", "append_fn"))


# ---- D-047: eval-only chunk override --------------------------------------------------------
def test_eval_chunk_size_defaults_to_chunk_size_and_override_wins():
    from nerf.training import eval_chunk_size
    assert eval_chunk_size({"chunk_size": 32768}) == 32768                      # Phase 6 behaviour: key absent
    assert eval_chunk_size({"chunk_size": 32768, "eval_chunk_size": None}) == 32768
    assert eval_chunk_size({"chunk_size": 32768, "eval_chunk_size": 2048}) == 2048


def test_phase9_passes_the_eval_chunk_to_train_and_evaluate_only(tmp_path, monkeypatch):
    sweep, prof, cfg = _sweep(tmp_path, monkeypatch)
    assert prof["eval"]["eval_chunk_size"] == 2048
    assert H.train_extra_args(prof) == ["--eval-chunk-size", "2048"]
    assert sweep._eval_extra_args() == ["--eval-chunk-size", "2048"]
    plain = sr.Sweep(sr.SweepConfig(repo_root=REPO, paths=st.Paths(str(tmp_path / "p6")), config_dir="x"), log=lambda m: None)
    assert plain._eval_extra_args() == [] and "--eval-chunk-size" not in plain._train_cmd(
        __import__("test_sweep_runner").make_run(), False)


def test_train_and_evaluate_scripts_accept_the_flag_with_default_none():
    import re
    for s in ("train.py", "evaluate.py"):
        src = open(os.path.join(REPO, "scripts", s)).read()
        assert re.search(r'"--eval-chunk-size",\s*type=int,\s*default=None', src)
