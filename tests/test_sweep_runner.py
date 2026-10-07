"""Retry/stop policy, preflight gating, interrupt handling and idempotence, with a MOCKED subprocess.

No GPU, no training, no real data: a fake Popen produces the same on-disk artifacts and
log lines the real train.py / evaluate.py would, so every branch of the policy is exercised
in milliseconds.
"""

import json
import os
import threading
import time
import zipfile

import pytest

from utils import sweep_checks as chk
from utils import sweep_runner as sr
from utils import sweep_state as st
from utils.sweep_queue import Run

ITERS = 1000
FINAL = st.final_ckpt_name(ITERS)
COMMIT = "c" * 40
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_ckpt(path, good=True):
    if good:
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("data.pkl", b"payload" * 100)
    else:
        with open(path, "wb") as f:
            f.write(b"PK\x03\x04 truncated")


def loss_lines(n, nan_at=None, diverge=False):
    out = []
    for i in range(n):
        loss = 0.01 * (1.0 if not diverge or i < n // 2 else 1.0 + 0.4 * (i - n // 2))
        loss = loss * (0.97 ** i if not diverge else 1.0)
        out.append("[r] iter %d loss %s psnr 20.000\n" % ((i + 1) * 500, "nan" if i == nan_at else "%.5f" % loss))
    return out


class FakeProc:
    def __init__(self, lines, rc, hang=False):
        self.lines, self.rc, self.hang, self.killed = lines, rc, hang, False
        self.done = threading.Event()
        self.stdout = self._gen()

    def _gen(self):
        for l in self.lines:
            yield l
        self.done.set()
        while self.hang and not self.killed:
            time.sleep(0.005)

    def poll(self):
        if self.killed:
            return -15
        return self.rc if (self.done.is_set() and not self.hang) else None

    def terminate(self): self.killed = True
    def kill(self): self.killed = True
    def wait(self, timeout=None): return -15 if self.killed else self.rc


class Launcher:
    """Callable replacing subprocess.Popen. `train_plan` entries: dict(lines, rc, ckpt, finish, hang)."""

    def __init__(self, run):
        self.run, self.calls, self.train_plan = run, [], []
        self.eval_psnr = 42.0

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        script = [c for c in cmd if c.endswith(("train.py", "evaluate.py"))][0]
        if script.endswith("train.py"):
            plan = self.train_plan.pop(0)
            root = cmd[cmd.index("--output-root") + 1]
            rd = os.path.join(root, cmd[cmd.index("--run-id") + 1])
            os.makedirs(rd, exist_ok=True)
            if plan.get("ckpt"):
                make_ckpt(os.path.join(rd, plan["ckpt"]))
            if plan.get("finish"):
                make_ckpt(os.path.join(rd, FINAL))
                json.dump({"iterations": ITERS, "seed": self.run.seed, "condition_id": self.run.condition_id,
                           "dataset_id": self.run.dataset_id, "n_poisoned_verified": self.run.n_poisoned_expected,
                           "n_train_views": 100, "batch_size": 1024, "final_train_loss": 0.001,
                           "final_train_psnr": 30.0, "train_set_sha256": "ab" * 32},
                          open(os.path.join(rd, "summary.json"), "w"))
            return FakeProc(plan.get("lines", loss_lines(3)), plan.get("rc", 0), plan.get("hang", False))
        ckpt = cmd[cmd.index("evaluate.py") + 2] if False else [c for c in cmd if c.endswith(".tar")][0]
        rid = cmd[cmd.index("--run-id") + 1]
        json.dump({"checkpoint": ckpt, "test_psnr": self.eval_psnr, "n_test_views": 75, "elapsed_minutes": 13.6},
                  open(os.path.join(os.path.dirname(ckpt), "eval_%s.json" % rid), "w"))
        return FakeProc(["[eval] done\n"], 0)


PF = {"ok": True, "failed": [], "checks": [("frozen eval set", True, "frozen eval set PASS, aggregate 211a")],
      "warnings": [], "commit": COMMIT, "gpu": {}, "env": {"gpu": "G", "driver": "580", "torch": "2.14", "hostname": "h"}}


def make_run(cid="C3", seed=1, did="C3_seed1", n=20):
    return Run(cid, seed, "configs/poisoning/budget_20.yaml", 20, "random", "hard_erasure", did, n)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(chk, "check_git_clean", lambda root, *a, **k: (True, "clean", COMMIT))
    cfg = sr.SweepConfig(repo_root=REPO, paths=st.Paths(str(tmp_path / "exp")), config_dir="x",
                         poll_s=0.01, gpu_wait_s=0, max_retries=2)
    run = make_run()
    sweep = sr.Sweep(cfg, log=lambda m: None)
    launcher = Launcher(run)
    sweep.popen = launcher
    sweep.preflight_fn = lambda r: dict(PF)
    sweep.queue = lambda: [run]
    sweep.iterations_for = lambda r: ITERS
    return sweep, run, launcher, cfg


def train_calls(l): return [c for c in l.calls if any(x.endswith("train.py") for x in c)]
def eval_calls(l): return [c for c in l.calls if any(x.endswith("evaluate.py") for x in c)]


def test_success_path_completes_with_log_entry_and_one_results_row(env):
    sweep, run, L, cfg = env
    L.train_plan = [{"finish": True, "lines": loss_lines(3)}]
    assert sweep.run_queue() == sr.EXIT_OK
    info = st.classify_run(cfg.paths, run.run_id, ITERS)
    assert info["state"] == "COMPLETE"
    rows = st.read_results(cfg.paths.results_csv)
    assert list(rows) == [run.run_id] and rows[run.run_id]["git_commit"] == COMMIT
    md = open(os.path.join(cfg.paths.logs, run.run_id + ".md")).read()
    assert "Experiment Log — C3_seed1" in md and "not computed" in md and COMMIT in md
    # train.py ran with the sweep's required flags; evaluate.py ran as a separate later process
    t = train_calls(L)[0]
    assert "--skip-final-eval" in t and "--output-root" in t and "--resume" not in t
    assert L.calls.index(t) < L.calls.index(eval_calls(L)[0])


def test_second_run_of_a_complete_run_launches_nothing(env):
    sweep, run, L, cfg = env
    L.train_plan = [{"finish": True}]
    sweep.run_queue()
    n = len(L.calls)
    assert sweep.run_queue() == sr.EXIT_OK and len(L.calls) == n
    assert len(st.read_results(cfg.paths.results_csv)) == 1                   # no duplicate row


def test_crash_is_retried_with_resume_and_recorded(env):
    sweep, run, L, cfg = env
    L.train_plan = [{"rc": -9, "ckpt": "000500.tar", "lines": loss_lines(2)}, {"finish": True}]
    assert sweep.run_queue() == sr.EXIT_OK
    calls = train_calls(L)
    assert len(calls) == 2 and "--resume" not in calls[0] and "--resume" in calls[1]
    ev = st.read_events(cfg.paths.run_dir(run.run_id))
    kinds = [e["event"] for e in ev]
    assert kinds.count("retry") == 1 and kinds.count("resume") == 1 and kinds[0] == "start"
    assert [e for e in ev if e["event"] == "resume"][0]["from_iteration"] == 500
    assert st.read_results(cfg.paths.results_csv)[run.run_id]["resume_events"] == "1"


def test_retries_exhausted_marks_failed_and_stops_the_queue(env):
    sweep, run, L, cfg = env
    L.train_plan = [{"rc": 1, "ckpt": "000500.tar"} for _ in range(3)]
    assert sweep.run_queue() == sr.EXIT_STOPPED
    assert len(train_calls(L)) == 3                                           # 1 + max_retries(2)
    assert st.classify_run(cfg.paths, run.run_id, ITERS)["state"] == "FAILED"
    assert "retries are exhausted" in st.read_stopped(cfg.paths)["reason"]
    assert eval_calls(L) == []


def test_nan_loss_kills_the_run_marks_failed_and_is_not_retried(env):
    sweep, run, L, cfg = env
    L.train_plan = [{"lines": loss_lines(5, nan_at=3), "hang": True, "ckpt": "000500.tar"}]
    assert sweep.run_queue() == sr.EXIT_STOPPED
    assert len(train_calls(L)) == 1                                           # NO retry for non-convergence
    f = json.load(open(os.path.join(cfg.paths.run_dir(run.run_id), "FAILED.json")))
    assert "nonfinite" in f["reason"] and "iteration 2000" in f["reason"]
    assert "FAILED" in open(cfg.paths.stopped).read() or st.read_stopped(cfg.paths)["run_id"] == run.run_id


def test_windowed_divergence_stops_without_retry(env):
    sweep, run, L, cfg = env
    L.train_plan = [{"lines": loss_lines(60, diverge=True), "rc": 0, "finish": True}]
    assert sweep.run_queue() == sr.EXIT_STOPPED
    assert len(train_calls(L)) == 1
    assert "diverging" in st.read_stopped(cfg.paths)["reason"]
    assert eval_calls(L) == []                                                # a diverged run is never evaluated


def test_preflight_failure_stops_before_anything_launches(env):
    sweep, run, L, cfg = env
    sweep.preflight_fn = lambda r: {**PF, "ok": False, "failed": [("git tree clean", "dirty")]}
    assert sweep.run_queue() == sr.EXIT_STOPPED
    assert L.calls == []
    assert "preflight failed" in st.read_stopped(cfg.paths)["reason"]
    assert not os.path.exists(os.path.join(cfg.paths.run_dir(run.run_id), "FAILED.json"))   # not the run's fault


def test_existing_stopped_marker_blocks_the_queue(env):
    sweep, run, L, cfg = env
    st.write_stopped(cfg.paths, "x", "earlier problem")
    assert sweep.run_queue() == sr.EXIT_STOPPED and L.calls == []
    st.clear_stopped(cfg.paths)
    L.train_plan = [{"finish": True}]
    assert sweep.run_queue() == sr.EXIT_OK


def test_a_failed_run_halts_the_queue_until_moved_aside(env):
    sweep, run, L, cfg = env
    st.write_failed(cfg.paths.run_dir(run.run_id), "diverged earlier")
    assert sweep.run_queue() == sr.EXIT_STOPPED and L.calls == []
    assert "FAILED" in st.read_stopped(cfg.paths)["reason"]


def test_second_instance_is_refused(env):
    sweep, run, L, cfg = env
    with st.SweepLock(cfg.paths.lockfile):
        assert sweep.run_queue() == sr.EXIT_LOCKED
    assert L.calls == []


def test_graceful_stop_flag_stops_before_the_next_run(env):
    sweep, run, L, cfg = env
    os.makedirs(cfg.paths.runs, exist_ok=True)
    open(cfg.paths.stop_after_current, "w").close()
    assert sweep.run_queue() == sr.EXIT_OK and L.calls == []
    assert not os.path.exists(cfg.paths.stop_after_current)                   # consumed


def test_interrupt_kills_the_child_and_leaves_a_resumable_partial(env):
    sweep, run, L, cfg = env
    L.train_plan = [{"lines": loss_lines(3), "hang": True, "ckpt": "000500.tar"}]
    threading.Timer(0.3, sweep.terminate.set).start()
    assert sweep.run_queue() == sr.EXIT_OK
    assert st.classify_run(cfg.paths, run.run_id, ITERS)["state"] == "PARTIAL"
    assert any(e["event"] == "interrupted" for e in st.read_events(cfg.paths.run_dir(run.run_id)))
    assert st.read_stopped(cfg.paths) is None                                 # an interrupt is not a failure


def test_truncated_newest_checkpoint_is_quarantined_and_the_previous_one_used(env):
    sweep, run, L, cfg = env
    rd = cfg.paths.run_dir(run.run_id)
    os.makedirs(rd)
    make_ckpt(os.path.join(rd, "000250.tar"))
    make_ckpt(os.path.join(rd, "000500.tar"), good=False)                      # kill -9 during torch.save
    L.train_plan = [{"finish": True}]
    assert sweep.run_queue() == sr.EXIT_OK
    assert os.path.exists(os.path.join(rd, "000500.tar.corrupt")) and not os.path.exists(os.path.join(rd, "000500.tar"))
    assert "--resume" in train_calls(L)[0]
    ev = st.read_events(rd)
    assert any(e["event"] == "checkpoint_quarantined" for e in ev)
    assert [e for e in ev if e["event"] == "resume"][0]["from_iteration"] == 250


def test_killed_before_first_checkpoint_restarts_clean_and_keeps_old_tb(env):
    sweep, run, L, cfg = env
    rd = cfg.paths.run_dir(run.run_id)
    os.makedirs(os.path.join(rd, "tb"))
    L.train_plan = [{"finish": True}]
    assert sweep.run_queue() == sr.EXIT_OK
    assert "--resume" not in train_calls(L)[0]
    assert any(n.startswith("tb.aborted-") for n in os.listdir(rd))
    assert any(e["event"] == "restart_clean" for e in st.read_events(rd))


def test_training_done_but_not_evaluated_runs_only_the_eval(env):
    sweep, run, L, cfg = env
    rd = cfg.paths.run_dir(run.run_id)
    os.makedirs(rd)
    L.train_plan = [{"finish": True}]
    L("--x", ) if False else None
    Launcher.__call__(L, ["python", "-u", "train.py", "--output-root", cfg.paths.runs, "--run-id", run.run_id])  # create artifacts
    L.calls.clear()
    assert sweep.run_queue() == sr.EXIT_OK
    assert train_calls(L) == [] and len(eval_calls(L)) == 1
    assert st.classify_run(cfg.paths, run.run_id, ITERS)["state"] == "COMPLETE"


def test_invalid_eval_output_fails_validation(env):
    sweep, run, L, cfg = env
    L.train_plan = [{"finish": True}]
    L.eval_psnr = float("nan")
    assert sweep.run_queue() == sr.EXIT_STOPPED
    assert "post-run validation failed" in st.read_stopped(cfg.paths)["reason"]
    assert st.read_results(cfg.paths.results_csv) == {}


def test_head_changing_during_a_run_fails_validation(env, monkeypatch):
    sweep, run, L, cfg = env
    L.train_plan = [{"finish": True}]
    monkeypatch.setattr(chk, "check_git_clean", lambda root, *a, **k: (True, "clean", "d" * 40))   # != PF commit
    assert sweep.run_queue() == sr.EXIT_STOPPED
    assert "HEAD changed" in st.read_stopped(cfg.paths)["reason"]


def test_queue_runs_in_order_and_stops_at_the_first_failure(env):
    sweep, run, L, cfg = env
    r2 = make_run("C4", 1, "C4_seed1", 30)
    r3 = make_run("C5", 1, "C5_seed1", 50)
    sweep.queue = lambda: [run, r2, r3]
    L.train_plan = [{"finish": True}, {"rc": 1}, {"rc": 1}, {"rc": 1}]
    assert sweep.run_queue() == sr.EXIT_STOPPED
    states = [st.classify_run(cfg.paths, r.run_id, ITERS)["state"] for r in (run, r2, r3)]
    assert states == ["COMPLETE", "FAILED", "PENDING"]                         # r3 never started
