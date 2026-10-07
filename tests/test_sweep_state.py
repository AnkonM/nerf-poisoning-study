"""State derived from disk: classification, results.csv idempotence, events, markers, lock."""

import json
import os
import subprocess
import sys

import pytest

from utils import sweep_state as st

ITERS = 200000
FINAL = st.final_ckpt_name(ITERS)


@pytest.fixture
def paths(tmp_path):
    return st.Paths(str(tmp_path))


def _mk(paths, rid, ckpts=(), summary=False, evalj=False, row=False, failed=False):
    d = paths.run_dir(rid)
    os.makedirs(d, exist_ok=True)
    for c in ckpts:
        open(os.path.join(d, c), "wb").write(b"x")
    if summary:
        json.dump({"iterations": ITERS}, open(os.path.join(d, "summary.json"), "w"))
    if evalj:
        json.dump({"checkpoint": os.path.join(d, FINAL), "test_psnr": 40.0}, open(os.path.join(d, "eval_%s.json" % rid), "w"))
    if row:
        st.append_results_row(paths.results_csv, {"run_id": rid})
    if failed:
        st.write_failed(d, "boom")


def test_pending_when_nothing_or_only_leftovers(paths):
    assert st.classify_run(paths, "r", ITERS)["state"] == "PENDING"
    os.makedirs(os.path.join(paths.run_dir("r"), "tb"))               # killed before the first checkpoint
    assert st.classify_run(paths, "r", ITERS)["state"] == "PENDING"


def test_partial_variants_and_next_action(paths):
    _mk(paths, "a", ckpts=["025000.tar", "050000.tar"])
    i = st.classify_run(paths, "a", ITERS)
    assert (i["state"], i["next"], i["last_checkpoint_iter"]) == ("PARTIAL", "train", 50000)
    _mk(paths, "b", ckpts=[FINAL])                                      # periodic ckpt at the last iteration, no summary yet
    assert st.classify_run(paths, "b", ITERS)["next"] == "train"
    _mk(paths, "c", ckpts=[FINAL], summary=True)
    assert st.classify_run(paths, "c", ITERS)["next"] == "eval"
    _mk(paths, "d", ckpts=[FINAL], summary=True, evalj=True)
    assert st.classify_run(paths, "d", ITERS)["next"] == "results"


def test_complete_needs_all_four_artifacts(paths):
    _mk(paths, "ok", ckpts=[FINAL], summary=True, evalj=True, row=True)
    assert st.classify_run(paths, "ok", ITERS)["state"] == "COMPLETE"
    for missing in ("ckpt", "summary", "eval", "row"):
        rid = "m_" + missing
        _mk(paths, rid, ckpts=[] if missing == "ckpt" else [FINAL], summary=missing != "summary",
            evalj=missing != "eval", row=missing != "row")
        assert st.classify_run(paths, rid, ITERS)["state"] != "COMPLETE"


def test_eval_of_a_different_checkpoint_does_not_count(paths):
    _mk(paths, "x", ckpts=[FINAL], summary=True)
    json.dump({"checkpoint": "/somewhere/150000.tar", "test_psnr": 40.0},
              open(os.path.join(paths.run_dir("x"), "eval_x.json"), "w"))
    assert st.classify_run(paths, "x", ITERS)["next"] == "eval"


def test_failed_marker_wins(paths):
    _mk(paths, "f", ckpts=[FINAL], summary=True, evalj=True, row=True, failed=True)
    i = st.classify_run(paths, "f", ITERS)
    assert i["state"] == "FAILED" and i["reason"] == "boom"


def test_results_row_is_append_only_and_idempotent(paths):
    assert st.append_results_row(paths.results_csv, {"run_id": "a", "seed": 1})
    assert not st.append_results_row(paths.results_csv, {"run_id": "a", "seed": 1})
    assert st.append_results_row(paths.results_csv, {"run_id": "b"})
    assert list(st.read_results(paths.results_csv)) == ["a", "b"]
    assert open(paths.results_csv).read().count("run_id,condition_id") == 1      # one header


def test_results_csv_with_foreign_header_is_refused(paths):
    os.makedirs(os.path.dirname(paths.results_csv))
    open(paths.results_csv, "w").write("something,else\n1,2\n")
    with pytest.raises(RuntimeError, match="refusing to append"):
        st.append_results_row(paths.results_csv, {"run_id": "a"})
    assert open(paths.results_csv).read() == "something,else\n1,2\n"


def test_events_tolerate_a_torn_last_line(paths):
    d = paths.run_dir("e")
    st.append_event(d, "start", n=1)
    st.append_event(d, "resume", n=2)
    open(os.path.join(d, "harness_events.jsonl"), "a").write('{"event": "torn", "t": 1')   # kill -9 mid-write
    assert [e["event"] for e in st.read_events(d)] == ["start", "resume"]


def test_stopped_marker_round_trip(paths):
    assert st.read_stopped(paths) is None
    st.write_stopped(paths, "C3_seed1", "reason x")
    assert st.read_stopped(paths)["reason"] == "reason x"
    assert st.clear_stopped(paths) and st.read_stopped(paths) is None and not st.clear_stopped(paths)


def test_lock_refuses_a_second_holder_and_releases(paths):
    a = st.SweepLock(paths.lockfile)
    a.acquire()
    assert st.holder_pid(paths.lockfile) == os.getpid()
    with pytest.raises(st.LockHeld, match="refusing to start a second"):
        st.SweepLock(paths.lockfile).acquire()
    a.release()
    assert st.holder_pid(paths.lockfile) is None
    st.SweepLock(paths.lockfile).acquire()                                # free again


def test_lock_is_released_when_the_holder_is_killed(paths):
    code = ("import sys,time; sys.path.insert(0,%r); from utils import sweep_state as st;"
            "l=st.SweepLock(%r); l.acquire(); print('held',flush=True); time.sleep(60)"
            % (os.path.join(os.path.dirname(__file__), "..", "src"), paths.lockfile))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    with pytest.raises(st.LockHeld):
        st.SweepLock(paths.lockfile).acquire()
    p.kill(); p.wait()                                                     # kill -9: no cleanup code runs
    st.SweepLock(paths.lockfile).acquire()                                 # stale lock does NOT linger
