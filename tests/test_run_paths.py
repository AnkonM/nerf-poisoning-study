"""--output-root (D-037): run-directory resolution and the resume guard.

Default behaviour must be provably identical to the old hard-coded expression,
and the "refuse a populated run dir without --resume" guard must hold wherever
the run directory is.
"""

import os
import subprocess
import sys

from utils.run_paths import existing_checkpoints, resolve_run_dir

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_default_matches_old_expression():
    for rid in ("control_seed1", "C3_seed2", "lego_sanity"):
        old = os.path.join(REPO, "experiments", "runs", rid)
        assert resolve_run_dir(REPO, rid) == old
        assert resolve_run_dir(REPO, rid, None) == old


def test_override_resolves_under_given_root(tmp_path):
    got = resolve_run_dir(REPO, "control_seed1", str(tmp_path))
    assert got == os.path.join(str(tmp_path), "control_seed1")
    assert not got.startswith(os.path.join(REPO, "experiments"))


def test_override_expands_user_and_makes_absolute(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_run_dir(REPO, "r", "~/x") == os.path.join(str(tmp_path), "x", "r")
    monkeypatch.chdir(tmp_path)
    assert resolve_run_dir(REPO, "r", "rel") == os.path.join(str(tmp_path), "rel", "r")


def test_resume_finds_checkpoints_under_override_root(tmp_path):
    run_dir = resolve_run_dir(REPO, "r1", str(tmp_path))
    assert existing_checkpoints(run_dir) == []          # nonexistent dir
    os.makedirs(run_dir)
    for name in ("050000.tar", "025000.tar", "notes.txt"):
        open(os.path.join(run_dir, name), "wb").close()
    assert existing_checkpoints(run_dir) == ["025000.tar", "050000.tar"]


def _run_train(*extra):
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "train.py"),
         os.path.join(REPO, "configs", "scenes", "lego_sanity.yaml"),
         "--allow-dirty", "--run-id", "guard_test", *extra],
        capture_output=True, text=True, timeout=120, cwd=REPO,
        env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "src")))


def test_populated_run_dir_refused_without_resume_under_override(tmp_path):
    run_dir = resolve_run_dir(REPO, "guard_test", str(tmp_path))
    os.makedirs(run_dir)
    open(os.path.join(run_dir, "025000.tar"), "wb").close()
    r = _run_train("--output-root", str(tmp_path))
    assert r.returncode != 0
    assert "Run directory already holds checkpoints" in (r.stderr + r.stdout)
    assert str(tmp_path) in (r.stderr + r.stdout)       # names the overridden dir
    # the guard fired before anything was trained or written
    assert os.listdir(run_dir) == ["025000.tar"]
