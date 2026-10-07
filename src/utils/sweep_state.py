"""Sweep state, derived from DISK -- there is no separate database (D-040).

A run is
  COMPLETE  iff its final checkpoint, summary.json (written by train.py only
            after training returns), its eval JSON and its results.csv row ALL exist;
  FAILED    iff a FAILED.json marker is in its run directory;
  PARTIAL   iff it has checkpoints (or finished training) but is not COMPLETE;
  PENDING   otherwise.
Every harness action is idempotent against this state, so restarting after a
crash, reboot or kill -9 picks up exactly where the disk says things are.

summary.json is required in addition to the final checkpoint because the periodic
checkpoint at the last iteration (200,000 is a multiple of 25,000) has the final
checkpoint's filename: it can exist while ~1 minute of in-training validation is
still to run and the real final save has not happened.
"""

import csv
import fcntl
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

RESULTS_FIELDS = [
    "run_id", "condition_id", "seed", "budget_percent", "attack_type", "view_selection",
    "dataset_id", "n_poisoned_verified", "train_set_sha256", "config_path", "git_commit",
    "iterations", "final_checkpoint", "eval_json", "n_test_views",
    "test_psnr_mean_whole_frame_db", "final_train_loss", "final_train_psnr",
    "train_wall_hours", "eval_minutes", "resume_events", "retry_events", "log_entry",
    "hardware", "finished_at_utc",
]


@dataclass(frozen=True)
class Paths:
    """All harness bookkeeping lives under runs/ (git-ignored) or logs/, results/ (the two
    prefixes the clean-tree preflight ignores): anything else under experiments/ would
    show as an untracked file and fail the preflight at the second run."""
    root: str

    @property
    def runs(self): return os.path.join(self.root, "runs")
    @property
    def logs(self): return os.path.join(self.root, "logs")
    @property
    def results_csv(self): return os.path.join(self.root, "results", "results.csv")
    @property
    def lockfile(self): return os.path.join(self.runs, ".sweep.lock")
    @property
    def stopped(self): return os.path.join(self.runs, "STOPPED")
    @property
    def stop_after_current(self): return os.path.join(self.runs, "STOP_AFTER_CURRENT")
    @property
    def hw_pin(self): return os.path.join(self.runs, "sweep_hardware.json")
    @property
    def harness_log(self): return os.path.join(self.logs, "sweep_harness.log")

    def run_dir(self, run_id: str) -> str: return os.path.join(self.runs, run_id)


def final_ckpt_name(iterations: int) -> str:
    return "%06d.tar" % iterations


def checkpoints(run_dir: str) -> List[str]:
    return sorted(f for f in os.listdir(run_dir) if f.endswith(".tar")) if os.path.isdir(run_dir) else []


def read_results(csv_path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="") as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def append_results_row(csv_path: str, row: Dict) -> bool:
    """Append-only, idempotent: returns False (writes nothing) if run_id already has a row."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f))
        if header != RESULTS_FIELDS:
            raise RuntimeError("%s has header %s, expected %s -- refusing to append (D-030 lesson)"
                               % (csv_path, header, RESULTS_FIELDS))
        if row["run_id"] in read_results(csv_path):
            return False
        new = False
    else:
        new = True
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULTS_FIELDS})
        f.flush()
        os.fsync(f.fileno())
    return True


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def classify_run(paths: Paths, run_id: str, iterations: int, results: Optional[dict] = None) -> Dict:
    d = paths.run_dir(run_id)
    results = read_results(paths.results_csv) if results is None else results
    cks = checkpoints(d)
    final = final_ckpt_name(iterations)
    has_final = final in cks
    has_summary = os.path.isfile(os.path.join(d, "summary.json"))
    ev = _load_json(os.path.join(d, "eval_%s.json" % run_id))
    has_eval = bool(ev) and str(ev.get("checkpoint", "")).endswith(final) and ev.get("test_psnr") is not None
    has_row = run_id in results
    last_iter = int(cks[-1].split(".")[0]) if cks else 0
    info = {"run_id": run_id, "last_checkpoint_iter": last_iter, "has_final": has_final,
            "has_summary": has_summary, "has_eval": has_eval, "has_row": has_row}
    failed = _load_json(os.path.join(d, "FAILED.json"))
    if failed is not None or os.path.exists(os.path.join(d, "FAILED.json")):
        return {**info, "state": "FAILED", "next": None, "reason": (failed or {}).get("reason", "FAILED.json present")}
    if has_final and has_summary and has_eval and has_row:
        return {**info, "state": "COMPLETE", "next": None}
    if not cks and not has_summary:
        return {**info, "state": "PENDING", "next": "train"}
    if not (has_final and has_summary):
        nxt = "train"
    elif not has_eval:
        nxt = "eval"
    else:
        nxt = "results"
    return {**info, "state": "PARTIAL", "next": nxt}


# -------------------------------------------------------------------- events
def append_event(run_dir: str, kind: str, **fields) -> None:
    os.makedirs(run_dir, exist_ok=True)
    rec = {"t": time.time(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": kind, **fields}
    with open(os.path.join(run_dir, "harness_events.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_events(run_dir: str) -> List[dict]:
    p = os.path.join(run_dir, "harness_events.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a torn last line from a kill -9 mid-write
    return out


# ------------------------------------------------------------------- markers
def write_failed(run_dir: str, reason: str, **details) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "FAILED.json"), "w") as f:
        json.dump({"reason": reason, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}, f, indent=2)


def write_stopped(paths: Paths, run_id: Optional[str], reason: str) -> None:
    os.makedirs(paths.runs, exist_ok=True)
    with open(paths.stopped, "w") as f:
        json.dump({"run_id": run_id, "reason": reason,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "how_to_resume": "investigate, then: python scripts/run_sweep.py clear-stopped "
                                    "(a FAILED run must also have its run directory moved aside)"}, f, indent=2)


def read_stopped(paths: Paths) -> Optional[dict]:
    return _load_json(paths.stopped) if os.path.exists(paths.stopped) else None


def clear_stopped(paths: Paths) -> bool:
    if os.path.exists(paths.stopped):
        os.remove(paths.stopped)
        return True
    return False


# ---------------------------------------------------------------------- lock
class LockHeld(Exception):
    pass


class SweepLock:
    """Single-instance guard: flock(LOCK_EX|LOCK_NB) on a lockfile.

    The OS releases it when the holder dies for any reason, including kill -9,
    so a crashed harness never leaves a stale lock behind.
    """

    def __init__(self, path: str):
        self.path = path
        self.fd = None

    def acquire(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = os.read(self.fd, 64).decode().strip() or "unknown"
            os.close(self.fd)
            self.fd = None
            raise LockHeld("another run_sweep instance holds %s (pid %s); refusing to start a second "
                           "training process" % (self.path, holder))
        os.ftruncate(self.fd, 0)
        os.write(self.fd, ("%d\n" % os.getpid()).encode())

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()


def holder_pid(lockfile: str) -> Optional[int]:
    """PID in the lockfile if (and only if) the lock is actually held right now."""
    if not os.path.exists(lockfile):
        return None
    fd = os.open(lockfile, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            txt = os.read(fd, 64).decode().strip()
            return int(txt) if txt.isdigit() else -1
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)
