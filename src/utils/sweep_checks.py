"""Fail-closed preflight checks, training-log parsing and the divergence detector.

Everything here is a plain function over explicit arguments so it can be tested
without a GPU, a git repo or the real data. Nothing here modifies anything.
"""

import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from importlib import metadata
from typing import Dict, List, Optional, Tuple

# Harness-owned outputs live inside the repo (experiments/logs, experiments/results),
# so the "clean tree" check must not count them or run 2 would refuse to start
# because of run 1's log entry. Anything ELSE that differs from HEAD is a failure.
OUTPUT_PREFIXES = ("experiments/logs/", "experiments/results/")

LOSS_RE = re.compile(r"iter (\d+) loss (\S+) psnr (\S+)")


# ---------------------------------------------------------------- log parsing
def parse_loss_points(lines) -> List[Tuple[int, float, float]]:
    """(iteration, loss, timestamp) from harness train.log lines `<epoch> <text>`.

    Later lines win on a repeated iteration (a retry re-logs from its resume
    point). Non-numeric loss tokens ('nan', 'inf') are kept as floats.
    """
    by_iter: Dict[int, Tuple[int, float, float]] = {}
    for line in lines:
        m = LOSS_RE.search(line)
        if not m:
            continue
        try:
            ts = float(line.split(" ", 1)[0])
        except ValueError:
            ts = float("nan")
        by_iter[int(m.group(1))] = (int(m.group(1)), float(m.group(2)), ts)
    return [by_iter[k] for k in sorted(by_iter)]


def throughput(points, recent: int = 4) -> Optional[float]:
    """it/s over the last `recent` windows of consecutive log points (None if <2)."""
    pts = [p for p in points if not math.isnan(p[2])][-(recent + 1):]
    if len(pts) < 2 or pts[-1][2] <= pts[0][2]:
        return None
    return (pts[-1][0] - pts[0][0]) / (pts[-1][2] - pts[0][2])


# ---------------------------------------------------------- divergence detector
def check_loss_health(points, window: int = 20, ratio: float = 2.0) -> Dict:
    """Non-finite loss, or a WINDOWED divergence. Never a single sample.

    Loss at a log point is one random view's batch loss, so it is noisy (and a
    poisoned run's loss legitimately depends on which view was drawn). A
    single-point min/max rule false-positives on healthy runs -- the Phase 3
    lesson. So: compare the MEDIAN of the older half against the median of the
    newer half of the most recent 2*window log points; flag only when the newer
    median exceeds `ratio` x the older. A healthy run decreases (ratio < 1); a
    plateau is ~1; a lone spike moves neither median. A non-finite loss is
    flagged at once, since it does not recover.
    """
    losses = [p[1] for p in points]
    for it, loss, _ in points:
        if not math.isfinite(loss):
            return {"status": "nonfinite", "iteration": it, "detail": "loss=%r at iteration %d" % (loss, it)}
    if len(losses) < 2 * window:
        return {"status": "insufficient", "n_points": len(losses), "detail": "need %d log points" % (2 * window)}
    recent = losses[-2 * window:]
    older, newer = statistics.median(recent[:window]), statistics.median(recent[window:])
    r = newer / older if older > 0 else float("inf") if newer > 0 else 1.0
    status = "diverging" if r > ratio else "ok"
    return {"status": status, "ratio": r, "older_median": older, "newer_median": newer,
            "detail": "median(newer %d)=%.5g vs median(older %d)=%.5g, ratio %.3f (limit %.1f)"
                      % (window, newer, window, older, r, ratio)}


# ------------------------------------------------------------------ preflight
def _run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def check_git_clean(repo_root: str, ignore_prefixes=OUTPUT_PREFIXES) -> Tuple[bool, str, str]:
    """(ok, detail, commit). Refuses any change to the repo outside harness outputs."""
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    commit = head.stdout.strip() if head.returncode == 0 else "unknown"
    st = _run(["git", "status", "--porcelain=v1", "-uall"], cwd=repo_root)
    if st.returncode != 0:
        return False, "git status failed: %s" % st.stderr.strip(), commit
    dirty = []
    for line in st.stdout.splitlines():
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if not path.startswith(tuple(ignore_prefixes)):
            dirty.append(line)
    if dirty:
        return False, "working tree not clean (%d path(s), e.g. %s)" % (len(dirty), dirty[0]), commit
    return True, "clean at %s (ignoring harness outputs under %s)" % (commit[:10], ", ".join(ignore_prefixes)), commit


def check_freeze(repo_root: str, python: str = sys.executable) -> Tuple[bool, str, List[str]]:
    r = _run([python, os.path.join(repo_root, "scripts", "freeze_eval_set.py"), "--verify"], cwd=repo_root)
    out = r.stdout
    warnings = [l for l in out.splitlines() if l.startswith("WARNING")]
    agg = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("recomputed aggregate")), "?")
    ok = r.returncode == 0 and "VERIFY: PASS" in out
    return ok, ("frozen eval set PASS, aggregate %s" % agg) if ok else "freeze_eval_set --verify FAILED:\n" + out[-600:], warnings


def check_dataset(repo_root: str, run, manifest_path: str) -> Tuple[bool, str]:
    d = os.path.join(repo_root, run.dataset_rel)
    train = os.path.join(d, "train")
    if not os.path.isdir(train):
        return False, "dataset dir missing: %s" % train
    n = len([f for f in os.listdir(train) if f.endswith(".png")])
    if n != 100:
        return False, "%s has %d training images, expected 100" % (run.dataset_rel, n)
    if not os.path.isfile(os.path.join(d, "transforms_train.json")):
        return False, "%s has no transforms_train.json" % run.dataset_rel
    with open(manifest_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset_id"] == run.dataset_id]
    flagged = sum(1 for r in rows if r["poisoned"] == "True")
    if len(rows) != 100 or flagged != run.n_poisoned_expected:
        return False, ("manifest for %s: %d rows, %d poisoned; expected 100 rows and "
                       "round(%s/100 x 100) = %d poisoned"
                       % (run.dataset_id, len(rows), flagged, run.budget_percent, run.n_poisoned_expected))
    return True, "%s: 100 images, manifest %d/100 poisoned = round(%s/100 x 100)" % (
        run.dataset_rel, flagged, run.budget_percent)


def check_disk(path: str, min_free_gb: float) -> Tuple[bool, str]:
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free / 1e9
    return free >= min_free_gb, "%.0f GB free at %s (threshold %.0f GB)" % (free, path, min_free_gb)


def gpu_snapshot() -> Dict:
    """name, uuid, driver, memory used (MiB), persistence + any compute processes."""
    q = _run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.used,persistence_mode,"
              "temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"])
    if q.returncode != 0:
        raise RuntimeError("nvidia-smi failed: %s" % q.stderr.strip())
    f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
    apps = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"]).stdout
    procs = [l.strip() for l in apps.splitlines() if l.strip()]
    return {"name": f[0], "uuid": f[1], "driver": f[2], "mem_used_mib": float(f[3]),
            "persistence": f[4], "temp_c": f[5], "sm_clock": f[6], "power_w": f[7], "compute_procs": procs}


def check_gpu(max_mem_mib: float, wait_s: float = 30.0, expect_idle_procs=()) -> Tuple[bool, str, Dict]:
    """No foreign compute process and GPU memory below baseline+margin.

    Polls for up to `wait_s` because a just-killed process's context takes a
    moment to be released.
    """
    deadline = time.time() + wait_s
    while True:
        snap = gpu_snapshot()
        procs = [p for p in snap["compute_procs"] if p not in expect_idle_procs]
        ok = not procs and snap["mem_used_mib"] <= max_mem_mib
        if ok or time.time() >= deadline:
            break
        time.sleep(2)
    if procs:
        return False, "foreign compute process(es) on the GPU: %s" % procs, snap
    if snap["mem_used_mib"] > max_mem_mib:
        return False, "GPU memory in use %.0f MiB exceeds the limit %.0f MiB" % (snap["mem_used_mib"], max_mem_mib), snap
    return True, "%s idle: %.0f MiB used (limit %.0f), no compute processes" % (snap["name"], snap["mem_used_mib"], max_mem_mib), snap


def hardware_identity(gpu: Dict) -> Dict:
    return {"gpu_name": gpu["name"], "gpu_uuid": gpu["uuid"], "driver": gpu["driver"],
            "torch": metadata.version("torch")}


def check_hardware_pin(path: str, ident: Dict) -> Tuple[bool, str]:
    """Pin the sweep to ONE GPU/driver/torch build (D-038's single-machine rule).

    Written on the first run, compared before every later run. A different GPU,
    driver or torch build would add a floating-point confound to every
    between-condition comparison, so a mismatch fails closed.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(ident, f, indent=2)
        return True, "hardware pinned for this sweep: %s" % ident
    with open(path) as f:
        pinned = json.load(f)
    diff = {k: (pinned.get(k), ident.get(k)) for k in ident if pinned.get(k) != ident.get(k)}
    if diff:
        return False, "hardware/software differs from the sweep's pinned identity (pinned, now): %s" % diff
    return True, "matches pinned hardware identity (%s, driver %s, torch %s)" % (
        ident["gpu_name"], ident["driver"], ident["torch"])


def environment_record(gpu: Dict) -> Dict:
    gov = "n/a"
    try:
        gs = sorted({open("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor" % i).read().strip()
                     for i in range(os.cpu_count())})
        gov = ",".join(gs)
    except OSError:
        pass
    cuda = next((s for s in metadata.version("torch").split("+")[1:]), "")
    return {"hostname": os.uname().nodename, "gpu": gpu["name"], "driver": gpu["driver"],
            "persistence_mode": gpu["persistence"], "cpu_governor": gov,
            "torch": metadata.version("torch"), "cuda_build": cuda}
