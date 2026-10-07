"""`run_sweep.py status`: the 24-run table, ETAs, GPU/disk state and any STOPPED marker."""

import json
import os
import time

from utils import sweep_checks as chk
from utils import sweep_state as st

# Used only to project time for runs with no data of their own: D-038 measured
# 5.83 it/s and ~0.55 h/run of in-training validation + final eval (extrapolated).
DEFAULT_IPS = 5.83
DEFAULT_OVERHEAD_H = 0.55


def _run_points(paths, run_id):
    p = os.path.join(paths.run_dir(run_id), "train.log")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return chk.parse_loss_points(f)


def format_status(sweep) -> str:
    c = sweep.cfg
    paths = c.paths
    queue = sweep.queue()
    results = st.read_results(paths.results_csv)
    rows, remaining_s, cur = [], 0.0, None
    done_ips = []
    for r in queue:
        iters = sweep.iterations_for(r)
        info = st.classify_run(paths, r.run_id, iters, results)
        pts = _run_points(paths, r.run_id)
        ips = chk.throughput(pts)
        ev = st.read_events(paths.run_dir(r.run_id))
        resumes = sum(1 for e in ev if e["event"] == "resume")
        retries = sum(1 for e in ev if e["event"] == "retry")
        it = max(info["last_checkpoint_iter"], pts[-1][0] if pts else 0)
        if info["state"] == "COMPLETE":
            it = iters
            if ips:
                done_ips.append(ips)
        rows.append((r, info, it, ips, pts[-1][1] if pts else None, resumes, retries, iters))
    ref_ips = (sum(done_ips) / len(done_ips)) if done_ips else DEFAULT_IPS
    cur_eta = None
    for r, info, it, ips, loss, resumes, retries, iters in rows:
        if info["state"] in ("PENDING", "PARTIAL"):
            use = ips or ref_ips
            left = (iters - it) / use + DEFAULT_OVERHEAD_H * 3600
            remaining_s += left
            if cur is None and info["state"] == "PARTIAL":
                cur, cur_eta = r.run_id, left
    out = ["%-14s %-9s %9s %7s %10s %7s %7s  %s" % ("run", "state", "iter", "it/s", "last loss", "resumes", "retries", "dataset")]
    for r, info, it, ips, loss, resumes, retries, iters in rows:
        out.append("%-14s %-9s %9s %7s %10s %7d %7d  %s" % (
            r.run_id, info["state"], "%d/%d" % (it, iters), ("%.2f" % ips) if ips else "-",
            ("%.5f" % loss) if loss is not None else "-", resumes, retries, r.dataset_id))
    n = {s: sum(1 for x in rows if x[1]["state"] == s) for s in ("COMPLETE", "PARTIAL", "PENDING", "FAILED")}
    out.append("")
    out.append("COMPLETE %(COMPLETE)d  PARTIAL %(PARTIAL)d  PENDING %(PENDING)d  FAILED %(FAILED)d" % n)
    pid = st.holder_pid(paths.lockfile)
    out.append("harness: %s" % ("RUNNING (pid %s)" % pid if pid else "not running"))
    if cur_eta is not None:
        out.append("current run %s ETA %.1f h (projection from its own recent it/s)" % (cur, cur_eta / 3600))
    out.append("remaining (all runs) ETA %.1f h = %.1f days  [projection: %.2f it/s + %.2f h/run overhead; D-038 figures where a run has no data]"
               % (remaining_s / 3600, remaining_s / 86400, ref_ips, DEFAULT_OVERHEAD_H))
    try:
        g = chk.gpu_snapshot()
        out.append("GPU %s: %s C, SM %s MHz, %s W, %.0f MiB used, persistence %s" % (
            g["name"], g["temp_c"], g["sm_clock"], g["power_w"], g["mem_used_mib"], g["persistence"]))
    except Exception as e:  # noqa: BLE001
        out.append("GPU: unavailable (%s)" % e)
    import shutil
    os.makedirs(paths.runs, exist_ok=True)
    out.append("disk free at %s: %.0f GB" % (paths.runs, shutil.disk_usage(paths.runs).free / 1e9))
    stopped = st.read_stopped(paths)
    if stopped:
        out.append("")
        out.append("*** STOPPED MARKER *** %s" % json.dumps(stopped, indent=2))
    if os.path.exists(paths.stop_after_current):
        out.append("graceful stop requested (STOP_AFTER_CURRENT present): will stop after the current run")
    return "\n".join(out)
