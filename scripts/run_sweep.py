#!/usr/bin/env python
"""Phase 6 sweep harness CLI -- thin entry point; logic lives in src/utils/sweep_*.py.

    python scripts/run_sweep.py run            # run / resume the 24-run sweep (the default)
    python scripts/run_sweep.py run --dry-run  # show the queue + preflight the next run; run nothing
    python scripts/run_sweep.py status         # 24-run table, ETAs, GPU, disk, STOPPED marker
    python scripts/run_sweep.py queue          # print the seed-major queue only
    python scripts/run_sweep.py stop           # graceful: finish the CURRENT run, then exit
    python scripts/run_sweep.py stop --now     # SIGTERM: stop the child now (resume costs <= 25,000 it)
    python scripts/run_sweep.py clear-stopped  # remove the STOPPED marker after investigating

Defaults write to the REAL sweep paths (experiments/runs, experiments/logs,
experiments/results/results.csv). `--root DIR` redirects all three (and the lock
file) to DIR/{runs,logs,results,...}; `--config-dir DIR` points at scratch configs.
Exit codes: 0 done / graceful stop; 1 harness crash (restart is safe); 10 STOPPED
(queue halted by policy -- do not restart blindly); 11 another instance holds the lock.
"""

import argparse
import os
import signal
import sys
import time

from utils import sweep_state as st
from utils.sweep_runner import EXIT_OK, EXIT_STOPPED, Sweep, SweepConfig
from utils.sweep_status import format_status

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "status", "queue", "stop", "clear-stopped"])
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "experiments"),
                    help="base for runs/, logs/, results/results.csv, sweep.lock (default: the REAL experiments/)")
    ap.add_argument("--config-dir", default=os.path.join(REPO_ROOT, "configs", "poisoning"))
    ap.add_argument("--only", nargs="*", help="restrict to these run ids (sandbox testing)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", action="store_true", help="with `stop`: SIGTERM the running harness")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="SANDBOX ONLY: tolerate a dirty working tree. Never used for the real sweep.")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--poll-s", type=float, default=15.0)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--gpu-mem-max-mib", type=float, default=2048.0)
    ap.add_argument("--gpu-wait-s", type=float, default=30.0)
    a = ap.parse_args()

    paths = st.Paths(os.path.abspath(a.root))
    cfg = SweepConfig(repo_root=REPO_ROOT, paths=paths, config_dir=os.path.abspath(a.config_dir),
                      only=a.only, allow_dirty=a.allow_dirty, dry_run=a.dry_run, max_retries=a.max_retries,
                      poll_s=a.poll_s, min_free_gb=a.min_free_gb, gpu_mem_max_mib=a.gpu_mem_max_mib,
                      gpu_wait_s=a.gpu_wait_s)
    os.makedirs(paths.logs, exist_ok=True)

    def log(msg: str) -> None:       # stdout (-> journal under systemd) AND a file
        line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
        print(line, flush=True)
        with open(paths.harness_log, "a") as f:
            f.write(line + "\n")

    sweep = Sweep(cfg, log=log)

    if a.command == "status":
        print(format_status(sweep))
        return 0
    if a.command == "queue":
        for i, r in enumerate(sweep.queue(), 1):
            print("%2d  %-14s dataset=%-12s poisoned=%d  %s" % (i, r.run_id, r.dataset_id, r.n_poisoned_expected, r.config_path))
        return 0
    if a.command == "clear-stopped":
        print("STOPPED marker removed" if st.clear_stopped(paths) else "no STOPPED marker present")
        return 0
    if a.command == "stop":
        if a.now:
            pid = st.holder_pid(paths.lockfile)
            if not pid or pid < 0:
                print("no running harness found")
                return 1
            os.kill(pid, signal.SIGTERM)
            print("SIGTERM sent to harness pid %d: its training child will be stopped now; the run resumes "
                  "from its latest checkpoint (<= 25,000 iterations redone)" % pid)
        else:
            os.makedirs(paths.runs, exist_ok=True)
            open(paths.stop_after_current, "w").close()
            print("graceful stop requested: the harness finishes the CURRENT run (train + eval + record), then exits")
        return 0

    # run
    if a.dry_run:
        print(format_status(sweep))
        stopped = st.read_stopped(paths)
        if stopped:
            print("\nwould refuse to run: STOPPED marker present")
            return EXIT_STOPPED
        nxt = next((r for r in sweep.queue()
                    if st.classify_run(paths, r.run_id, sweep.iterations_for(r))["state"] != "COMPLETE"), None)
        if nxt is None:
            print("\nnothing to do: all runs COMPLETE")
            return EXIT_OK
        print("\nnext run: %s (dataset %s). Preflight (read-only, nothing is started):" % (nxt.run_id, nxt.dataset_id))
        cfg.gpu_wait_s = 0
        pf = sweep.preflight(nxt, create_pin=False)
        for name, ok, detail in pf["checks"]:
            print("  [%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))
        for w in pf["warnings"]:
            print("  WARNING:", w)
        return 0 if pf["ok"] else 1
    return sweep.run_queue()


if __name__ == "__main__":
    sys.exit(main())
