#!/usr/bin/env python
"""Phase 9 laptop queue CLI (DECISION_LOG.md D-045/D-046). Thin entry point; logic in src/utils/phase9_harness.py,
which layers over the Phase 6 sweep code without changing it. OPT-IN: --hardware-profile is required.

    python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 queue        # print the queue
    python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 run --dry-run # preflight the next run; start nothing
    python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 run           # run / resume the queue
    python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 status
    python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 stop [--now]
    python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 clear-stopped

Writes experiments/runs/lp9_*, experiments/logs/lp9_*.md and experiments/results/results_phase9.csv. NEVER writes
results.csv. `--root DIR` redirects runs/, logs/, results/ (sandbox testing); `--profile-config FILE` replaces the profile file.
Exit codes as the Phase 6 harness: 0 done / graceful stop; 1 crash (restart is safe); 10 STOPPED; 11 lock held.
"""

import argparse
import json
import os
import signal
import sys
import time

from utils import phase9_harness as H
from utils import sweep_state as st
from utils.sweep_runner import EXIT_OK, EXIT_STOPPED, SweepConfig
from utils.sweep_status import format_status

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "status", "queue", "stop", "clear-stopped"])
    ap.add_argument("--hardware-profile", required=True, help="profile name, e.g. laptop_phase9 (configs/phase9/<name>.yaml)")
    ap.add_argument("--profile-config", default=None)
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "experiments"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="SANDBOX ONLY")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--poll-s", type=float, default=15.0)
    a = ap.parse_args()

    ppath = a.profile_config or os.path.join(REPO_ROOT, "configs", "phase9", a.hardware_profile + ".yaml")
    prof = H.load_profile(ppath)
    if prof["profile"] != a.hardware_profile:
        sys.exit("profile file %s declares %r, not %r" % (ppath, prof["profile"], a.hardware_profile))
    paths = H.phase9_paths(a.root, prof)
    pf = prof["preflight"]
    cfg = SweepConfig(repo_root=REPO_ROOT, paths=paths, config_dir=os.path.join(REPO_ROOT, prof["queue"]["config_dir"]),
                      only=a.only, allow_dirty=a.allow_dirty, dry_run=a.dry_run, max_retries=a.max_retries,
                      poll_s=a.poll_s, min_free_gb=float(pf["min_free_gb"]), gpu_mem_max_mib=float(pf["gpu_mem_max_mib"]),
                      gpu_wait_s=float(pf["gpu_wait_s"]), extra_train_args=H.train_extra_args(prof))
    os.makedirs(paths.logs, exist_ok=True)

    def log(msg):
        line = "%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
        print(line, flush=True)
        with open(paths.harness_log, "a") as f:
            f.write(line + "\n")

    sweep = H.Phase9Sweep(cfg, prof, log=log)

    if a.command == "queue":
        for i, r in enumerate(sweep.queue(), 1):
            print("%2d  %-16s %-12s dataset=%-12s poisoned=%d  role=%s  %s" % (i, r.run_id, r.condition_id, r.dataset_id,
                                                                       r.n_poisoned_expected, r.role, r.config_path))
        return 0
    if a.command == "status":
        print(format_status(sweep))
        print("\nNOTE: the ETA lines above use Phase 6 (HPC) constants (5.83 it/s, 0.55 h/run overhead). Laptop speed test "
              "(D-046): 5.36 it/s and ~4.4 h/run of validation + final eval => ~14.7 h per run.")
        return 0
    if a.command == "clear-stopped":
        print("STOPPED marker removed" if st.clear_stopped(paths) else "no STOPPED marker present")
        return 0
    if a.command == "stop":
        if a.now:
            pid = st.holder_pid(paths.lockfile)
            if not pid or pid < 0:
                print("no running Phase 9 harness found")
                return 1
            os.kill(pid, signal.SIGTERM)
            print("SIGTERM sent to pid %d: the training child is stopped now; the run resumes from its latest checkpoint "
                  "(<= 25,000 iterations redone)" % pid)
        else:
            os.makedirs(paths.runs, exist_ok=True)
            open(paths.stop_after_current, "w").close()
            print("graceful stop requested: the harness finishes the CURRENT run (train + eval + record), then exits")
        return 0

    if a.dry_run:
        print(format_status(sweep))
        if st.read_stopped(paths):
            print("\nwould refuse to run: STOPPED marker present")
            return EXIT_STOPPED
        nxt = next((r for r in sweep.queue()
                    if st.classify_run(paths, r.run_id, sweep.iterations_for(r))["state"] != "COMPLETE"), None)
        if nxt is None:
            print("\nnothing to do: all runs COMPLETE")
            return EXIT_OK
        print("\nnext run: %s (dataset %s). Preflight (read-only, nothing is started):" % (nxt.run_id, nxt.dataset_id))
        cfg.gpu_wait_s = 0
        res = sweep.preflight(nxt, create_pin=False)
        for name, ok, detail in res["checks"]:
            print("  [%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))
        for w in res["warnings"]:
            print("  WARNING:", w)
        return 0 if res["ok"] else 1
    return sweep.run_queue()


if __name__ == "__main__":
    sys.exit(main())
