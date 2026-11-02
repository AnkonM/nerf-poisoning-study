#!/usr/bin/env python
"""Render the 75 frozen eval views from final checkpoints (thin CLI; logic in src/utils/render_eval.py).

    python scripts/render_eval_views.py --all                 # sequential, resumable, one child process per run
    python scripts/render_eval_views.py --run-id C3_seed2     # one run, in this process
Output: experiments/renders/<run_id>/ (git-ignored). Needs the GPU free of other compute processes.
"""

import argparse
import os
import sys
import time

from utils import render_eval as re_
from utils.sweep_queue import build_queue

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--run-id")
    ap.add_argument("--renders-root", default=os.path.join(REPO_ROOT, "experiments", "renders"))
    a = ap.parse_args()
    os.makedirs(a.renders_root, exist_ok=True)

    if a.run_id:
        meta = re_.render_run(REPO_ROOT, a.run_id, a.renders_root)
        print("OK %s max dev fp16 %.5f dB, fp32 %.5f dB" % (a.run_id, meta["validation"]["deviation_db"],
                                                             meta["validation"]["deviation_fp32_db"]))
        return 0

    def log(msg):
        print("%s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg), flush=True)

    ids = [r.run_id for r in build_queue(os.path.join(REPO_ROOT, "configs", "poisoning"))]
    status = re_.run_all(REPO_ROOT, ids, a.renders_root, log)
    bad = [k for k, v in status.items() if v == "FAILED"]
    log("SUMMARY: %s" % status)
    if bad:
        log("FAILED RUNS (left for the project lead): %s" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
