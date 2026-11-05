#!/usr/bin/env python
"""Package Phase 6 outputs for manual transfer (thin CLI; logic in src/utils/transfer_pack.py). Read-only on sources.

    package_transfer.py checkpoints   # experiments/results/phase6_final_checkpoints.csv
    package_transfer.py runs          # ~/transfer/phase6_runs_<n>.zip
    package_transfer.py renders       # ~/transfer/phase6_renders_<n>.zip  (only when all 24 renders are validated)
    package_transfer.py manifest      # ~/transfer/TRANSFER_MANIFEST.sha256 (runs + renders)
"""

import argparse
import json
import os
import sys

from utils import transfer_pack as T
from utils.sweep_queue import build_queue

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.expanduser("~/transfer")


def _group_files(root):
    groups = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        groups[name] = T.list_files(p) if os.path.isdir(p) else [p]
    return groups


def _zip_groups(root, prefix):
    groups = _group_files(root)
    sizes = {g: sum(os.path.getsize(f) for f in fs) for g, fs in groups.items()}
    over = [g for g, s in sizes.items() if s > T.LIMIT_BYTES]
    if over:
        print("WARNING: single group exceeds limit:", over)
    os.makedirs(OUT, exist_ok=True)
    for i, part in enumerate(T.plan_parts(sizes), 1):
        files = [f for g in part for f in groups[g]]
        zp = os.path.join(OUT, "%s_%d.zip" % (prefix, i))
        n, size = T.write_zip(zp, REPO, files)
        print("%s: %d files, %d bytes (%.3f GB), groups=%d [%s .. %s]" % (zp, n, size, size / 1e9, len(part), part[0], part[-1]))
        if size >= T.LIMIT_BYTES:
            sys.exit("zip %s is %d bytes >= limit" % (zp, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["checkpoints", "runs", "renders", "manifest"])
    a = ap.parse_args()
    runs = os.path.join(REPO, "experiments", "runs")
    renders = os.path.join(REPO, "experiments", "renders")
    if a.stage == "checkpoints":
        rows = []
        for r in build_queue(os.path.join(REPO, "configs", "poisoning")):
            d = os.path.join(runs, r.run_id)
            it = json.load(open(os.path.join(d, "summary.json")))["iterations"]
            p = os.path.join(d, "%06d.tar" % it)
            rows.append({"run_id": r.run_id, "checkpoint": os.path.relpath(p, REPO),
                         "sha256": T.sha256_file(p), "bytes": os.path.getsize(p)})
        T.write_checkpoint_csv(os.path.join(REPO, "experiments", "results", "phase6_final_checkpoints.csv"), rows)
        print("wrote %d rows" % len(rows))
    elif a.stage == "runs":
        _zip_groups(runs, "phase6_runs")
    elif a.stage == "renders":
        ids = [r.run_id for r in build_queue(os.path.join(REPO, "configs", "poisoning"))]
        missing = [i for i in ids if not os.path.isfile(os.path.join(renders, i, "render_meta.json"))]
        stray = [n for n in os.listdir(renders) if n not in ids]
        if missing or stray:
            sys.exit("renders incomplete: missing %s, stray dirs %s" % (missing, stray))
        _zip_groups(renders, "phase6_renders")
    else:
        # .sweep.lock is a PID file the (enabled) nerf-sweep.service rewrites on every autostart; it is not data
        files = [f for f in T.list_files(runs) + T.list_files(renders) if os.path.basename(f) != ".sweep.lock"]
        os.makedirs(OUT, exist_ok=True)
        T.write_manifest(os.path.join(OUT, "TRANSFER_MANIFEST.sha256"), REPO, files)
        print("manifest: %d files" % len(files))


if __name__ == "__main__":
    sys.exit(main())
