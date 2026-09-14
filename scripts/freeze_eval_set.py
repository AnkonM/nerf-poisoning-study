#!/usr/bin/env python3
"""Freeze (or verify) the held-out evaluation set. Phase 4 Step 6.

METHODOLOGY.md §4 / README.md ground rules: the held-out views are frozen
before the first poisoning run and never touched again. This writes a SHA-256
manifest plus one aggregate checksum, and makes the files read-only.

WHAT IS FROZEN — deliberately wider than ROADMAP.md Phase 4's literal wording
("freeze data/blender_scenes/eval_holdout/"):

  data/blender_scenes/eval_holdout/      held-out rendered views
  data/masks/eval_holdout/               needed for METHODOLOGY.md §6 MASKED metrics
  data/background_plates/eval_holdout/   needed for §6's secondary plate-distance metric
  data/blender_scenes/transforms_test.json   the poses that DEFINE the held-out set

Freezing only the images would leave the masks, plates and poses that every
held-out metric is computed against mutable — which would defeat the point of
the rule while appearing to satisfy it.

Usage:
    python scripts/freeze_eval_set.py --freeze
    python scripts/freeze_eval_set.py --verify [--root OTHER_DATA_ROOT]
"""

import argparse
import hashlib
import os
import stat
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(REPO_ROOT, "data", "blender_scenes",
                        "eval_holdout_SHA256SUMS.txt")

FROZEN_DIRS = [
    "blender_scenes/eval_holdout",
    "masks/eval_holdout",
    "background_plates/eval_holdout",
]
FROZEN_FILES = ["blender_scenes/transforms_test.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(data_root):
    """(relative_path, sha256) for every frozen file, in sorted order."""
    entries = []
    for d in FROZEN_DIRS:
        base = os.path.join(data_root, d)
        if not os.path.isdir(base):
            raise SystemExit("missing frozen directory: %s" % base)
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if os.path.isfile(p):
                entries.append(("%s/%s" % (d, name), sha256(p)))
    for f in FROZEN_FILES:
        p = os.path.join(data_root, f)
        if not os.path.isfile(p):
            raise SystemExit("missing frozen file: %s" % p)
        entries.append((f, sha256(p)))
    return sorted(entries)


def aggregate(entries):
    """One checksum over the whole manifest — the single number to quote."""
    h = hashlib.sha256()
    for rel, digest in entries:
        h.update(("%s  %s\n" % (digest, rel)).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "data"),
                    help="data root to check (for verifying a regenerated copy)")
    args = ap.parse_args()

    entries = collect(args.root)
    agg = aggregate(entries)

    if args.freeze:
        if os.path.exists(MANIFEST):
            raise SystemExit("manifest already exists — the set is already frozen.\n"
                             "Refusing to overwrite: %s" % MANIFEST)
        with open(MANIFEST, "w") as f:
            f.write("# Frozen held-out evaluation set (Phase 4 Step 6).\n")
            f.write("# METHODOLOGY.md §4 — never regenerate, never modify.\n")
            f.write("# aggregate-sha256: %s\n" % agg)
            f.write("# files: %d\n" % len(entries))
            for rel, digest in entries:
                f.write("%s  %s\n" % (digest, rel))

        # read-only: owner loses write, and so does everyone else
        n = 0
        for d in FROZEN_DIRS:
            base = os.path.join(args.root, d)
            for name in os.listdir(base):
                p = os.path.join(base, name)
                if os.path.isfile(p):
                    os.chmod(p, os.stat(p).st_mode & ~stat.S_IWUSR
                             & ~stat.S_IWGRP & ~stat.S_IWOTH)
                    n += 1
            os.chmod(base, os.stat(base).st_mode & ~stat.S_IWUSR
                     & ~stat.S_IWGRP & ~stat.S_IWOTH)
        for f_ in FROZEN_FILES:
            p = os.path.join(args.root, f_)
            os.chmod(p, os.stat(p).st_mode & ~stat.S_IWUSR
                     & ~stat.S_IWGRP & ~stat.S_IWOTH)
            n += 1
        print("froze %d files across %d directories" % (n, len(FROZEN_DIRS)))
        print("manifest: %s" % os.path.relpath(MANIFEST, REPO_ROOT))
        print("aggregate-sha256: %s" % agg)
        return 0

    # ---- verify ----------------------------------------------------------
    if not os.path.exists(MANIFEST):
        raise SystemExit("no manifest at %s — nothing has been frozen yet" % MANIFEST)
    recorded = {}
    recorded_agg = None
    for line in open(MANIFEST):
        if line.startswith("# aggregate-sha256:"):
            recorded_agg = line.split(":", 1)[1].strip()
        elif not line.startswith("#") and line.strip():
            digest, rel = line.split("  ", 1)
            recorded[rel.strip()] = digest

    current = dict(entries)
    missing = sorted(set(recorded) - set(current))
    extra = sorted(set(current) - set(recorded))
    changed = sorted(r for r in set(recorded) & set(current)
                     if recorded[r] != current[r])

    print("manifest files : %d" % len(recorded))
    print("checked under  : %s" % args.root)
    print("missing        : %d" % len(missing))
    print("unexpected     : %d" % len(extra))
    print("changed        : %d" % len(changed))
    for rel in (missing + extra + changed)[:10]:
        print("   ! %s" % rel)
    print("recorded aggregate : %s" % recorded_agg)
    print("recomputed aggregate: %s" % agg)
    ok = (not missing and not extra and not changed and agg == recorded_agg)
    print("VERIFY: %s" % ("PASS — byte-for-byte identical" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
