#!/usr/bin/env python3
"""Batch-render the Phase 4 dataset (originals, masks, background plates).

Thin CLI entry point (docs/PROJECT_STRUCTURE.md); logic lives in
src/data_pipeline/blender_render_views.py, which runs inside Blender.

Reads camera poses from data/blender_scenes/cameras.json (produced by
scripts/build_camera_rig.py) and writes, per D-021's Option A, DIRECTLY into
the WSL2 repo over the UNC path — no separate copy/sync step.

Streams Blender's per-view progress through live, so a long render is
distinguishable from a stall (D-016).

Usage:
    python scripts/render_scene.py configs/scenes/final_scene.yaml
    python scripts/render_scene.py configs/scenes/final_scene.yaml --limit 2
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.config import load_config  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BLENDER = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
RENDERER = os.path.join(REPO_ROOT, "src", "data_pipeline", "blender_render_views.py")


def to_windows_path(path: str) -> str:
    return subprocess.run(["wslpath", "-w", os.path.abspath(path)],
                          check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--blend", default=None)
    parser.add_argument("--cameras", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="render only the first N poses (pilot runs)")
    parser.add_argument("--splits", default=None,
                        help="comma-separated subset, e.g. train,eval_holdout")
    args = parser.parse_args()

    cfg = load_config(args.config)
    blend = args.blend or os.path.join(REPO_ROOT, "data", "raw",
                                       "%s.blend" % cfg["scene"]["name"])
    cameras = args.cameras or os.path.join(REPO_ROOT, "data", "blender_scenes",
                                           "cameras.json")
    for path, label in ((args.blender, "Blender"), (blend, "scene .blend"),
                        (cameras, "cameras.json")):
        if not os.path.exists(path):
            print("ERROR: %s not found at %s" % (label, path), file=sys.stderr)
            return 1

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        spec_path = f.name

    try:
        cmd = [args.blender, "--background", "--factory-startup",
               to_windows_path(blend),
               "--python", to_windows_path(RENDERER), "--",
               to_windows_path(spec_path), to_windows_path(cameras),
               to_windows_path(os.path.join(REPO_ROOT, "data"))]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.splits:
            cmd += ["--splits", args.splits]

        # Stream output rather than capturing: a multi-minute render must show
        # progress live (D-016), not dump a log at the end.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        ok = False
        for line in proc.stdout:
            if line.startswith("RENDER"):
                print(line.rstrip())
                sys.stdout.flush()
                if "DONE" in line:
                    ok = True
        proc.wait()
    finally:
        os.unlink(spec_path)

    return 0 if ok and proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
