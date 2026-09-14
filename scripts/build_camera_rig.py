#!/usr/bin/env python3
"""Generate the Phase 4 camera rig and pose files from a scene config.

Thin CLI entry point (docs/PROJECT_STRUCTURE.md); all logic lives in
src/data_pipeline/blender_build_rig.py, which runs inside Blender so that
`transform_matrix` is Blender's own `camera.matrix_world` and therefore matches
the loader's convention by construction (D-021, D-022).

Writes data/blender_scenes/{cameras.json, transforms_train.json,
transforms_val.json, transforms_test.json}.

Usage:
    python scripts/build_camera_rig.py configs/scenes/final_scene.yaml
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
BUILDER = os.path.join(REPO_ROOT, "src", "data_pipeline", "blender_build_rig.py")


def to_windows_path(path: str) -> str:
    return subprocess.run(["wslpath", "-w", os.path.abspath(path)],
                          check=True, capture_output=True, text=True).stdout.strip()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              check=True, capture_output=True,
                              text=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a scene config YAML")
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--blend", default=None,
                        help="scene .blend (default: data/raw/<scene name>.blend)")
    parser.add_argument("--out", default=None,
                        help="output dir (default: data/blender_scenes)")
    parser.add_argument("--measure-depth", action="store_true",
                        help="probe the scene's real depth range to set near/far")
    args = parser.parse_args()

    cfg = load_config(args.config)
    blend = args.blend or os.path.join(REPO_ROOT, "data", "raw",
                                       "%s.blend" % cfg["scene"]["name"])
    out_dir = args.out or os.path.join(REPO_ROOT, "data", "blender_scenes")

    for path, label in ((args.blender, "Blender"), (blend, "scene .blend")):
        if not os.path.exists(path):
            print("ERROR: %s not found at %s" % (label, path), file=sys.stderr)
            return 1
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        spec_path = f.name

    try:
        cmd = [args.blender, "--background", "--factory-startup",
               to_windows_path(blend),
               "--python", to_windows_path(BUILDER), "--",
               to_windows_path(spec_path), to_windows_path(out_dir),
               git_commit()]
        if args.measure_depth:
            cmd.append("--measure-depth")
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        os.unlink(spec_path)

    for line in proc.stdout.splitlines():
        if line.startswith("RIG"):
            print(line)
    if "RIG RESULT: PASS" not in proc.stdout:
        print("\n".join(proc.stdout.splitlines()[-40:]), file=sys.stderr)
        print("\n".join(proc.stderr.splitlines()[-20:]), file=sys.stderr)
        return 1

    print("Rig written to %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
