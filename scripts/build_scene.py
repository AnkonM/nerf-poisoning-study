#!/usr/bin/env python3
"""Build the main-study Blender scene from a scene config.

Thin CLI entry point (docs/PROJECT_STRUCTURE.md: "scripts/ — thin CLI entry
points, no logic lives here"). All scene-construction logic lives in
src/data_pipeline/blender_build_scene.py, which runs inside Blender.

Why this launcher exists at all: Blender's bundled Python has no `yaml` (only
json — verified, DECISION_LOG.md D-021), so it cannot read configs/ directly.
Rather than installing pyyaml into the Blender install (untracked per-machine
state, which would break "reproducible from a config + a commit hash"), this
script resolves the YAML on the WSL2 side with the project's own loader and
hands Blender a derived, ephemeral JSON. The YAML stays the single source of
truth.

Usage:
    python scripts/build_scene.py configs/scenes/final_scene.yaml
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

# Verified working invocation, DECISION_LOG.md D-021. Blender is a native
# Windows install (D-008) and is NOT on the inherited Windows PATH from WSL2,
# so the explicit path is required.
DEFAULT_BLENDER = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"

BUILDER = os.path.join(REPO_ROOT, "src", "data_pipeline", "blender_build_scene.py")


def to_windows_path(path: str) -> str:
    """Convert a WSL2 path to its \\\\wsl.localhost\\... UNC form for Blender."""
    return subprocess.run(["wslpath", "-w", os.path.abspath(path)],
                          check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a scene config YAML")
    parser.add_argument("--blender", default=DEFAULT_BLENDER,
                        help="path to blender.exe")
    parser.add_argument("--output", default=None,
                        help="output .blend path (default: data/raw/<scene name>.blend)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if "scene" not in cfg:
        parser.error("%s has no `scene:` section — is this a scene config?"
                     % args.config)

    out_blend = args.output or os.path.join(
        REPO_ROOT, "data", "raw", "%s.blend" % cfg["scene"]["name"])
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)

    if not os.path.exists(args.blender):
        print("ERROR: Blender not found at %s" % args.blender, file=sys.stderr)
        return 1

    # Derived, ephemeral — configs/ remains the source of truth.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        spec_path = f.name

    try:
        cmd = [
            args.blender, "--background", "--factory-startup",
            "--python", to_windows_path(BUILDER),
            "--", to_windows_path(spec_path), to_windows_path(out_blend),
        ]
        print("$ %s" % " ".join('"%s"' % c if " " in c else c for c in cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        os.unlink(spec_path)

    for line in proc.stdout.splitlines():
        if line.startswith("BUILD"):
            print(line)
    if "BUILD RESULT: PASS" not in proc.stdout:
        print("--- blender stdout (tail) ---", file=sys.stderr)
        print("\n".join(proc.stdout.splitlines()[-40:]), file=sys.stderr)
        print("--- blender stderr (tail) ---", file=sys.stderr)
        print("\n".join(proc.stderr.splitlines()[-20:]), file=sys.stderr)
        return 1

    print("Scene written to %s" % out_blend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
