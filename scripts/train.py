#!/usr/bin/env python
"""Train one NeRF from a resolved config. Thin CLI entry point — all logic
lives in src/nerf/training.py, per docs/PROJECT_STRUCTURE.md.

Usage:
    python scripts/train.py <config_path> [--run-id RUN_ID]
"""

import argparse
import json
import os
import subprocess
import sys

from nerf.training import train_from_config
from utils.config import load_config


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a config YAML (e.g. configs/scenes/lego_sanity.yaml)")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier; defaults to the config file's basename. "
        "Checkpoints/tensorboard logs go to experiments/runs/<run-id>/.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = args.run_id or os.path.splitext(os.path.basename(args.config))[0]
    run_dir = os.path.join("experiments", "runs", run_id)

    print(f"Config: {args.config}")
    print(f"Git commit: {_git_commit_hash()}")
    print(f"Run dir: {run_dir}")

    summary = train_from_config(cfg, run_dir=run_dir, run_id=run_id)
    summary["config_path"] = args.config
    summary["git_commit"] = _git_commit_hash()

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
