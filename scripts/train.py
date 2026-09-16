#!/usr/bin/env python
"""Train one NeRF from a resolved config. Thin CLI entry point — all logic
lives in src/nerf/training.py and src/utils/dataset_id.py, per
docs/PROJECT_STRUCTURE.md.

Two modes, distinguished by whether the config declares `dataset.path_template`:

* **Scene configs** (configs/scenes/*.yaml, the Phase 3 PoC configs) carry a
  literal `dataset.path` and train exactly as before. Unchanged behaviour.

* **Phase 6 condition configs** (configs/poisoning/*.yaml) carry
  `dataset.path: null` plus a `path_template`, and REQUIRE `--seed`. The
  training set is resolved per (condition, seed) and then VERIFIED BY CONTENT
  before a single gradient step — see src/utils/dataset_id.py for why a path
  check alone is not enough (D-032).

Usage:
    python scripts/train.py configs/scenes/lego_sanity.yaml
    python scripts/train.py configs/poisoning/budget_20.yaml --seed 1 --skip-final-eval
"""

import argparse
import json
import os
import subprocess
import sys

from nerf.training import train_from_config
from utils.config import load_config
from utils.dataset_id import resolve_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=REPO_ROOT).strip()
    except Exception:
        return "unknown"


def _git_commit_hash() -> str:
    return _git("rev-parse", "HEAD")


def _git_is_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a config YAML")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Training seed. REQUIRED for condition configs; it selects both the "
        "model initialisation and the poisoned dataset, which METHODOLOGY.md §8 "
        "ties to the same seed set.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run identifier; defaults to <condition_id>_seed<N> for condition "
        "configs, else the config file's basename. Checkpoints and tensorboard "
        "logs go to experiments/runs/<run-id>/.",
    )
    parser.add_argument(
        "--skip-final-eval", action="store_true",
        help="Skip the automatic full-test-set eval at the end of training "
        "(runs in-process while training's VRAM allocations are still held — "
        "see docs/DECISION_LOG.md D-016/D-019). The Phase 6 sweep always passes "
        "this and evaluates separately via scripts/evaluate.py.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Permit resuming from checkpoints already in the run directory. "
        "WITHOUT this flag, a non-empty run directory is a hard error. "
        "src/nerf/training.py auto-resumes from the newest checkpoint it finds, "
        "silently — across a 24-run sweep with programmatic run ids that would "
        "quietly continue a previous run instead of starting a new one. Note a "
        "resumed run is NOT equivalent to an uninterrupted one: RNG state is not "
        "checkpointed, so the training-view sequence differs after the resume "
        "point (D-032). Record this in the run's experiments/logs/ entry.",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Permit training from a dirty working tree. Off by default: every "
        "results.csv row must trace to a git commit hash "
        "(docs/PROJECT_STRUCTURE.md), which is meaningless if the tree has "
        "uncommitted changes.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    is_condition = bool(dataset_cfg.get("path_template"))

    if _git_is_dirty() and not args.allow_dirty:
        sys.exit(
            "Refusing to train from a dirty working tree — a recorded commit "
            "hash would not describe the code that actually ran. Commit first, "
            "or pass --allow-dirty for an exploratory run whose numbers will "
            "NOT go into results.csv."
        )

    dataset_info = None
    if is_condition:
        if args.seed is None:
            sys.exit(
                "--seed is required for condition configs: it selects both the "
                "model initialisation and the poisoned dataset (METHODOLOGY.md "
                "§8). Valid seeds for this config: %s" % (cfg.get("seeds"),)
            )
        if args.seed not in cfg.get("seeds", []):
            sys.exit(
                "seed %d is not in this config's declared seeds %s. Seeds are "
                "fixed in the config before training, never chosen afterward "
                "(METHODOLOGY.md §8)." % (args.seed, cfg.get("seeds"))
            )
        cfg["reproducibility"]["seed"] = args.seed
        # Resolve AND verify. Raises loudly rather than training wrong data.
        dataset_info = resolve_dataset(cfg, args.seed, repo_root=REPO_ROOT)
        cfg["dataset"]["path"] = os.path.join(REPO_ROOT, dataset_info["dataset_path"])
        run_id = args.run_id or "%s_seed%d" % (cfg["condition_id"], args.seed)
    else:
        if args.seed is not None:
            cfg["reproducibility"]["seed"] = args.seed
        run_id = args.run_id or os.path.splitext(os.path.basename(args.config))[0]

    run_dir = os.path.join(REPO_ROOT, "experiments", "runs", run_id)

    existing = sorted(f for f in os.listdir(run_dir) if f.endswith(".tar")) \
        if os.path.isdir(run_dir) else []
    if existing and not args.resume:
        sys.exit(
            "Run directory already holds checkpoints: %s\n"
            "  %s\n"
            "src/nerf/training.py would SILENTLY resume from the newest one. "
            "Pass --resume to do that deliberately (and record it in the run's "
            "log entry), or remove the directory to start clean."
            % (run_dir, ", ".join(existing))
        )

    print(f"Config: {args.config}")
    print(f"Git commit: {_git_commit_hash()}")
    print(f"Run dir: {run_dir}")
    print(f"Seed: {cfg['reproducibility']['seed']}")
    if dataset_info:
        print("Dataset VERIFIED: %s (%d/%d views poisoned, matching "
              "round(%s/100 x 100)) sha256=%s"
              % (dataset_info["dataset_id"], dataset_info["n_poisoned_verified"],
                 dataset_info["n_train_views"], cfg["poisoning"]["budget_percent"],
                 dataset_info["train_set_sha256"][:16]))

    summary = train_from_config(
        cfg, run_dir=run_dir, run_id=run_id, skip_final_test_eval=args.skip_final_eval
    )
    summary["config_path"] = args.config
    summary["git_commit"] = _git_commit_hash()
    summary["run_id"] = run_id
    summary["seed"] = cfg["reproducibility"]["seed"]
    summary["iterations"] = cfg["training"]["iterations"]
    summary["batch_size"] = cfg["training"]["batch_size"]
    summary["resumed"] = bool(existing and args.resume)
    if existing and args.resume:
        summary["resumed_from"] = existing[-1]
        summary["resume_caveat"] = (
            "RNG state is not checkpointed, so the training-view sequence after "
            "the resume point differs from an uninterrupted run (D-032)."
        )
    if dataset_info:
        summary.update({"condition_id": cfg["condition_id"], **dataset_info})

    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
