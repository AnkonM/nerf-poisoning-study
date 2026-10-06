"""Where a training run's directory lives, and what is already in it.

scripts/train.py used to hard-code `<repo>/experiments/runs/<run_id>`. D-037
adds an optional `--output-root` that changes ONLY the parent directory (so
benchmark and smoke runs can be kept out of the sweep-visible tree). With no
override the result is exactly the old expression.
"""

import os
from typing import List, Optional


def resolve_run_dir(repo_root: str, run_id: str, output_root: Optional[str] = None) -> str:
    if output_root is None:
        return os.path.join(repo_root, "experiments", "runs", run_id)
    return os.path.join(os.path.abspath(os.path.expanduser(output_root)), run_id)


def existing_checkpoints(run_dir: str) -> List[str]:
    """Checkpoint filenames already in `run_dir`, sorted (zero-padded names)."""
    if not os.path.isdir(run_dir):
        return []
    return sorted(f for f in os.listdir(run_dir) if f.endswith(".tar"))
