#!/usr/bin/env python
"""Export TensorBoard scalar curves (train/loss, train/psnr,
val/psnr_subset) for a training run to a durable CSV + PNG pair, and
report a simple divergence check.

This is an audit-trail utility, not a paper-metrics script — it does not
belong in src/metrics/ (that's for docs/METHODOLOGY.md §6 metrics). Kept
here since the run's TensorBoard event files (experiments/runs/<run_id>/tb/)
are themselves a durable audit artifact this script reads.

Generalised in Phase 6 Step 1 (D-032) beyond its original Phase 3 PoC-only
output path, and to handle a RESUMED run correctly. Every `scripts/train.py`
invocation opens a new SummaryWriter, so a resumed run's tb/ directory holds
MULTIPLE event files with overlapping step ranges — e.g. an original run that
died mid-training, plus one or more --resume attempts that re-log some of the
same steps. `EventAccumulator` pointed at a directory does NOT merge these:
it concatenates every file's points, so an overlapping step appears more than
once with different values (verified: 2-3x duplicate entries at every step
from 75000 onward, across a real 3-file resumed run). This module instead
merges per scalar tag, per step, keeping the value from the file with the
LATEST embedded wall-clock time (files are named
`events.out.tfevents.<wall_time>.<host>.<pid>.<n>`, which sorts
chronologically by filename) — i.e. the most recent write wins, which is
exactly what actually happened on disk: an aborted resume's steps are
superseded by whichever resume actually reached them last and produced the
final checkpoint.

Usage:
    python scripts/export_tb_curves.py <run_id>
    python scripts/export_tb_curves.py <run_id> --results-subdir phase6

Reads experiments/runs/<run_id>/tb/ (one or more event files), writes:
    experiments/results/<subdir>/<run_id>/curves.csv
    experiments/results/<subdir>/<run_id>/train_loss.png
    experiments/results/<subdir>/<run_id>/val_psnr_subset.png
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ("train/loss", "train/psnr", "val/psnr_subset")


def _merged_series(tb_dir: str, tag: str):
    """step -> value across every event file in tb_dir, latest file wins.

    Files are read in filename order, which is chronological (the embedded
    `wall_time` is the first numeric field), so a later resume's write for a
    given step always overwrites an earlier attempt's write for that step.
    """
    files = sorted(f for f in os.listdir(tb_dir) if f.startswith("events.out.tfevents"))
    merged = {}
    for f in files:
        ea = EventAccumulator(os.path.join(tb_dir, f))
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars(tag):
            merged[e.step] = e.value       # later file in the loop wins
    steps = sorted(merged)
    return steps, [merged[s] for s in steps], files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument(
        "--results-subdir", default=None,
        help="Subdirectory under experiments/results/. Defaults to 'phase3_poc' "
        "for run ids starting with that prefix (unchanged historical behaviour), "
        "else 'phase6'.",
    )
    args = parser.parse_args()
    run_id = args.run_id
    subdir = args.results_subdir or ("phase3_poc" if run_id.startswith("phase3_poc") else "phase6")

    tb_dir = os.path.join("experiments", "runs", run_id, "tb")
    out_dir = os.path.join("experiments", "results", subdir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    loss_steps, loss_vals, files = _merged_series(tb_dir, "train/loss")
    _, tpsnr_vals, _ = _merged_series(tb_dir, "train/psnr")
    vpsnr_steps, vpsnr_vals, _ = _merged_series(tb_dir, "val/psnr_subset")
    if len(files) > 1:
        print(f"[{run_id}] merged {len(files)} event files (resumed run) — "
              f"latest write wins per step: {files}")

    # --- CSV: one row per train/loss+psnr step, val PSNR sparsely filled in ---
    val_by_step = dict(zip(vpsnr_steps, vpsnr_vals))
    csv_path = os.path.join(out_dir, "curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "train_psnr", "val_psnr_subset"])
        for step, loss, psnr in zip(loss_steps, loss_vals, tpsnr_vals):
            writer.writerow([step, loss, psnr, val_by_step.get(step, "")])

    # --- Divergence check: any NaN/Inf, or a sustained loss increase ---
    # NOTE (Phase 3 closeout): an earlier version compared the tail-mean loss
    # to the single lowest-ever loss sample, which false-positived because
    # per-step batches are drawn from one randomly chosen training image and
    # the single global min is an outlier lucky batch, not a meaningful
    # baseline. Fixed to compare first-half vs. second-half mean loss
    # instead, which is robust to single-batch noise.
    loss_arr = np.array(loss_vals, dtype=np.float64)
    nan_inf = bool(np.any(~np.isfinite(loss_arr)))
    n = len(loss_arr)
    first_half_mean = loss_arr[: n // 2].mean() if n else float("nan")
    second_half_mean = loss_arr[n // 2:].mean() if n else float("nan")
    max_val = loss_arr.max() if n else float("nan")
    max_step = loss_steps[int(loss_arr.argmax())] if n else None
    diverged = bool(n and (second_half_mean > first_half_mean or max_step > loss_steps[-1] * 0.5))
    print(f"[{run_id}] loss samples: {n}, first_half_mean={first_half_mean:.5f}, "
          f"second_half_mean={second_half_mean:.5f}, final={loss_arr[-1]:.5f}, "
          f"max={max_val:.5f} at step={max_step} (of {loss_steps[-1]})")
    print(f"[{run_id}] NaN/Inf present: {nan_inf}")
    print(f"[{run_id}] divergence flag (loss not trending down, or late-training loss spike): {diverged}")

    # --- Plots ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_steps, loss_vals, linewidth=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("train/loss (per-batch MSE)")
    ax.set_title(f"{run_id} — training loss vs. iteration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    loss_png = os.path.join(out_dir, "train_loss.png")
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vpsnr_steps, vpsnr_vals, marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("val-subset PSNR (dB)")
    ax.set_title(f"{run_id} — val-subset PSNR vs. iteration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    val_png = os.path.join(out_dir, "val_psnr_subset.png")
    fig.savefig(val_png, dpi=150)
    plt.close(fig)

    print(f"[{run_id}] wrote {csv_path}, {loss_png}, {val_png}")


if __name__ == "__main__":
    main()
