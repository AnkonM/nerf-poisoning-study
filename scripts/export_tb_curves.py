#!/usr/bin/env python
"""Export TensorBoard scalar curves (train/loss, train/psnr,
val/psnr_subset) for a Phase 3 PoC run to a durable CSV + PNG pair, and
report a simple divergence check.

This is an audit-trail utility for docs/ROADMAP.md Phase 3's training
runs, not a paper-metrics script — it does not belong in src/metrics/
(that's for docs/METHODOLOGY.md §6 metrics). Kept here rather than in a
scratch location since the run's TensorBoard event files
(experiments/runs/<run_id>/tb/) are itself a durable audit artifact this
script reads, and it may be reused for later phases' runs too.

Usage:
    python scripts/export_tb_curves.py <run_id>

Reads experiments/runs/<run_id>/tb/, writes:
    experiments/results/phase3_poc/<run_id>/curves.csv
    experiments/results/phase3_poc/<run_id>/train_loss.png
    experiments/results/phase3_poc/<run_id>/val_psnr_subset.png
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _scalar_series(ea: EventAccumulator, tag: str):
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def main() -> None:
    run_id = sys.argv[1]
    tb_dir = os.path.join("experiments", "runs", run_id, "tb")
    out_dir = os.path.join("experiments", "results", "phase3_poc", run_id)
    os.makedirs(out_dir, exist_ok=True)

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    loss_steps, loss_vals = _scalar_series(ea, "train/loss")
    tpsnr_steps, tpsnr_vals = _scalar_series(ea, "train/psnr")
    vpsnr_steps, vpsnr_vals = _scalar_series(ea, "val/psnr_subset")

    # --- CSV: one row per train/loss+psnr step, val PSNR sparsely filled in ---
    val_by_step = dict(zip(vpsnr_steps, vpsnr_vals))
    csv_path = os.path.join(out_dir, "curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "train_psnr", "val_psnr_subset"])
        for step, loss, psnr in zip(loss_steps, loss_vals, tpsnr_vals):
            writer.writerow([step, loss, psnr, val_by_step.get(step, "")])

    # --- Divergence check: any NaN/Inf, or a sustained loss increase ---
    # NOTE: an earlier version of this check compared the tail-mean loss to
    # the single lowest-ever loss sample, which false-positived on both
    # poisoned conditions -- with per-step batches drawn from one randomly
    # chosen training image (some images/regions are much easier to fit
    # than others, e.g. the flat erased region), the single global min is
    # an outlier lucky batch, not a meaningful convergence baseline. Fixed
    # to compare first-half vs. second-half mean loss instead, which is
    # robust to single-batch noise and directly answers "is loss trending
    # down overall".
    loss_arr = np.array(loss_vals, dtype=np.float64)
    nan_inf = bool(np.any(~np.isfinite(loss_arr)))
    n = len(loss_arr)
    first_half_mean = loss_arr[: n // 2].mean() if n else float("nan")
    second_half_mean = loss_arr[n // 2 :].mean() if n else float("nan")
    max_val = loss_arr.max() if n else float("nan")
    max_step = loss_steps[int(loss_arr.argmax())] if n else None
    # Divergence: loss trending up overall (second half worse than first),
    # or the worst single loss value occurring late in training rather than
    # during the expected early/precrop warmup.
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
