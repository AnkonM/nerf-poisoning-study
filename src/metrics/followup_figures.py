"""Figures for the Phase 8 follow-up diagnostics (D-044). Same palette/sidecar conventions as figures.py."""

from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import aggregate as A
from metrics.figures import COND_COLOR, SEED_MARKER


def paired_figure(diffs: Sequence[Dict], a: str, b: str, seeds: Sequence[int], title: str):
    """One panel per metric: per-seed (b - a) with mean, zero line marked."""
    n = len(diffs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, d in zip(axes, diffs):
        ys = [d["diff_seed%d" % s] for s in seeds]
        for s, y in zip(seeds, ys):
            ax.scatter(s, y, marker=SEED_MARKER[s], s=55, color=COND_COLOR[b], zorder=3)
        ax.hlines(d["mean_diff"], seeds[0] - 0.35, seeds[-1] + 0.35, color="black", lw=1.6, zorder=2)
        ax.axhline(0, color="grey", lw=1, ls="--", zorder=1)
        ax.set_xticks(list(seeds)); ax.set_xlim(seeds[0] - 0.6, seeds[-1] + 0.6)
        ax.set_xlabel("seed"); ax.grid(alpha=0.3, axis="y")
        ax.set_title("%s\n%d of %d seeds positive" % (d["metric"], d["n_positive"], len(seeds)), fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.text(0.5, 0.005, "Points: per-seed difference (%s minus %s) on the same seed, same poisoned views; black line: mean; "
             "dashed: zero. Units are those of each metric." % (b, a), ha="center", fontsize=8)
    return fig


def spread_figure(pooled: Dict[str, Dict], per_run: Dict[str, List[Dict]], conditions: Sequence[str], title: str):
    """Left: pooled per-view target PSNR (IQR bar, min-max line, median dot). Right: per-run IQR (3 seeds)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, c in enumerate(conditions):
        p = pooled[c]
        ax1.vlines(i, p["min"], p["max"], color=COND_COLOR[c], lw=1.2)
        ax1.vlines(i, p["q1"], p["q3"], color=COND_COLOR[c], lw=7, alpha=0.8)
        ax1.scatter([i], [p["median"]], color="white", edgecolor="black", s=28, zorder=5)
        for r in per_run[c]:
            ax2.scatter(i + (r["seed"] - 2) * 0.12, r["iqr"], marker=SEED_MARKER[r["seed"]], color=COND_COLOR[c], s=45)
        ax2.hlines(pooled[c]["iqr"], i - 0.3, i + 0.3, color="black", lw=1.4)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(conditions))); ax.set_xticklabels(conditions); ax.grid(alpha=0.3, axis="y")
    ax1.set_ylabel("per-view target-region PSNR (dB)")
    ax1.set_title("pooled over 3 seeds x 75 views: thin = min-max, thick = IQR, circle = median", fontsize=9)
    ax2.set_ylabel("IQR of per-view target PSNR (dB)")
    ax2.set_title("per run (o seed 1, square seed 2, triangle seed 3); black = pooled IQR", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def depth_figure(per_view: Dict[str, Dict[str, np.ndarray]], bins_deg: Sequence[float], title: str):
    """Per condition (seed 1): mean opacity per view; |depth - control depth| per view vs nearest poisoned angle."""
    conds = list(per_view)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, c in enumerate(conds):
        o = 1.0 - per_view[c]["mean_acc"]        # shown as 1 - opacity: every value is 1 to ~1e-7, so the raw axis is unreadable
        ax0.scatter(i + np.random.default_rng(0).uniform(-0.15, 0.15, o.size), o, s=8, alpha=0.55, color=COND_COLOR[c])
        q1, med, q3 = np.percentile(o, [25, 50, 75])
        ax0.vlines(i, q1, q3, color="black", lw=3); ax0.scatter([i], [med], color="white", edgecolor="black", s=26, zorder=5)
    ax0.set_xticks(range(len(conds))); ax0.set_xticklabels(conds); ax0.grid(alpha=0.3, axis="y")
    ax0.set_ylabel("1 - mean accumulated opacity over M_dil (per eval view)")
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    ax0.set_title("1 - opacity (fine network); bar = IQR, circle = median", fontsize=9)
    for c in conds:
        d = per_view[c]
        if "depth_abs_diff" in d and d.get("nearest_deg") is not None:
            ax1.scatter(d["nearest_deg"], d["depth_abs_diff"], s=14, alpha=0.7, color=COND_COLOR[c], label=c)
    for b in bins_deg:
        ax1.axvline(b, color="grey", ls=":", lw=1)
    ax1.set_xlabel("angular distance to nearest poisoned training view (deg); dotted = post-hoc E2 thresholds")
    ax1.set_ylabel("mean |depth - control depth| over M_dil (scene units)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax1.set_title("depth difference from control (seed 1), per eval view", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.text(0.5, 0.005, "Opacity cannot separate the mug from the surface behind it; no ground-truth depth exists for the plate "
             "scene, so depth is compared with the control model only. One seed per condition.", ha="center", fontsize=8)
    return fig
