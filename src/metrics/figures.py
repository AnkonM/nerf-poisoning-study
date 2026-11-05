"""Phase 8 figures (D-043). matplotlib only; Okabe-Ito colour-blind-safe palette; every figure
is saved as PNG + PDF + a sidecar JSON tying it to the per_run.csv hash, run_ids, columns plotted
and the git commit."""

import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import aggregate as A

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
COND_COLOR = {"control": OI["black"], "C1": OI["sky"], "C2": OI["blue"], "C3": OI["orange"],
              "C4": OI["vermillion"], "C5": OI["purple"], "C6": OI["green"], "C7": "#8c564b"}
SEED_MARKER = {1: "o", 2: "s", 3: "^"}
LABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS (lower = closer to GT)",
         "plate_psnr": "PSNR vs background plate (dB)", "plate_mse": "MSE vs background plate"}
REGION_NAME = {"tgt": "target region (M_dil)", "col": "collateral region", "whole": "whole frame",
               "rawtgt": "target region, raw mask (sensitivity)"}
SPREAD_SHORT = {True: "init-only\nvariance", False: "view-selection\n+ init variance"}
X_LABEL = "Poisoning budget (% of V_target)"
FOOT = ("Points: individual seeds (o seed 1, square seed 2, triangle seed 3); line: mean; bars: sample std (ddof=1, n=3). "
        "control: initialisation-only variance (one dataset). C1-C5: view-selection + initialisation variance.")


def save_fig(fig, name: str, outdir: str, meta: Dict) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    files = []
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, "%s.%s" % (name, ext))
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        files.append(p)
    side = dict(meta, figure=name, files=[os.path.basename(f) for f in files])
    sp = os.path.join(outdir, "%s.json" % name)
    with open(sp, "w") as f:
        json.dump(side, f, indent=2, sort_keys=True)
    plt.close(fig)
    return files + [sp]


def _metric_label(col: str) -> str:
    if col.startswith("plate_"):
        return LABEL[col]
    region, kind = col.split("_")
    return "%s\n%s" % (REGION_NAME[region], LABEL[kind])


def budget_figure(rows, prefix: str, title: str, raw_prefix: Optional[str] = None):
    """3 panels (PSNR, SSIM, LPIPS) of `prefix` vs budget, per-seed points over mean +/- std."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, kind in zip(axes, ("psnr", "ssim", "lpips")):
        col = "%s_%s" % (prefix, kind)
        _budget_axis(ax, rows, col)
        if raw_prefix:
            _budget_axis(ax, rows, "%s_%s" % (raw_prefix, kind), thin=True)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(_metric_label(col))
        ax.set_xticks(list(A.BUDGET_PERCENT.values()))
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=OI["blue"], marker="o", label="D-042 definition")]
    if raw_prefix:
        handles.append(plt.Line2D([], [], color="grey", lw=0.8, ls="--", marker=".",
                                  label="sensitivity: raw (undilated) mask"))
    axes[0].legend(handles=handles, loc="best", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(wspace=0.42)
    fig.text(0.5, -0.03, FOOT, ha="center", fontsize=7.5, wrap=True)
    return fig


def _budget_axis(ax, rows, col, thin=False):
    xs, means, stds = [], [], []
    color = "grey" if thin else OI["blue"]
    for cond in A.BUDGET_CONDITIONS:
        v = np.array(A.values(rows, cond, col))
        x = A.BUDGET_PERCENT[cond]
        xs.append(x); means.append(v.mean()); stds.append(v.std(ddof=1))
        for seed, y in zip(sorted(r["seed"] for r in rows if r["condition"] == cond), v):
            ax.scatter(x + (seed - 2) * (0.5 if not thin else 0.0) + (0.25 if thin else 0), y,
                       marker=SEED_MARKER[seed], s=14 if thin else 38, color=color,
                       alpha=0.7 if thin else 0.9, zorder=3, edgecolor="none")
    ax.errorbar(xs, means, yerr=stds, color=color, lw=0.8 if thin else 1.6, ls="--" if thin else "-",
                capsize=2 if thin else 3, zorder=2, alpha=0.8)


def plate_figure(rows, gt_vs_plate_mean: float, title: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _budget_axis(ax, rows, "plate_psnr")
    ax.axhline(gt_vs_plate_mean, color=OI["vermillion"], ls=":", lw=1.5,
               label="clean GT vs plate (mean over the 75 eval views, M_dil): %.2f dB" % gt_vs_plate_mean)
    ax.set_xlabel(X_LABEL); ax.set_ylabel(LABEL["plate_psnr"])
    ax.set_xticks(list(A.BUDGET_PERCENT.values())); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, -0.06, FOOT, ha="center", fontsize=7, wrap=True)
    return fig


def pair_figure(rows, ca: str, cb: str, cols: Sequence[str], ylims: Dict[str, tuple], title: str):
    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4.6))
    for ax, col in zip(np.atleast_1d(axes), cols):
        for i, cond in enumerate((ca, cb)):
            v = np.array(A.values(rows, cond, col))
            seeds = sorted(r["seed"] for r in rows if r["condition"] == cond)
            for s, y in zip(seeds, v):
                ax.scatter(i + (s - 2) * 0.06, y, marker=SEED_MARKER[s], s=48, color=COND_COLOR[cond], zorder=3)
            ax.hlines(v.mean(), i - 0.22, i + 0.22, color=COND_COLOR[cond], lw=2)
            ax.errorbar([i + 0.3], [v.mean()], yerr=[v.std(ddof=1)], color="grey", capsize=3, lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["%s\n%s" % (c, SPREAD_SHORT[c in A.SEED_INVARIANT]) for c in (ca, cb)], fontsize=7.5)
        ax.set_ylim(*ylims[col]); ax.set_ylabel(_metric_label(col)); ax.grid(alpha=0.3, axis="y")
        ax.set_xlim(-0.6, 1.6)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(wspace=0.42)
    fig.text(0.5, -0.06, "Points: seeds (o seed 1, square 2, triangle 3); coloured line: mean; grey bar (right): "
             "sample std (ddof=1, n=3). The two spreads are different kinds of variance and are not like for like.",
             ha="center", fontsize=7.5)
    return fig


def e1_figure(dist: Dict[str, Dict[str, np.ndarray]]):
    """dist[metric][condition] -> pooled per-view values (75 views x 3 seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (metric, label) in zip(axes, (("tgt_psnr", "Target-region PSNR (dB)"),
                                          ("plate_psnr", "PSNR vs background plate (dB)"))):
        data = [dist[metric][c] for c in A.CONDITIONS]
        parts = ax.violinplot(data, showextrema=False, widths=0.8)
        for pc, c in zip(parts["bodies"], A.CONDITIONS):
            pc.set_facecolor(COND_COLOR[c]); pc.set_alpha(0.25)
        rng = np.random.default_rng(0)
        for i, (c, d) in enumerate(zip(A.CONDITIONS, data), start=1):
            ax.scatter(i + rng.uniform(-0.18, 0.18, d.size), d, s=3, color=COND_COLOR[c], alpha=0.35)
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            ax.vlines(i, q1, q3, color="black", lw=3)
            ax.scatter([i], [med], color="white", edgecolor="black", zorder=5, s=25, label="median" if i == 1 else None)
            ax.scatter([i], [d.mean()], marker="D", color="red", zorder=5, s=18, label="mean" if i == 1 else None)
        ax.set_xticks(range(1, len(A.CONDITIONS) + 1)); ax.set_xticklabels(A.CONDITIONS)
        ax.set_ylabel(label); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.suptitle("EXPLORATORY (post-hoc-motivated, D-043 E1): per-view distributions, 75 views x 3 seeds pooled. "
                 "Black bar = IQR.", fontsize=10)
    return fig


def e2_figure(points: Dict[str, Dict[str, np.ndarray]]):
    """points[cond] -> dict(az, el, nearest, psnr, seed) over (seed x view)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, xl) in zip(axes, (("az", "Eval-view azimuth (deg)"), ("el", "Eval-view elevation (deg)"),
                                    ("nearest", "Angular distance to nearest POISONED training view (deg)"))):
        for cond in ("C3", "C7"):
            p = points[cond]
            for s in (1, 2, 3):
                m = p["seed"] == s
                ax.scatter(p[key][m], p["psnr"][m], s=14, marker=SEED_MARKER[s], color=COND_COLOR[cond],
                           alpha=0.6, label="%s seed %d" % (cond, s))
        ax.set_xlabel(xl, fontsize=8.5); ax.set_ylabel("Target-region PSNR (dB)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("EXPLORATORY (post-hoc-motivated, D-043 E2): C3 (random) vs C7 (strategic), per-view target PSNR; "
                 "angles from the scene origin", fontsize=10)
    return fig


def e3_figure(band_table: Dict[str, np.ndarray], labels: Sequence[str]):
    """band_table[cond] -> array [n_seeds, n_bands] of per-run mean-over-views band MSE."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.8 / len(A.CONDITIONS)
    for i, c in enumerate(A.CONDITIONS):
        t = band_table[c]
        x = np.arange(len(labels)) + (i - len(A.CONDITIONS) / 2 + 0.5) * w
        ax.bar(x, t.mean(0), width=w, color=COND_COLOR[c], alpha=0.55, label=c)
        for s in range(t.shape[0]):
            ax.scatter(x, t[s], marker=SEED_MARKER[s + 1], s=10, color="black", zorder=3)
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(["%s px" % l for l in labels])
    ax.set_xlabel("Distance from the dilated target mask (collateral region only)")
    ax.set_ylabel("Mean squared error, render vs clean GT")
    ax.grid(alpha=0.3, axis="y"); ax.legend(ncol=4, fontsize=8)
    fig.suptitle("EXPLORATORY (post-hoc-motivated, D-043 E3): collateral error by distance band; "
                 "bars = mean over seeds, dots = seeds", fontsize=10)
    return fig


def image_grid(rows_imgs: Sequence[Sequence[np.ndarray]], row_labels: Sequence[str],
               col_labels: Sequence[str], title: str):
    nr, nc = len(rows_imgs), len(col_labels)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 3.0 * nr), squeeze=False)
    for i in range(nr):
        for j in range(nc):
            ax = axes[i][j]
            ax.imshow(np.clip(rows_imgs[i][j], 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig
