"""Phase 9 decision evidence (DECISION_LOG.md D-045). Pure functions; no I/O.

Descriptive statistics only. Aggregates reuse `metrics.followup` (`run_region_aggregates`: keys
`mean_of_psnr`, `pooled_mse_view_mean_psnr`)."""

from typing import Dict, Sequence

import numpy as np

AGG_KEYS = ("mean_of_psnr", "pooled_mse_view_mean_psnr")


def average_ranks(x: Sequence[float]) -> np.ndarray:
    """1-based ranks, ties get the average rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _residual(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residuals of an OLS fit of y on [1, z]."""
    A = np.column_stack([np.ones(len(z)), z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def partial_spearman(x: Sequence[float], y: Sequence[float], control: Sequence[float]) -> float:
    """Spearman correlation of x and y controlling for `control`: rank-transform all three (average ranks),
    residualise the two rank vectors on the control rank by OLS, Pearson of the residuals."""
    rx, ry, rz = average_ranks(x), average_ranks(y), average_ranks(control)
    ex, ey = _residual(rx, rz), _residual(ry, rz)
    if np.allclose(ex, 0) or np.allclose(ey, 0):
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def resolvability_ratio(a_vals: Sequence[float], b_vals: Sequence[float]) -> float:
    """|mean(a) - mean(b)| / sample sd (ddof=1) of b's per-seed values (E4: a = C7, b = C3)."""
    b = np.asarray(b_vals, dtype=np.float64)
    return float(abs(np.mean(a_vals) - b.mean()) / b.std(ddof=1))


def step_ratio(hi_vals: Sequence[float], lo_vals: Sequence[float]) -> float:
    """|mean(lo) - mean(hi)| / mean of the two per-condition sample sds (ddof=1) (E5)."""
    hi, lo = np.asarray(hi_vals, dtype=np.float64), np.asarray(lo_vals, dtype=np.float64)
    return float(abs(lo.mean() - hi.mean()) / (0.5 * (hi.std(ddof=1) + lo.std(ddof=1))))


def e1_stats(near_deg: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(near_deg, dtype=np.float64)
    return {"mean_deg": float(a.mean()), "median_deg": float(np.median(a)), "max_deg": float(a.max())}


def hours_per_run(iterations: int, its_per_s: float, val_events: int, val_seconds: float,
                  final_eval_seconds: float) -> float:
    return iterations / its_per_s / 3600.0 + (val_events * val_seconds + final_eval_seconds) / 3600.0
