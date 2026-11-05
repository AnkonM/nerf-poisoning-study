"""Phase 8 analysis primitives (D-043). Pure functions over already-computed D-042 metrics;
nothing here recomputes a metric or writes outside its caller's chosen output directory."""

import csv
import hashlib
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

CONDITIONS = ["control", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
BUDGET_CONDITIONS = ["control", "C1", "C2", "C3", "C4", "C5"]      # hard erasure, random (D-043 B)
BUDGET_PERCENT = {"control": 0, "C1": 5, "C2": 10, "C3": 20, "C4": 30, "C5": 50}
SEED_INVARIANT = ("control", "C7")           # one dataset, 3 init seeds (D-032)
PAIRS = {"hard_vs_soft": ("C3", "C6"), "random_vs_strategic": ("C3", "C7")}   # D-043 B

REGIONS = ("whole", "tgt", "col", "rawtgt")
METRIC_COLS = (["%s_%s" % (r, k) for r in REGIONS for k in ("psnr", "ssim", "lpips")]
               + ["plate_mse", "plate_psnr"])
PAIR_COLS = ["tgt_psnr", "tgt_ssim", "tgt_lpips", "col_psnr", "col_ssim", "col_lpips",
             "plate_psnr", "plate_mse"]
BAND_EDGES = (0, 10, 30, 100, None)           # px; (0,10], (10,30], (30,100], >100


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_per_run(path: str) -> List[Dict]:
    """per_run.csv -> list of dicts with metric columns as float and seed as int."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r = dict(r)
            r["seed"] = int(r["seed"])
            for c in METRIC_COLS:
                r[c] = float(r[c])
            rows.append(r)
    return rows


def spread_kind(condition: str) -> str:
    return ("initialisation-only (one dataset, 3 init seeds)" if condition in SEED_INVARIANT
            else "view-selection + initialisation")


def values(rows: Sequence[Dict], condition: str, col: str) -> List[float]:
    vals = sorted(((r["seed"], r[col]) for r in rows if r["condition"] == condition))
    return [v for _, v in vals]


def condition_summary(rows: Sequence[Dict]) -> List[Dict]:
    """Per condition x metric: n, mean, sample std (ddof=1), min, max, per-seed values."""
    out = []
    for cond in CONDITIONS:
        seeds = sorted(r["seed"] for r in rows if r["condition"] == cond)
        if not seeds:
            continue
        for col in METRIC_COLS:
            v = np.array(values(rows, cond, col), dtype=np.float64)
            out.append({
                "condition": cond, "metric": col, "n": int(v.size),
                "mean": float(v.mean()),
                "std_ddof1": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                "min": float(v.min()), "max": float(v.max()),
                "per_seed": ";".join("%d:%r" % (s, x) for s, x in zip(seeds, v.tolist())),
                "spread_kind": spread_kind(cond),
            })
    return out


def ranges_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return max(a) >= min(b) and max(b) >= min(a)


def pair_comparisons(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for name, (ca, cb) in PAIRS.items():
        for col in PAIR_COLS:
            a, b = values(rows, ca, col), values(rows, cb, col)
            out.append({
                "pair": name, "cond_a": ca, "cond_b": cb, "metric": col,
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "mean_b_minus_a": float(np.mean(b) - np.mean(a)),
                "std_a_ddof1": float(np.std(a, ddof=1)), "std_b_ddof1": float(np.std(b, ddof=1)),
                "per_seed_a": ";".join("%r" % x for x in a), "per_seed_b": ";".join("%r" % x for x in b),
                "range_a": "[%r, %r]" % (min(a), max(a)), "range_b": "[%r, %r]" % (min(b), max(b)),
                "ranges_overlap": bool(ranges_overlap(a, b)),
                "spread_kind_a": spread_kind(ca), "spread_kind_b": spread_kind(cb),
            })
    return out


# ---- E1 -----------------------------------------------------------------------
def distribution_stats(x: Sequence[float]) -> Dict:
    x = np.asarray(x, dtype=np.float64)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(med), "q1": float(q1),
            "q3": float(q3), "min": float(x.min()), "max": float(x.max())}


# ---- E2 -----------------------------------------------------------------------
def positions_from_transforms(frames: Sequence[Dict]) -> np.ndarray:
    """Camera positions = translation column of each camera-to-world matrix, [N,3]."""
    return np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames], dtype=np.float64)


def azimuth_elevation_deg(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    az = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
    el = np.degrees(np.arcsin(pos[:, 2] / np.linalg.norm(pos, axis=1)))
    return az, el


def angular_distance_deg(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Pairwise angle (deg) between camera positions seen from the origin: [len(p), len(q)]."""
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    v = q / np.linalg.norm(q, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(u @ v.T, -1.0, 1.0)))


def nearest_angular_distance_deg(eval_pos: np.ndarray, poisoned_pos: np.ndarray) -> np.ndarray:
    if len(poisoned_pos) == 0:
        raise ValueError("no poisoned views")
    return angular_distance_deg(eval_pos, poisoned_pos).min(axis=1)


# ---- E3 -----------------------------------------------------------------------
def distance_to_mask(m_dil: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) from every pixel to the nearest M_dil pixel (0 inside)."""
    src = (~m_dil).astype(np.uint8)          # zeros are the 'targets' distanceTransform measures to
    return cv2.distanceTransform(src, cv2.DIST_L2, 5)


def band_labels(edges=BAND_EDGES) -> List[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append("%d-%d" % (lo, hi) if hi is not None else ">%d" % lo)
    return out


def band_masks(m_dil: np.ndarray, edges=BAND_EDGES) -> List[np.ndarray]:
    """Collateral-region bands lo < d <= hi (last band d > lo). Subsets of ~m_dil."""
    d = distance_to_mask(m_dil)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (d > lo) if hi is None else ((d > lo) & (d <= hi))
        out.append(b & ~m_dil)
    return out


def band_mse(sq_err: np.ndarray, bands: Sequence[np.ndarray]) -> List[float]:
    """Mean of sq_err over each band; nan where the band is empty."""
    return [float(sq_err[b].mean()) if b.any() else float("nan") for b in bands]
