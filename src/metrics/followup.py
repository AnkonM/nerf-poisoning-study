"""Phase 8 follow-up diagnostics, Part 2 (D-044). Pure functions over already-computed D-042
per-run / per-view tables; nothing here recomputes a D-042 metric or writes anywhere."""

import csv
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np

from metrics import aggregate as A


# ---- inputs -------------------------------------------------------------------------------
def read_manifest_poisoned(path: str) -> Dict[str, set]:
    """dataset_id -> set of poisoned view_index (rows with poisoned == True)."""
    out = defaultdict(set)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["dataset_id"], set())
            if r["poisoned"] == "True":
                out[r["dataset_id"]].add(int(r["view_index"]))
    return dict(out)


def read_per_view(path: str) -> List[Dict]:
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def read_reference(path: str) -> Dict[int, Dict]:
    with open(path, newline="") as f:
        return {int(r["view_index"]): {"included": int(r["included"]), "mse": float(r["gt_vs_plate_mse"])}
                for r in csv.DictReader(f)}


# ---- 2a -----------------------------------------------------------------------------------
def assert_paired_views(poisoned: Dict[str, set], a: str, b: str, seeds: Sequence[int]) -> None:
    for s in seeds:
        ia, ib = "%s_seed%d" % (a, s), "%s_seed%d" % (b, s)
        if ia not in poisoned or ib not in poisoned:
            raise AssertionError("manifest has no dataset %s or %s" % (ia, ib))
        if poisoned[ia] != poisoned[ib]:
            raise AssertionError("seed %d: %s and %s do not poison the identical view set" % (s, ia, ib))


def paired_differences(rows: Sequence[Dict], a: str, b: str, cols: Sequence[str],
                       seeds: Sequence[int]) -> List[Dict]:
    """Per metric: per-seed (b - a), mean, sample std (ddof=1), sign counts, consistency."""
    out = []
    for col in cols:
        va = {r["seed"]: r[col] for r in rows if r["condition"] == a}
        vb = {r["seed"]: r[col] for r in rows if r["condition"] == b}
        d = np.array([vb[s] - va[s] for s in seeds], dtype=np.float64)
        n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
        rec = {"pair": "%s_minus_%s" % (b, a), "metric": col}
        for s, x in zip(seeds, d):
            rec["diff_seed%d" % s] = float(x)
        rec.update({"mean_diff": float(d.mean()), "std_diff_ddof1": float(d.std(ddof=1)),
                    "n_positive": n_pos, "n_negative": n_neg,
                    "sign_consistent": bool(n_pos == len(seeds) or n_neg == len(seeds))})
        out.append(rec)
    return out


# ---- 2b -----------------------------------------------------------------------------------
def rank_desc(values: Sequence[float]) -> List[int]:
    """Rank 1 = largest value (ordinal; ties broken by position)."""
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks.tolist()


def seed_grid(rows: Sequence[Dict], cols: Sequence[str], conditions: Sequence[str],
              seeds: Sequence[int]):
    """Returns (grid rows, per-seed summary rows, residual-correlation rows)."""
    grid, summary, corr = [], [], []
    for col in cols:
        val = {(r["condition"], r["seed"]): r[col] for r in rows}
        cmean = {c: float(np.mean([val[(c, s)] for s in seeds])) for c in conditions}
        resid = {(c, s): val[(c, s)] - cmean[c] for c in conditions for s in seeds}
        for s in seeds:
            ranks = rank_desc([val[(c, s)] for c in conditions])
            for c, rk in zip(conditions, ranks):
                grid.append({"metric": col, "seed": s, "condition": c, "value": val[(c, s)],
                             "rank_in_seed_desc": rk, "residual_from_condition_mean": resid[(c, s)],
                             "spread_kind": A.spread_kind(c)})
            summary.append({"metric": col, "seed": s,
                            "mean_residual_over_conditions": float(np.mean([resid[(c, s)] for c in conditions]))})
        for s1, s2 in combinations(seeds, 2):
            x = np.array([resid[(c, s1)] for c in conditions])
            y = np.array([resid[(c, s2)] for c in conditions])
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
            corr.append({"metric": col, "seed_a": s1, "seed_b": s2, "pearson_r_of_residuals": r,
                         "n_conditions": len(conditions)})
    return grid, summary, corr


# ---- 2c -----------------------------------------------------------------------------------
def mse_from_psnr(psnr: np.ndarray) -> np.ndarray:
    """MSE = 10^(-PSNR/10) (data range 1); infinite PSNR -> 0."""
    psnr = np.asarray(psnr, dtype=np.float64)
    out = np.power(10.0, -psnr / 10.0)
    out[np.isposinf(psnr)] = 0.0
    return out


def psnr_from_mse_value(mse: float) -> float:
    return float("inf") if mse <= 0 else float(-10.0 * np.log10(mse))


REGION_COLUMNS = {   # region -> (per-view PSNR column or None, per-view MSE column or None, pixel-count column, included-only)
    "tgt": ("tgt_psnr", None, "n_tgt_px", True),
    "col": ("col_psnr", None, "n_col_px", False),
    "plate": ("plate_psnr", "plate_mse", "n_tgt_px", True),
}


def run_region_aggregates(per_view: Sequence[Dict], region: str) -> Dict[str, float]:
    """The three aggregates of one run and region: mean of per-view PSNR (D-043), pixel-weighted
    pooled MSE -> PSNR, and view-mean MSE -> PSNR."""
    pcol, mcol, ncol, inc_only = REGION_COLUMNS[region]
    v = [r for r in per_view if (r["included"] == 1 or not inc_only)]
    psnr = np.array([r[pcol] for r in v], dtype=np.float64)
    mse = np.array([r[mcol] for r in v], dtype=np.float64) if mcol else mse_from_psnr(psnr)
    w = np.array([r[ncol] for r in v], dtype=np.float64)
    return {"n_views": len(v),
            "mean_of_psnr": float(psnr.mean()),
            "pooled_mse_pixel_weighted_psnr": psnr_from_mse_value(float((mse * w).sum() / w.sum())),
            "pooled_mse_view_mean_psnr": psnr_from_mse_value(float(mse.mean()))}


AGGREGATES = ("mean_of_psnr", "pooled_mse_pixel_weighted_psnr", "pooled_mse_view_mean_psnr")


def condition_aggregate_table(run_rows: Sequence[Dict]) -> List[Dict]:
    """run_rows: one dict per (run, region) holding the three aggregates. -> per condition/region/aggregate:
    mean over seeds, sample std (ddof=1), per-seed values, min, max."""
    out = []
    by = defaultdict(list)
    for r in run_rows:
        by[(r["condition"], r["region"])].append(r)
    for (cond, region), rs in by.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        for agg in AGGREGATES:
            v = np.array([r[agg] for r in rs])
            out.append({"condition": cond, "region": region, "aggregate": agg, "n": len(v),
                        "mean": float(v.mean()), "std_ddof1": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                        "min": float(v.min()), "max": float(v.max()),
                        "per_seed": ";".join("%d:%r" % (r["seed"], x) for r, x in zip(rs, v.tolist()))})
    return out


def separation_flips(run_rows: Sequence[Dict], conditions: Sequence[str], regions: Sequence[str]) -> List[Dict]:
    """For every condition pair and region: overlap/separated under each aggregate (D-043 rule: per-seed
    [min, max] intervals) and whether any alternative differs from the D-043 aggregate."""
    vals = defaultdict(list)
    for r in sorted(run_rows, key=lambda r: r["seed"]):
        for agg in AGGREGATES:
            vals[(r["condition"], r["region"], agg)].append(r[agg])
    out = []
    for region in regions:
        for ca, cb in combinations(conditions, 2):
            st = {agg: ("overlap" if A.ranges_overlap(vals[(ca, region, agg)], vals[(cb, region, agg)])
                        else "separated") for agg in AGGREGATES}
            out.append({"region": region, "cond_a": ca, "cond_b": cb,
                        **{"statement_" + k: v for k, v in st.items()},
                        "flips_vs_mean_of_psnr": bool(any(st[a] != st["mean_of_psnr"] for a in AGGREGATES[1:])),
                        "d043_pair": bool((ca, cb) in [tuple(p) for p in A.PAIRS.values()])})
    return out


# ---- 2d -----------------------------------------------------------------------------------
def farther_than_reference(per_view: Sequence[Dict], ref: Dict[int, Dict]) -> Dict[str, float]:
    """Target-region MSE: d_gt = MSE(render, GT), d_plate = plate_mse, d_ref = MSE(GT, plate)."""
    v = [r for r in per_view if r["included"] == 1]
    d_gt = mse_from_psnr(np.array([r["tgt_psnr"] for r in v]))
    d_plate = np.array([r["plate_mse"] for r in v])
    d_ref = np.array([ref[int(r["view_index"])]["mse"] for r in v])
    both = (d_gt > d_ref) & (d_plate > d_ref)
    n = len(v)
    return {"n_views": n, "n_both": int(both.sum()), "fraction_both": float(both.sum() / n),
            "n_gt_only_farther": int((d_gt > d_ref).sum()), "n_plate_only_farther": int((d_plate > d_ref).sum()),
            # context: over views, the largest value of min(d_gt, d_plate) / d_ref (> 1 would mean farther from both)
            "max_over_views_min_ratio": float(np.minimum(d_gt / d_ref, d_plate / d_ref).max())}


# ---- 2e -----------------------------------------------------------------------------------
def spread_stats(x: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "q1": float(q1), "median": float(med), "q3": float(q3),
            "iqr": float(q3 - q1), "min": float(x.min()), "max": float(x.max()),
            "range": float(x.max() - x.min())}
