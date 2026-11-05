"""Per-run §6 metric computation and the output-file contracts (D-042 definition H).

Writes ONLY under experiments/results/metrics/. Never touches results.csv, runs/,
renders/ or data/.
"""

import csv
import json
import os
from typing import Dict, List

import numpy as np

from metrics import DEFINITION_VERSION
from metrics import image_metrics as M

REGIONS = ("whole", "tgt", "col", "rawtgt")      # rawtgt = M_raw sensitivity variant
KINDS = ("psnr", "ssim", "lpips")

VIEW_COLUMNS = (["view_index", "included", "area_fraction_800", "n_tgt_px", "n_col_px", "n_rawtgt_px"]
                + ["%s_%s" % (r, k) for r in REGIONS for k in KINDS]
                + ["plate_mse", "plate_psnr"])

# Which views each aggregate averages over (D-042 C): target-region metrics use the
# included views only; collateral and whole-frame use all 75.
INCLUDED_ONLY = ("tgt", "rawtgt", "plate")

AGG_COLUMNS = (["%s_%s" % (r, k) for r in REGIONS for k in KINDS] + ["plate_mse", "plate_psnr"])
RUN_COLUMNS = (["run_id", "condition", "seed", "resumed", "config_path", "render_meta_sha256",
                "checkpoint_sha256", "definition_version", "n_views", "n_included",
                "lpips_batch_size", "lpips_oom_fallback", "eval_json_psnr",
                "whole_psnr_dev_vs_eval_json"]
               + AGG_COLUMNS + ["n_inf_psnr_views"])


def compute_view_rows(renders: np.ndarray, gt: np.ndarray, data: Dict, meter) -> List[Dict]:
    m_dil, m_raw = data["m_dil"], data["m_raw"]
    n = renders.shape[0]
    lp = meter.maps(renders, gt)                 # [N,H,W]
    rows = []
    for i in range(n):
        r, g = renders[i], gt[i]
        sq = M.sq_err_map(r, g)
        ss = M.ssim_map(r, g)
        regions = {"whole": np.ones_like(m_dil[i]), "tgt": m_dil[i], "col": ~m_dil[i], "rawtgt": m_raw[i]}
        row = {"view_index": i, "included": int(data["included"][i]),
               "area_fraction_800": float(data["area_fraction_800"][i]),
               "n_tgt_px": int(m_dil[i].sum()), "n_col_px": int((~m_dil[i]).sum()),
               "n_rawtgt_px": int(m_raw[i].sum())}
        for name, reg in regions.items():
            row["%s_psnr" % name] = M.psnr_from_mse(M.region_mse(sq, reg))
            row["%s_ssim" % name] = M.region_mean(ss, reg)
            row["%s_lpips" % name] = M.region_mean(lp[i], reg)
        pmse = M.region_mse(M.sq_err_map(r, data["plates"][i]), m_dil[i])
        row["plate_mse"], row["plate_psnr"] = pmse, M.psnr_from_mse(pmse)
        rows.append(row)
    return rows


def aggregate(view_rows: List[Dict]) -> Dict:
    inc = [r for r in view_rows if r["included"]]
    agg = {"n_views": len(view_rows), "n_included": len(inc)}
    for col in AGG_COLUMNS:
        pool = inc if col.split("_")[0] in INCLUDED_ONLY else view_rows
        agg[col] = float(np.mean([r[col] for r in pool])) if pool else float("nan")
    agg["n_inf_psnr_views"] = sum(1 for r in view_rows for c in ("whole_psnr", "tgt_psnr", "col_psnr",
                                                                 "rawtgt_psnr", "plate_psnr")
                                  if not np.isfinite(r[c]))
    return agg


def reference_gt_vs_plate(data: Dict) -> List[Dict]:
    """Scale for reading plate distance: clean GT vs plate over M_dil, per eval view."""
    rows = []
    for i in range(data["gt"].shape[0]):
        mse = M.region_mse(M.sq_err_map(data["gt"][i], data["plates"][i]), data["m_dil"][i])
        rows.append({"view_index": i, "included": int(data["included"][i]),
                     "gt_vs_plate_mse": mse, "gt_vs_plate_psnr": M.psnr_from_mse(mse)})
    return rows


def _fmt(v):
    if isinstance(v, float):
        return repr(v)           # round-trippable, writes 'inf'/'nan' explicitly
    return v


def write_csv_atomic(path: str, columns: List[str], rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    os.replace(tmp, path)


def upsert_run_row(path: str, row: Dict) -> None:
    """Replace the row with this run_id (or append); never duplicates. Refuses a header
    mismatch. Atomic."""
    rows: List[Dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != RUN_COLUMNS:
                raise ValueError("%s: header mismatch, refusing to rewrite" % path)
            rows = list(rd)
    new = {k: _fmt(v) for k, v in row.items()}
    for i, r in enumerate(rows):
        if r["run_id"] == new["run_id"]:
            rows[i] = new
            break
    else:
        rows.append(new)
    write_csv_atomic(path, RUN_COLUMNS, rows)


def write_json_atomic(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


__all__ = ["DEFINITION_VERSION"]
