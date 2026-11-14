#!/usr/bin/env python
"""Phase 9 decision evidence E1-E7 and the D-045 rule (plan of record: DECISION_LOG.md D-045).

Descriptive; reads existing CSVs, poses and the manifest only. Never trains, never builds a dataset, never writes
per_run.csv or any existing output. Writes experiments/results/phase9/decision_evidence.{csv,md} and gates.csv.

Usage:  python scripts/phase9_decision_evidence.py configs/analysis/phase9_decision_evidence.yaml
"""

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from metrics import aggregate as A
from metrics import followup as FU
from metrics import phase9_evidence as P9
from poisoning.view_selection import sample_farthest_point
from utils.config import load_config

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def write_csv(path, rows):
    cols = list(rows[0].keys())
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    os.replace(path + ".tmp", path)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = load_config(sys.argv[1])
    P = lambda k: os.path.join(REPO, cfg["inputs"][k])
    out = os.path.join(REPO, cfg["outputs"]["dir"])
    os.makedirs(out, exist_ok=True)
    seeds, conds, budgets, aggs, rule = cfg["seeds"], cfg["hard_erasure_conditions"], cfg["budgets"], cfg["aggregates"], cfg["rule"]

    per_run = {r["run_id"]: r for r in A.read_per_run(P("per_run_csv"))}
    ev_pos = A.positions_from_transforms(json.load(open(P("eval_transforms")))["frames"])
    tr_pos = A.positions_from_transforms(json.load(open(P("train_transforms")))["frames"])
    pois = FU.read_manifest_poisoned(P("manifest_csv"))
    assert len(ev_pos) == 75 and len(tr_pos) == 100

    def dsid(c, s):
        return "C7" if c == "C7" else "%s_seed%d" % (c, s)

    rows = []      # decision_evidence.csv: quantity, scope, aggregate, value, detail
    def add(q, scope, agg, value, detail=""):
        rows.append({"quantity": q, "scope": scope, "aggregate": agg, "value": float(value) if isinstance(value, (int, float, np.floating)) else value, "detail": detail})

    # ---- E1 (existing conditions) ---------------------------------------------------------
    e1 = {}
    for c in conds:
        for s in seeds:
            views = sorted(pois[dsid(c, s)])
            exp = round(budgets[c] / 100 * 100)
            assert len(views) == exp, (c, s, len(views))
            near = A.nearest_angular_distance_deg(ev_pos, tr_pos[views])
            e1[(c, s)] = P9.e1_stats(near)
            for k, v in e1[(c, s)].items():
                add("E1_" + k, "%s_seed%d" % (c, s), "", v, "n_poisoned=%d" % len(views))

    # ---- E2 (C8 plan, no dataset built) -----------------------------------------------------
    cams = json.load(open(P("cameras_json")))
    v_target = sorted(int(n.split("_")[1]) for n in cams["v_target"]["views"])
    c8 = {}
    for s in seeds:
        sel, n = sample_farthest_point(tr_pos, v_target, cfg["c8"]["budget_percent"], s)
        assert n == 20 and len(set(sel)) == 20
        c8[s] = P9.e1_stats(A.nearest_angular_distance_deg(ev_pos, tr_pos[sel]))
        for k, v in c8[s].items():
            add("E2_" + k, "C8_seed%d" % s, "", v, "views=" + " ".join(map(str, sel)))

    # ---- per-run aggregates -----------------------------------------------------------------
    agg = {}
    for c in conds:
        for s in seeds:
            rid = "%s_seed%d" % (c, s)
            pv = FU.read_per_view(os.path.join(REPO, cfg["inputs"]["per_view_dir"], rid + ".csv"))
            assert len(pv) == 75
            for region in ("tgt", "plate"):
                a = FU.run_region_aggregates(pv, region)
                for k in aggs:
                    agg[(c, s, region, k)] = a[k]
            # the mean-of-PSNR aggregate must equal per_run.csv's value (consistency, not recomputation)
            assert abs(agg[(c, s, "tgt", "mean_of_psnr")] - float(per_run[rid]["tgt_psnr"])) < 1e-6, rid

    cond_vals = lambda c, region, k: [agg[(c, s, region, k)] for s in seeds]

    # ---- E3 ---------------------------------------------------------------------------------
    e3 = {}
    for setname, cset in (("18run_C1-C5+C7", conds), ("15run_C1-C5", [c for c in conds if c != "C7"])):
        for k in aggs:
            x, y, z = [], [], []
            for c in cset:
                for s in seeds:
                    x.append(e1[(c, s)]["mean_deg"]); y.append(agg[(c, s, "tgt", k)]); z.append(budgets[c])
            e3[(setname, k)] = P9.partial_spearman(x, y, z)
            add("E3_partial_spearman", setname, k, e3[(setname, k)], "n=%d; E1 mean vs target PSNR | budget" % len(x))

    # ---- E4 ---------------------------------------------------------------------------------
    e4 = {}
    for k in aggs:
        e4[k] = P9.resolvability_ratio(cond_vals("C7", "tgt", k), cond_vals("C3", "tgt", k))
        add("E4_R", "C7_vs_C3_target", k, e4[k],
            "mean(C7)=%.4f mean(C3)=%.4f sd(C3)=%.4f" % (np.mean(cond_vals("C7", "tgt", k)), np.mean(cond_vals("C3", "tgt", k)), np.std(cond_vals("C3", "tgt", k), ddof=1)))

    # ---- E5 ---------------------------------------------------------------------------------
    e5 = {}
    for region in ("tgt", "plate"):
        for k in aggs:
            hi, lo = cond_vals("C4", region, k), cond_vals("C5", region, k)
            e5[(region, k)] = P9.step_ratio(hi, lo)
            add("E5_step_ratio", "C4_to_C5_" + region, k, e5[(region, k)],
                "mean(C4)=%.4f mean(C5)=%.4f" % (np.mean(hi), np.mean(lo)))

    # ---- E7 (read-only probes) ----------------------------------------------------------------
    pr = cfg["probes"]
    for d in pr["real_photo_dirs"]:
        add("E7_dir_exists", d, "", int(os.path.isdir(os.path.join(REPO, d))))
    for n in pr["import_names"]:
        add("E7_importable_find_spec", n, "", int(importlib.util.find_spec(n) is not None), "find_spec only; not imported")
    for pat in pr["inpaint_code_grep"]:
        hits = subprocess.run(["grep", "-rIl", "-i", "--exclude=phase9_decision_evidence.py", pat, "src", "scripts"], cwd=REPO, capture_output=True, text=True).stdout.split()
        add("E7_code_mentions", pat, "", len(hits), " ".join(hits))
    nvcc = shutil.which("nvcc")
    add("E7_nvcc", "path", "", nvcc or "absent")
    if nvcc:
        add("E7_nvcc", "version", "", subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1])

    # ---- E6 + G4 --------------------------------------------------------------------------------
    cost, g4 = None, None
    sp = P("speed_json")
    if os.path.exists(sp):
        S = json.load(open(sp))
        per_run_h = P9.hours_per_run(rule["iterations"], S["its_per_s"], rule["validation_events"], S["val_seconds_per_event"],
                                     S["val_seconds_per_event"] / rule["validation_views"] * rule["final_eval_views"])
        n_runs = rule["calibration_runs"] + rule["chosen_option_runs"]
        cost = {"hours_per_run": per_run_h, "calibration_plus_option_hours": n_runs * per_run_h,
                "both_options_hours": (n_runs + rule["follow_up_runs_if_both"]) * per_run_h,
                "calibration_seeds_2_3_extra_hours": 2 * per_run_h}
        for k, v in cost.items():
            add("E6_" + k, "laptop", "", v, "its_per_s=%.4f val_s=%.2f" % (S["its_per_s"], S["val_seconds_per_event"]))
        g4 = cost["calibration_plus_option_hours"] <= rule["cost_cap_hours"]
    else:
        add("E6_cost", "laptop", "", "speed_json missing", sp)

    # ---- gates -------------------------------------------------------------------------------
    c3_min = min(e1[("C3", s)]["mean_deg"] for s in seeds)
    c7_val = e1[("C7", 1)]["mean_deg"]
    g1 = all(c8[s]["mean_deg"] < c3_min and c8[s]["mean_deg"] < c7_val for s in seeds)
    g2 = all(v > 0 for v in e3.values())
    g3 = all(v >= rule["g3_min_resolvability_ratio"] for v in e4.values())
    c8_q = bool(g1 and g2 and g3 and g4) if g4 is not None else None
    e5_ok = all(v >= rule["b100_min_step_ratio"] for v in e5.values())
    b100_q = bool(e5_ok and (cost["calibration_plus_option_hours"] <= rule["cost_cap_hours"])) if cost else None
    both_fit = (cost["both_options_hours"] <= rule["cost_cap_hours"]) if cost else None
    if c8_q:
        choice = "C8"
    elif b100_q:
        choice = "B100"
    elif c8_q is None or b100_q is None:
        choice = "UNDECIDED (cost evidence missing)"
    else:
        choice = "NONE"
    gates = [
        {"gate": "G1", "detail": "C8 per-seed E1 means %s < min C3 %.3f and < C7 %.3f" % (
            [round(c8[s]["mean_deg"], 3) for s in seeds], c3_min, c7_val), "pass": g1},
        {"gate": "G2", "detail": "partial Spearman " + "; ".join("%s/%s=%.3f" % (a, b, v) for (a, b), v in e3.items()), "pass": g2},
        {"gate": "G3", "detail": "R " + "; ".join("%s=%.3f" % (k, v) for k, v in e4.items()) + " (>= %.1f)" % rule["g3_min_resolvability_ratio"], "pass": g3},
        {"gate": "G4", "detail": ("%.2f h for %d runs vs cap %d" % (cost["calibration_plus_option_hours"], rule["calibration_runs"] + rule["chosen_option_runs"], rule["cost_cap_hours"])) if cost else "no speed test", "pass": g4},
        {"gate": "C8_qualifies", "detail": "G1 and G2 and G3 and G4", "pass": c8_q},
        {"gate": "B100_E5", "detail": "ratios " + "; ".join("%s/%s=%.3f" % (a, b, v) for (a, b), v in e5.items()) + " (>= %.1f)" % rule["b100_min_step_ratio"], "pass": e5_ok},
        {"gate": "B100_qualifies", "detail": "E5 and cap", "pass": b100_q},
        {"gate": "both_fit_cap_together", "detail": ("%.2f h" % cost["both_options_hours"]) if cost else "", "pass": both_fit},
        {"gate": "SELECTION", "detail": choice, "pass": ""},
    ]
    for g in gates:
        print("%-22s %-5s %s" % (g["gate"], {True: "PASS", False: "FAIL", None: "n/a", "": ""}[g["pass"]], g["detail"]))
    write_csv(os.path.join(out, "decision_evidence.csv"), rows)
    write_csv(os.path.join(out, "gates.csv"), [{"gate": g["gate"], "pass": g["pass"], "detail": g["detail"]} for g in gates])

    with open(os.path.join(out, "decision_evidence.md"), "w") as f:
        f.write("# Phase 9 decision evidence (D-045)\n\n| quantity | scope | aggregate | value | detail |\n|---|---|---|---|---|\n")
        for r in rows:
            v = ("%.4f" % r["value"]) if isinstance(r["value"], float) else r["value"]
            d = r["detail"] if len(str(r["detail"])) < 120 else str(r["detail"])[:117] + "..."
            f.write("| %s | %s | %s | %s | %s |\n" % (r["quantity"], r["scope"], r["aggregate"], v, d))
        f.write("\n## Gates\n\n| gate | result | detail |\n|---|---|---|\n")
        for g in gates:
            f.write("| %s | %s | %s |\n" % (g["gate"], g["pass"], g["detail"]))


if __name__ == "__main__":
    main()
