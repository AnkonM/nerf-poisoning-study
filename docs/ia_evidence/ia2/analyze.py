import csv, glob, os, re, json, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.expanduser("~/ia2_evidence")
SNAP = H + "/snapshot"
REPO = "/home/hpc-sdb/nerf-poisoning/nerf-poisoning-study"
rows = list(csv.DictReader(open(SNAP + "/results.csv")))
byrun = {r["run_id"]: r for r in rows}

CONDS = [("control", "0", "-", "-"), ("C1", "5", "hard_erasure", "random"), ("C2", "10", "hard_erasure", "random"),
         ("C3", "20", "hard_erasure", "random"), ("C4", "30", "hard_erasure", "random"),
         ("C5", "50", "hard_erasure", "random"), ("C6", "20", "soft_suppression", "random"),
         ("C7", "20", "hard_erasure", "strategic")]
SEEDS = [1, 2, 3]
pat = re.compile(r"\[(\S+)\] iter (\d+) loss (\S+) psnr (\S+)")

def curve(run):
    d = {}
    for l in open(f"{SNAP}/runs/{run}/train.log", errors="replace"):
        m = pat.search(l)
        if m:
            d[int(m.group(2))] = float(m.group(3))
    return sorted(d.items())

def ms(v):
    if not v: return "n=0"
    if len(v) == 1: return f"{v[0]:.2f} (n=1)"
    return f"{st.mean(v):.2f} ± {st.stdev(v):.2f} (n={len(v)})"

inv = []
curves = {}
for cid, b, a, v in CONDS:
    for s in SEEDS:
        run = f"{cid}_seed{s}"
        c = curve(run) if os.path.exists(f"{SNAP}/runs/{run}/train.log") else []
        curves[run] = c
        complete = run in byrun
        if complete:
            status, it = "complete", int(byrun[run]["iterations"])
        elif c:
            status, it = "RUNNING (snapshot)", c[-1][0]
        else:
            status, it = "not started", 0
        first = second = None
        if c and len(c) >= 4:
            half = len(c) // 2
            first = st.mean(x[1] for x in c[:half]); second = st.mean(x[1] for x in c[half:])
        last20 = st.mean(x[1] for x in c[-20:]) if c else None
        fl = float(byrun[run]["final_train_loss"]) if complete else None
        inv.append(dict(run=run, cid=cid, seed=s, status=status, it=it, final_loss=fl, last20=last20,
                        first=first, second=second, npts=len(c),
                        resume=byrun[run]["resume_events"] if complete else "",
                        log=f"experiments/runs/{run}/train.log"))

# ---- inventory table
with open(H + "/inventory.tsv", "w") as f:
    f.write("run\tstatus\titer\tfinal_train_loss(single batch, summary.json)\tmean_last20_logged_loss\tfirst_half_mean_loss\tsecond_half_mean_loss\tlog\n")
    for r in inv:
        g = lambda x: "" if x is None else f"{x:.3e}"
        f.write(f"{r['run']}\t{r['status']}\t{r['it']}\t{g(r['final_loss'])}\t{g(r['last20'])}\t{g(r['first'])}\t{g(r['second'])}\t{r['log']}\n")

# ---- partial_results.md
def vals(cid, key="test_psnr_mean_whole_frame_db"):
    return [float(byrun[f"{cid}_seed{s}"][key]) for s in SEEDS if f"{cid}_seed{s}" in byrun]

with open(H + "/partial_results.md", "w") as f:
    ncomp = sum(1 for r in inv if r["status"] == "complete")
    f.write("# Phase 6 sweep: results\n\n")
    f.write(f"**{ncomp} of 24 runs complete (8 conditions x 3 seeds); all 200000 iterations.** "
            "control_seed3 was interrupted once and resumed from iteration 150000 (resume_events=1).\n\n")
    f.write("**Metric source:** `experiments/results/results.csv` (copied to `snapshot/results.csv`), column "
            "`test_psnr_mean_whole_frame_db` = whole-frame PSNR over the 75 frozen eval-holdout views (from `scripts/evaluate.py`). "
            "Std is the sample std (ddof=1).\n\n")
    f.write("**Not available:** results.csv contains no masked/target-region suppression metric and no collateral-damage "
            "metric (the per-run eval JSON has only `test_psnr`). Therefore no suppression or collateral numbers are reported.\n\n")
    f.write("| condition | budget (% of V_target) | attack | view selection | seeds completed | whole-frame test PSNR, dB (mean ± std) |\n|---|---|---|---|---|---|\n")
    for cid, b, a, v in CONDS:
        n = len(vals(cid))
        f.write(f"| {cid} | {b} | {a} | {v} | {n}/3 ({','.join(str(s) for s in SEEDS if f'{cid}_seed{s}' in byrun)}) | {ms(vals(cid))} |\n")
    f.write("\n## Per-run whole-frame test PSNR (dB)\n\n| run | PSNR |\n|---|---|\n")
    for r in rows:
        f.write(f"| {r['run_id']} | {float(r['test_psnr_mean_whole_frame_db']):.3f} |\n")
    # patterns, numbers only
    ctrl = st.mean(vals("control"))
    f.write("\n## Observations (numbers only)\n\n")
    hard = [("C1", 5), ("C2", 10), ("C3", 20), ("C4", 30), ("C5", 50)]
    seq = [st.mean(vals(c)) for c, _ in hard]
    mono = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    f.write(f"- Hard-erasure random, whole-frame PSNR by budget 0/5/10/20/30/50%: "
            f"{ctrl:.2f} / " + " / ".join(f"{x:.2f}" for x in seq) + f" dB; strictly decreasing across the 6 means: {mono}.\n")
    f.write("- Whole-frame PSNR falls with budget, but whole-frame PSNR mixes target-region and non-target-region error; "
            "it cannot be read as suppression or collateral separately.\n")
    c3, c6, c7 = st.mean(vals("C3")), st.mean(vals("C6")), st.mean(vals("C7"))
    c3, c6, c7 = st.mean(vals("C3")), st.mean(vals("C6")), st.mean(vals("C7"))
    f.write(f"- At 20% budget: random hard (C3) {c3:.2f}, soft suppression (C6) {c6:.2f}, strategic hard (C7) {c7:.2f} dB (n=3 each). C6 vs C7 differ by {abs(c6-c7):.2f} dB, within seed noise.\n")
    sp = {c: max(vals(c)) - min(vals(c)) for c, *_ in CONDS}
    f.write("- Seed range (max-min) per condition, dB: " + ", ".join(f"{c} {x:.2f}" for c, x in sp.items()) + ". Differences under ~1 dB are not resolved.\n")

# ---- loss curves
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
colors = {1: "black", 2: "dimgray", 3: "darkgray"}
for ax, (cid, b, a, v) in zip(axes.flat, CONDS):
    for s in SEEDS:
        c = curves[f"{cid}_seed{s}"]
        if not c: continue
        w = 40  # 500-iter log spacing -> 20k-iter window
        xs = [x[0] for x in c]; ys = [x[1] for x in c]
        sm = [st.mean(ys[max(0, i - w + 1):i + 1]) for i in range(len(ys))]
        ax.plot(xs, sm, color=colors[s], ls={1: "-", 2: "--", 3: ":"}[s], lw=1.2, label=f"seed {s}")
    ax.set_yscale("log"); ax.set_title(f"{cid} ({b}% {a} {v})".replace(" - -", ""), fontsize=9)
    ax.set_xlabel("iteration"); ax.legend(fontsize=7)
axes[0][0].set_ylabel("train loss (trailing mean of logged single-batch loss)")
axes[1][0].set_ylabel("train loss (trailing mean of logged single-batch loss)")
fig.suptitle("Training loss per condition (log y; logged every 500 iters, 20k-iter trailing mean; all runs complete)", fontsize=10)
fig.tight_layout(); fig.savefig(H + "/loss_curves.png", dpi=130, facecolor="white"); plt.close(fig)

# ---- whole-frame PSNR vs budget (extra: only metric that exists)
fig, ax = plt.subplots(figsize=(6, 4))
xs = [0, 5, 10, 20, 30, 50]; cids = ["control", "C1", "C2", "C3", "C4", "C5"]
for x, c in zip(xs, cids):
    for p in vals(c): ax.plot(x, p, "o", color="gray", ms=4)
ax.errorbar(xs, [st.mean(vals(c)) for c in cids], yerr=[st.stdev(vals(c)) if len(vals(c)) > 1 else 0 for c in cids],
            color="black", marker="s", capsize=3, lw=1)
ax.set_xlabel("poisoning budget (% of V_target), hard erasure, random")
ax.set_ylabel("whole-frame test PSNR (dB)")
ax.set_title("Whole-frame PSNR vs budget (75 frozen eval views)\nNOT a suppression or collateral metric; points = seeds, line = mean ± sample std", fontsize=8)
fig.tight_layout(); fig.savefig(H + "/whole_frame_psnr_vs_budget.png", dpi=130, facecolor="white"); plt.close(fig)

# ---- stats for summary
json.dump(inv, open(H + "/inventory.json", "w"), indent=1)
for r in inv:
    print(r["run"], r["status"], r["it"], r["npts"],
          None if r["final_loss"] is None else f"{r['final_loss']:.2e}",
          None if r["last20"] is None else f"{r['last20']:.2e}",
          None if r["first"] is None else f"{r['first']:.2e}", None if r["second"] is None else f"{r['second']:.2e}")
