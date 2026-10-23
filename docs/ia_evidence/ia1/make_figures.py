"""IA-1 figures. Reads values from the tracked 4-view summary JSONs; run from repo root:
    .venv/bin/python docs/ia_evidence/ia1/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RES = Path("experiments/results/phase3_poc")
OUT = Path("docs/ia_evidence/ia1")
BUDGETS = [0, 20, 50]

summ = {b: json.loads((RES / f"phase3_poc_budget_{b:02d}_4view_summary.json").read_text()) for b in BUDGETS}

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(BUDGETS, [summ[b]["mean_masked_psnr"] for b in BUDGETS], "ko-", label="masked (target region)")
ax.plot(BUDGETS, [summ[b]["mean_full_psnr"] for b in BUDGETS], "ks--", mfc="white", label="full frame")
for b in BUDGETS:
    ax.annotate(f'{summ[b]["mean_masked_psnr"]:.1f}', (b, summ[b]["mean_masked_psnr"]), textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_xticks(BUDGETS)
ax.set_xlabel("poisoning budget (% of target-visible training views)")
ax.set_ylabel("PSNR (dB), mean of 4 held-out views")
ax.set_title("Lego PoC: hard erasure, random view selection")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "psnr_vs_budget.png", dpi=200, facecolor="white")

# each render file is 800x400: left half = ground truth, right half = NeRF render
def halves(path):
    im = Image.open(path).convert("RGB")
    w = im.width // 2
    return im.crop((0, 0, w, im.height)), im.crop((w, 0, im.width, im.height))

view = "200"
panels, titles = [], []
for b in BUDGETS:
    f = RES / summ[b]["views"][view]["file"]
    gt, render = halves(f)
    if b == 0:
        panels.append(gt); titles.append("ground truth")
    panels.append(render)
    titles.append(("clean render (0%)" if b == 0 else f"{b}% poisoned") + f"\n{summ[b]['views'][view]['masked_psnr']:.1f} dB masked")
fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.8))
for a, p, t in zip(axes, panels, titles):
    a.imshow(p); a.set_title(t, fontsize=9, pad=8); a.axis("off")
fig.tight_layout(pad=1.5)
fig.savefig(OUT / "lego_strip.png", dpi=200, facecolor="white")
