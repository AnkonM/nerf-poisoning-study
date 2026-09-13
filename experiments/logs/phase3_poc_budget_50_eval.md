# Experiment Log — phase3_poc_budget_50_eval

Evaluation run for `phase3_poc_budget_50` (see `phase3_poc_budget_50.md`
for the training run this evaluates). PSNR here is a PoC signal, not a
docs/METHODOLOGY.md §6 metric.

## Identification

- **Run ID:** phase3_poc_budget_50_eval
- **Condition:** evaluation of phase3_poc_budget_50 (50% budget, hard erasure, random)
- **Config file path:** `configs/poisoning/phase3_poc_budget_50.yaml`
- **Checkpoint evaluated:** `experiments/runs/phase3_poc_budget_50/030000.tar`
- **Git commit hash:** `66077b037171c72b1d8481094671a0b367e2f587`
- **Date/time:** 2026-09-13 22:44:05 – 22:45:14 (+02:00, run back-to-back with the other two eval runs)

## Environment

- **Where run:** local (WSL2), RTX 5060 laptop, same environment as training.

## Data / views evaluated

- **Held-out view indices:** 200, 266, 333, 399 (same fixed indices as the other two conditions, for direct comparability).
- **Eval set used:** `data/nerf_synthetic/lego` test split, via this condition's `dataset.path` -> `data/poisoned/phase3_poc_budget_50`'s symlinked (unmodified) `test`/`val`.

## Invocation

Same supplementary script as `phase3_poc_budget_00_eval.md`,
**`scripts/phase3_poc_eval_views.py`** (durable repo path — see
`docs/DECISION_LOG.md`):

```
python scripts/phase3_poc_eval_views.py configs/poisoning/phase3_poc_budget_50.yaml \
    experiments/runs/phase3_poc_budget_50/030000.tar phase3_poc_budget_50
```

## Metrics (PoC signal only)

| View | Full-frame PSNR | Masked (target-region) PSNR |
|---|---|---|
| 200 | 16.882 dB | 11.838 dB |
| 266 | 16.633 dB | 11.294 dB |
| 333 | 17.213 dB | 11.742 dB |
| 399 | 16.720 dB | 11.675 dB |
| **Mean** | **16.862 dB** | **11.637 dB** |

## Notes

Clear monotonic ordering across all three conditions on both metrics:

| Condition | Full-frame PSNR | Masked PSNR |
|---|---|---|
| 0% (control) | 28.269 dB | 23.591 dB |
| 20% | 24.341 dB | 19.358 dB |
| 50% | 16.862 dB | 11.637 dB |

Visual inspection of the side-by-side renders
(`experiments/results/phase3_poc/phase3_poc_budget_50_view200_psnr16.88.png`
vs. the control's `..._budget_00_view200_psnr28.58.png`) shows the target
object rendered as a hazy, translucent, roughly gray silhouette rather
than the sharp yellow/gray Lego model in the control — a qualitatively
obvious suppression effect, not just a numeric PSNR drop.
