# Experiment Log — phase3_poc_budget_20_eval

Evaluation run for `phase3_poc_budget_20` (see `phase3_poc_budget_20.md`
for the training run this evaluates). PSNR here is a PoC signal, not a
docs/METHODOLOGY.md §6 metric.

## Identification

- **Run ID:** phase3_poc_budget_20_eval
- **Condition:** evaluation of phase3_poc_budget_20 (20% budget, hard erasure, random)
- **Config file path:** `configs/poisoning/phase3_poc_budget_20.yaml`
- **Checkpoint evaluated:** `experiments/runs/phase3_poc_budget_20/030000.tar`
- **Git commit hash:** `66077b037171c72b1d8481094671a0b367e2f587`
- **Date/time:** 2026-09-13 22:42:56 – 22:44:05 (+02:00, run back-to-back with the other two eval runs)

## Environment

- **Where run:** local (WSL2), RTX 5060 laptop, same environment as training.

## Data / views evaluated

- **Held-out view indices:** 200, 266, 333, 399 (same fixed indices as phase3_poc_budget_00_eval, for direct comparability).
- **Eval set used:** `data/nerf_synthetic/lego` test split, via this condition's `dataset.path` -> `data/poisoned/phase3_poc_budget_20`'s symlinked (unmodified) `test`/`val`.

## Invocation

Same supplementary script as `phase3_poc_budget_00_eval.md`,
**`scripts/phase3_poc_eval_views.py`** (durable repo path — see
`docs/DECISION_LOG.md`):

```
python scripts/phase3_poc_eval_views.py configs/poisoning/phase3_poc_budget_20.yaml \
    experiments/runs/phase3_poc_budget_20/030000.tar phase3_poc_budget_20
```

## Metrics (PoC signal only)

| View | Full-frame PSNR | Masked (target-region) PSNR |
|---|---|---|
| 200 | 23.876 dB | 18.917 dB |
| 266 | 25.464 dB | 20.599 dB |
| 333 | 24.308 dB | 19.158 dB |
| 399 | 23.714 dB | 18.758 dB |
| **Mean** | **24.341 dB** | **19.358 dB** |

## Notes

Both metrics clearly below the control's (28.269 / 23.591 dB) at the same
4 views — a visible suppression signal at 20% budget, before the larger
drop at 50% (see phase3_poc_budget_50_eval.md).
