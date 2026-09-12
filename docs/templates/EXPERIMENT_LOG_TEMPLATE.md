# Experiment Log — <run_id>

Fill this out **before** a run's metrics are allowed into
`experiments/results/results.csv`. Copy this file to
`experiments/logs/<run_id>.md` and fill in every field — an empty field is a
sign the run isn't actually reproducible yet.

## Identification

- **Run ID:**
- **Condition:** (e.g. C3 — 20% budget, hard erasure, random selection)
- **Seed:**
- **Config file path:** `configs/poisoning/<...>.yaml`
- **Git commit hash:**
- **Date/time started:**
- **Date/time finished:**

## Environment

- **Where run:** local (WSL2) / Colab / Kaggle
- **GPU:**
- **PyTorch / CUDA versions:**
- **Environment file used:** (path + checksum if modified since last run)

## Data

- **Poisoned image set:** `data/poisoned/<condition_id>/` (+ manifest
  checksum)
- **Eval set used:** `data/blender_scenes/eval_holdout/` (+ checksum —
  confirm it matches the frozen checksum in `DECISION_LOG.md`)

## Training

- **Iterations/epochs:**
- **Final training loss:**
- **Convergence check:** (loss curve inspected — converged / diverged /
  flagged for review)
- **Anomalies observed:** (anything unusual — OOM, restart, manual
  intervention; if none, write "none")

## Metrics (computed by `scripts/evaluate.py`, not by hand)

| Metric | Value |
|---|---|
| Masked PSNR (target region) | |
| Masked SSIM (target region) | |
| Masked LPIPS (target region) | |
| Unmasked PSNR (non-target) | |
| Unmasked SSIM (non-target) | |
| Unmasked LPIPS (non-target) | |
| Distance to background_plate (target region) | |

## Notes

(Anything relevant to interpreting this run that isn't captured above.)
