# Experiment Log — phase3_poc_budget_20

Phase 3 PoC only (docs/ROADMAP.md Phase 3) — mechanism-verification, no
scientific conclusion drawn. Adapted from
docs/templates/EXPERIMENT_LOG_TEMPLATE.md (see phase3_poc_budget_00.md for
the general adaptation note).

## Identification

- **Run ID:** phase3_poc_budget_20
- **Condition:** Phase 3 PoC — 20% budget, hard erasure, random selection (seed 0). Not one of METHODOLOGY.md §5's C1-C7 (uses `lego_sanity.yaml`, not `final_scene.yaml`).
- **Seed:** 0 (both model init and random view-selection sampling)
- **Config file path:** `configs/poisoning/phase3_poc_budget_20.yaml`
- **Git commit hash:** `66077b037171c72b1d8481094671a0b367e2f587`
- **Date/time started:** 2026-09-13 18:47 (+02:00)
- **Date/time finished:** 2026-09-13 20:37:41 (+02:00)

## Environment

- **Where run:** local (WSL2), RTX 5060 laptop
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU
- **PyTorch / CUDA versions:** torch 2.14.0+cu130, CUDA 13.0
- **Environment file used:** project `.venv` (uv), unchanged since D-016

## Data

- **Poisoned image set:** `data/poisoned/phase3_poc_budget_20/` — 20 of 100 train views poisoned (hard erasure, view indices 1, 3, 6, 15, 22, 26, 42, 46, 52, 53, 55, 57, 59, 61, 68, 70, 73, 83, 92, 93; val/test symlinked unchanged from `data/nerf_synthetic/lego`). Manifest rows in `data/poisoned/MANIFEST.csv` (condition_id=phase3_poc_budget_20).
- **Eval set used:** same deviation as phase3_poc_budget_00.md — `data/nerf_synthetic/lego`'s `test`/`val` splits (no `eval_holdout/` yet).

## Training

- **Iterations/epochs:** 30,000 (D-018)
- **Final training loss:** 0.011897 (final train-batch MSE — higher and noisier than the control's 0.007276, consistent with 20% of training batches now targeting a poisoned/flattened region)
- **Convergence check:** converged — checked against the exported TensorBoard curves (`experiments/results/phase3_poc/phase3_poc_budget_20/curves.csv`, `train_loss.png`, `val_psnr_subset.png`). First-half vs. second-half mean training loss: 0.01731 -> 0.01138 (trending down, despite visibly more per-batch noise than the control, expected since batches drawn from poisoned vs. clean views have different optimal targets in the same image region); no NaN/Inf; worst loss value (0.04850) at iteration 500, not late-training. val-subset PSNR (5-image, clean, unpoisoned subset): 23.29 dB (iter 10k) -> 23.09 dB (iter 20k) -> 25.16 dB (iter 30k) — the small dip at 20k is within normal noise for a 5-image subset (not a regression: it's followed by the run's highest value at 30k), and all three points sit below the control's 25.75/27.19/27.93 dB at the same iterations, the expected suppression signature. No divergence.
- **Anomalies observed:** none.

## Metrics (PoC signal only — NOT docs/METHODOLOGY.md §6 metrics; see phase3_poc_budget_20_eval.md)

| Metric | Value |
|---|---|
| Masked PSNR (target region, 4 fixed held-out views) | 19.358 dB (PoC signal only) |
| Masked SSIM (target region) | not computed — Phase 5 |
| Masked LPIPS (target region) | not computed — Phase 5 |
| Unmasked PSNR (whole frame, 4 fixed held-out views) | 24.341 dB (PoC signal only) |
| Unmasked SSIM | not computed |
| Unmasked LPIPS | not computed |
| Distance to background_plate (target region) | not computed — no real background_plate for Lego (D-017) |

## Notes

Training-side half of a training/eval pair — see `phase3_poc_budget_20_eval.md`.
Masked/unmasked PSNR both clearly lower than the control (23.591/28.269 dB)
at the same 4 held-out views — visible suppression signal, as expected.
