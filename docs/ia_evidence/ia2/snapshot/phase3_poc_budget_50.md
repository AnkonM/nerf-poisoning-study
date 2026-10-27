# Experiment Log — phase3_poc_budget_50

Phase 3 PoC only (docs/ROADMAP.md Phase 3) — mechanism-verification, no
scientific conclusion drawn. Adapted from
docs/templates/EXPERIMENT_LOG_TEMPLATE.md (see phase3_poc_budget_00.md for
the general adaptation note).

## Identification

- **Run ID:** phase3_poc_budget_50
- **Condition:** Phase 3 PoC — 50% budget, hard erasure, random selection (seed 0). Not one of METHODOLOGY.md §5's C1-C7 (uses `lego_sanity.yaml`, not `final_scene.yaml`).
- **Seed:** 0 (both model init and random view-selection sampling)
- **Config file path:** `configs/poisoning/phase3_poc_budget_50.yaml`
- **Git commit hash:** `66077b037171c72b1d8481094671a0b367e2f587`
- **Date/time started:** 2026-09-13 20:38 (+02:00)
- **Date/time finished:** 2026-09-13 22:38:41 (+02:00)

## Environment

- **Where run:** local (WSL2), RTX 5060 laptop
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU
- **PyTorch / CUDA versions:** torch 2.14.0+cu130, CUDA 13.0
- **Environment file used:** project `.venv` (uv), unchanged since D-016

## Data

- **Poisoned image set:** `data/poisoned/phase3_poc_budget_50/` — 50 of 100 train views poisoned (hard erasure; view indices per `data/poisoned/MANIFEST.csv`, condition_id=phase3_poc_budget_50; val/test symlinked unchanged from `data/nerf_synthetic/lego`).
- **Eval set used:** same deviation as phase3_poc_budget_00.md — `data/nerf_synthetic/lego`'s `test`/`val` splits (no `eval_holdout/` yet).

## Training

- **Iterations/epochs:** 30,000 (D-018)
- **Final training loss:** 0.005972 (final train-batch MSE — this single-batch endpoint value is not directly comparable across conditions; see convergence check below for the more informative val-subset trend)
- **Convergence check:** converged — checked against the exported TensorBoard curves (`experiments/results/phase3_poc/phase3_poc_budget_50/curves.csv`, `train_loss.png`, `val_psnr_subset.png`), with visibly more per-batch loss noise than both other conditions (expected: half of training batches now target the flattened erasure region). First-half vs. second-half mean training loss: 0.02106 -> 0.01100 (trending down); no NaN/Inf in any of the 60 logged samples; worst loss value (0.05503) at iteration 500, not late-training — the largest visible late-training bump (~0.031 at iter ~22,500) is well below that early peak, consistent with normal SGD noise, not divergence. val-subset PSNR (5-image, clean, unpoisoned subset): 19.46 dB (iter 10k) -> 20.19 dB (iter 20k) -> 19.69 dB (iter 30k) — the small (~0.5 dB) dip from 20k to 30k is within normal noise for a 5-image subset, not a real regression (the authoritative post-training number is the full-precision 4-fixed-view eval in `phase3_poc_budget_50_eval.md`: 16.862 dB full / 11.637 dB masked). Clearly the lowest of the three conditions at every matched iteration (control: 25.75/27.19/27.93 dB; 20%: 23.29/23.09/25.16 dB), consistent with a budget-monotonic suppression effect. No divergence.
- **Anomalies observed:** none.

## Metrics (PoC signal only — NOT docs/METHODOLOGY.md §6 metrics; see phase3_poc_budget_50_eval.md)

| Metric | Value |
|---|---|
| Masked PSNR (target region, 4 fixed held-out views) | 11.637 dB (PoC signal only) |
| Masked SSIM (target region) | not computed — Phase 5 |
| Masked LPIPS (target region) | not computed — Phase 5 |
| Unmasked PSNR (whole frame, 4 fixed held-out views) | 16.862 dB (PoC signal only) |
| Unmasked SSIM | not computed |
| Unmasked LPIPS | not computed |
| Distance to background_plate (target region) | not computed — no real background_plate for Lego (D-017) |

## Notes

Training-side half of a training/eval pair — see `phase3_poc_budget_50_eval.md`.
Masked PSNR (11.6 dB) is far below both the control (23.6 dB) and 20%
condition (19.4 dB) — a clear, budget-monotonic suppression signal.
Visual side-by-side renders (`experiments/results/phase3_poc/`) show the
target object rendered as a hazy, gray, translucent blob rather than the
sharp Lego model — the qualitative "visibly suppressed" signature the
Phase 3 gate asks for.
