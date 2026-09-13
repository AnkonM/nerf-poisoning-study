# Experiment Log — phase3_poc_budget_00

Phase 3 PoC only (docs/ROADMAP.md Phase 3) — mechanism-verification, no
scientific conclusion drawn. Adapted from
docs/templates/EXPERIMENT_LOG_TEMPLATE.md: "Condition"/"Eval set" fields
below are Lego-PoC equivalents of the template's C1-C7/eval_holdout
language, since eval_holdout doesn't exist until Phase 4 (noted explicitly
where it deviates).

## Identification

- **Run ID:** phase3_poc_budget_00
- **Condition:** Phase 3 PoC control — 0% budget, hard erasure (n/a at 0%), random selection (n/a at 0%). Not one of METHODOLOGY.md §5's C1-C7.
- **Seed:** 0 (reproducibility.seed, configs/scenes/lego_sanity.yaml)
- **Config file path:** `configs/poisoning/phase3_poc_budget_00.yaml`
- **Git commit hash:** `a2cc6298de1e0e2ede640df0db6a3102e556d8a1` — **caveat:** this is the commit recorded by `scripts/train.py` at run start; the working tree at that moment already contained the (not-yet-committed) Phase 3 poisoning/config code that produced this run's actual dataset and hyperparameters. That code was committed shortly after as `66077b0` (mid-sweep, external to this conversation) — see the budget_20/budget_50 logs, whose recorded hash is `66077b0`. Flagging this discrepancy rather than silently normalizing it; all three runs used the same code, only the recorded hash differs by when each run happened to start relative to that commit.
- **Date/time started:** 2026-09-13 16:58 (+02:00)
- **Date/time finished:** 2026-09-13 18:47:15 (+02:00)

## Environment

- **Where run:** local (WSL2), RTX 5060 laptop
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU
- **PyTorch / CUDA versions:** torch 2.14.0+cu130, CUDA 13.0
- **Environment file used:** project `.venv` (uv), unchanged since D-016

## Data

- **Poisoned image set:** `data/poisoned/phase3_poc_budget_00/` (0 poisoned views — full copy of `data/nerf_synthetic/lego/train/`; val/test are symlinks to the same source, unchanged); manifest rows in `data/poisoned/MANIFEST.csv` (condition_id=phase3_poc_budget_00).
- **Eval set used:** **deviation from template** — `data/blender_scenes/eval_holdout/` doesn't exist yet (Phase 4 deliverable). This PoC's held-out set is `data/nerf_synthetic/lego`'s own `test`/`val` splits (200/100 views), loaded via `dataset.path` in the config.

## Training

- **Iterations/epochs:** 30,000 (see docs/DECISION_LOG.md D-018 for rationale; not the Phase 2/Phase 6 iteration count)
- **Final training loss:** 0.007276 (final train-batch MSE)
- **Convergence check:** converged — checked against the exported TensorBoard curves (`experiments/results/phase3_poc/phase3_poc_budget_00/curves.csv`, `train_loss.png`, `val_psnr_subset.png`, via `scripts/export_tb_curves.py`), not just recalled log-tailing impressions. First-half vs. second-half mean training loss: 0.01073 -> 0.00603 (trending down); no NaN/Inf in any of the 60 logged loss samples; the single worst loss value (0.04472) occurs at iteration 500 (expected early/precrop-warmup noise), not late in training. val-subset PSNR (5-image subset) rose monotonically: 25.75 dB (iter 10k) -> 27.19 dB (iter 20k) -> 27.93 dB (iter 30k). No divergence spikes.
- **Anomalies observed:** none. `torch.meshgrid` deprecation UserWarning only (pre-existing, harmless, unrelated to this run).

## Metrics (PoC signal only — NOT docs/METHODOLOGY.md §6 metrics; see phase3_poc_budget_00_eval.md)

| Metric | Value |
|---|---|
| Masked PSNR (target region, 4 fixed held-out views) | 23.591 dB (PoC signal only) |
| Masked SSIM (target region) | not computed — src/metrics/ SSIM/LPIPS don't exist until Phase 5 |
| Masked LPIPS (target region) | not computed — same reason |
| Unmasked PSNR (whole frame, 4 fixed held-out views) | 28.269 dB (PoC signal only) |
| Unmasked SSIM | not computed |
| Unmasked LPIPS | not computed |
| Distance to background_plate (target region) | not computed — no real background_plate exists for Lego (D-017); not meaningful for this PoC |

## Notes

Training-side half of a training/eval pair — see `phase3_poc_budget_00_eval.md`
for the actual evaluation run (checkpoint, view indices, command, PSNR
numbers) per this step's instruction to log evaluation as its own entry.
