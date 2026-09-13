# Experiment Log — phase3_poc_budget_00_eval

Evaluation run for `phase3_poc_budget_00` (see `phase3_poc_budget_00.md`
for the training run this evaluates). Logged as its own entry per this
step's instruction. Phase 3 PoC only — PSNR here is a PoC signal, not a
docs/METHODOLOGY.md §6 metric (that requires the frozen `eval_holdout/`
set, a Phase 4 deliverable).

## Identification

- **Run ID:** phase3_poc_budget_00_eval
- **Condition:** evaluation of phase3_poc_budget_00 (control, 0% budget)
- **Config file path:** `configs/poisoning/phase3_poc_budget_00.yaml`
- **Checkpoint evaluated:** `experiments/runs/phase3_poc_budget_00/030000.tar` (final, 30,000-iteration checkpoint)
- **Git commit hash:** `a2cc6298de1e0e2ede640df0db6a3102e556d8a1` (unchanged from the training run — evaluation ran against the same checkpoint/config, no code changes in between other than the scratch analysis script below)
- **Date/time:** 2026-09-13 22:40:31 – 22:42:56 (+02:00)

## Environment

- **Where run:** local (WSL2), RTX 5060 laptop, same environment as training.

## Data / views evaluated

- **Held-out view indices:** global concatenated-array indices 200, 266, 333, 399 (test-split views, same 4 indices used for all three conditions and matching the Phase 2 D-016 convention) — fixed, chosen before evaluating any condition, same 4 across all conditions for direct comparability.
- **Eval set used:** `data/nerf_synthetic/lego` test split (loaded via the run's own `dataset.path`, which for this condition points at `data/poisoned/phase3_poc_budget_00`, whose `test`/`val` are unmodified symlinks to `data/nerf_synthetic/lego`).

## Invocation

Not `scripts/evaluate.py` directly (that script's CLI evaluates the *full*
i_test split and returns one scalar — no way to select 4 specific indices
or compute a masked/target-region PSNR). Used a supplementary scratch
script, **`scripts/phase3_poc_eval_views.py`** (originally written as a
scratch file during this step, promoted to this durable repo path in the
follow-up audit-trail pass so it isn't lost — see `docs/DECISION_LOG.md`):

```
python scripts/phase3_poc_eval_views.py configs/poisoning/phase3_poc_budget_00.yaml \
    experiments/runs/phase3_poc_budget_00/030000.tar phase3_poc_budget_00
```

(This script shares its model-loading/rendering code path with
`src/nerf/training.py`'s `_build_model`/`_render_kwargs`/`render`, the same
functions `scripts/evaluate.py` uses — it does not reimplement rendering.)

## Metrics (PoC signal only)

| View | Full-frame PSNR | Masked (target-region) PSNR |
|---|---|---|
| 200 | 28.580 dB | 23.786 dB |
| 266 | 27.803 dB | 23.234 dB |
| 333 | 27.918 dB | 23.314 dB |
| 399 | 28.775 dB | 24.030 dB |
| **Mean** | **28.269 dB** | **23.591 dB** |

## Notes

An initial version of the scratch eval script had a bug — it thresholded
the loader's already-normalized ([0,1]) alpha channel against 127
(the raw-uint8-scale threshold from D-017), which matched zero pixels and
produced `masked_psnr = nan` for all views. Caught before recording any
numbers here; fixed to threshold at `127/255` in the normalized scale and
re-ran. No pipeline code (`src/`, `scripts/`) was affected — this bug was
confined to the ad hoc analysis script.
