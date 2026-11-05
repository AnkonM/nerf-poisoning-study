# Handoff — 2026-11-05, Phase 8 §6 metrics built and computed (analysis pending)

> **This file is an INDEX/SUMMARY only.** It exists to give a fresh AI
> conversation (or a human) fast orientation — it is never the
> authoritative source for anything. The authoritative sources are always:
> **`README.md`, `docs/ROADMAP.md`, `docs/METHODOLOGY.md`, and
> `docs/DECISION_LOG.md`.**
>
> **Before doing any work, read `README.md`, `docs/ROADMAP.md`,
> `docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is
> a map, not a substitute.** If anything here conflicts with those files,
> those files win — this one may simply be stale.

## Project one-line summary

How much of a NeRF's training data must be poisoned to suppress a target object, and the collateral damage — see [README.md](../../README.md).

## Hardware / environment summary

Laptop (WSL2, RTX 5060 8 GB, ~2 GB free VRAM because Windows holds ~6 GB), `uv` env; sweep was trained on the HPC machine. See [environment/SETUP.md](../../environment/SETUP.md).

## Current phase + status

Phase 8 row in [docs/ROADMAP.md](../ROADMAP.md): metrics built and computed; plots and analysis pending. Written 2026-11-05.

## Where things are

- Definitions (pre-registered before the 24-run computation): `DECISION_LOG.md` **D-042**; one-line clarification in `METHODOLOGY.md` §10. Pre-registration commit: `284d141`.
- Code: `src/metrics/{eval_data,image_metrics,compute}.py`, `scripts/compute_metrics.py <run_id | --all>`; tests `tests/test_metrics.py` (full suite: 158 passed).
- Outputs (tracked, small): `experiments/results/metrics/per_run.csv` (24 rows), `per_view/<run_id>.csv` (75 rows each), `reference_gt_vs_plate.csv`, `metrics_meta.json`.
- LPIPS alex backbone is read from `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` (sha256 pinned in code; never downloaded by the metric code). A fresh machine needs that one-time download.
- Validation: 24 rows x 75 views; max whole-frame deviation vs each run's eval JSON 0.0062 dB (control_seed2); no NaN/inf; `control_seed3` flagged `resumed=1`; LPIPS batch 4, no OOM fallback.

## Key locked decisions

D-042 (metric definitions), D-024 (3 px dilation), D-022 Finding 4 (shadow retention outside the mask is expected), D-027 (never re-render the eval set), D-041 (renders; provenance commit `6be1b89`).

## Immediate next steps

Phase 8 analysis: core plots with mean +/- std over seeds, C3-vs-C7 comparison, qualitative figures. Any metric bug is fixed via a new D-043+ entry and recomputation.

---

**Before doing any work, read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is a
map, not a substitute.**
