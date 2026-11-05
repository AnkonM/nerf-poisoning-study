# Handoff — 2026-11-05, Phase 8 analysis done (tables, figures, neutral draft); review pending

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

Laptop (WSL2, RTX 5060 8 GB), `uv` env; sweep trained on the HPC machine. See [environment/SETUP.md](../../environment/SETUP.md).

## Current phase + status

Phase 7/8 rows in [docs/ROADMAP.md](../ROADMAP.md) (updated 2026-11-05). Analysis only; nothing retrained, re-rendered or recomputed.

## Where things are

- Plan of record: `DECISION_LOG.md` **D-043** (commit `c678016`, before any aggregate); metrics definitions D-042 (`284d141`).
- Code: `src/metrics/{aggregate,figures}.py`, `scripts/analyze_phase8.py` (commit `d987c9a`); tests `tests/test_aggregate.py` (suite: 168 passed).
- Outputs (tracked): `experiments/results/phase8/` (condition_summary, pair_comparisons, E1-E5 CSVs); `paper/figures/` (figs 1-5, 6a/6b, 7, 8a-c as PNG+PDF+sidecar JSON, each recording the per_run.csv sha256 `ba10c206…ac40`).
- Draft: `docs/RESULTS_DRAFT_phase8.md` (descriptive; claims table, limitations, open questions). Regenerate everything with `python scripts/analyze_phase8.py`.

## Key locked decisions

D-042, D-043, D-024, D-022 Finding 4 (shadow retention expected), D-027, D-034 (no comparison with published PSNR), D-041.

## Immediate next steps

Project-lead review of the draft and its open questions (section 5); loss-curve sign-off still pending (D-041); Phase 9 only if warranted; Phase 10 write-up. Any metric or analysis change needs a new D-044+ entry.

---

**Before doing any work, read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is a
map, not a substitute.**
