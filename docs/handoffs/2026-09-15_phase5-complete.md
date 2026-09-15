# Handoff — 2026-09-15, Phase 5 complete

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

An empirical security study measuring how the fraction of poisoned training
views affects selective target-object suppression in a NeRF, and the
collateral damage that poisoning causes to the rest of the scene — see
[README.md](../../README.md).

## Hardware / environment summary

Windows 11 laptop, RTX 5060 (Blackwell, sm_120), WSL2 Ubuntu, `uv` for Python
(D-009), Blender 5.2.1 LTS native on Windows (D-008). Setup:
[environment/SETUP.md](../../environment/SETUP.md). **`pytest` was added as a
dev dependency in Phase 5** — run the suite with `python -m pytest tests/`.

## Current phase + status

From `docs/ROADMAP.md`'s status tracker at the time of writing:

| Phase | Status | Gate passed? |
|---|---|---|
| 0–3 | Complete | PASS (D-009–D-020) |
| 4 — Final scene + eval set | Complete | PASS — `\|V_target\|` = 100/100; eval set frozen, aggregate SHA-256 `211a5a59…5ab214`; `METHODOLOGY.md` FROZEN 2026-09-14 (D-021–D-027) |
| 5 — Poisoning pipeline build | **Complete** | **PASS** — 20 datasets, manifest counts exact, outside-mask strict byte identity (425 images, 0 deviations), 29 tests pass, D-020 closed (D-028–D-031) |
| 6 — Minimum poisoning-budget sweep | **Next** | — |

## What Phase 5 produced

- **20 poisoned datasets** in `data/poisoned/` (1.3 GB), not 8: `METHODOLOGY.md`
  §8 ties random view selection to the same seed as model init, so C1–C6 get one
  dataset per seed (`C3_seed2`, …) while `control` and `C7` are seed-invariant
  and have a single directory each (D-028).
- **One compositor** — `composite(original, mask, plate, alpha)` implements §3's
  general formula; hard erasure is its `alpha=0` case, verified bit-identical to
  the Phase 3 implementation over 10 cases (D-029).
- **Soft-suppression `alpha` = 0.4**, locked, in
  `configs/poisoning/soft_suppression_20.yaml` (D-029).
- **Eight condition configs** per `PROJECT_STRUCTURE.md`, each with
  `seeds: [1,2,3]`.
- **`tests/`** — 29 pytest tests, previously an empty directory.

## Key locked decisions relevant to picking up work right now

Everything from Phases 0–4 still applies — especially **D-002** (budget is % of
target-visible views), **D-022 Finding 4** (shadows survive poisoning by design;
do not "fix" it), **D-024** (3 px dilation), **D-026** (`METHODOLOGY.md` is
FROZEN — changing §1–§7 needs a DECISION_LOG entry *and* a §10 deviation line).
New in Phase 5:

- **D-028** — per-split loader paths; the `(-area, view_index)` strategic
  tie-break; and the 20-dataset seed policy.
- **D-029** — unified compositor, `alpha` = 0.4, and the loader's new
  `camera_angle_x`-agreement check.
- **D-030** — gate PASS and D-020's closure.
- **D-031** — `requirements.txt` header debt (see below).

## Known issues / debt

- **D-031 — `requirements.txt`'s own regeneration command strips its own
  header.** The file carries a 12-line header ("DO NOT HAND-EDIT" plus the exact
  `uv export` command), but running that command deletes the header, because
  `uv export` emits only its own preamble and the pinned requirements. **Not
  fixed.** It is a documentation-consistency defect, not a correctness one — the
  pinned versions are always right. If you regenerate `requirements.txt`, check
  the header survived. Suggested fix: move the explanation into
  `environment/SETUP.md` and leave `requirements.txt` purely machine-generated.
- **The manifest writer now refuses to rewrite a manifest whose header does not
  match its schema** (D-030). If you see it exit with a schema mismatch, that is
  the guard working — do not delete the file to get past it; a foreign-schema
  manifest means two different experiments are sharing one file. Phase 3's PoC
  manifest lives separately in `data/poisoned/MANIFEST_phase3_poc.csv`.
- **12 Phase 3 symlinks remain** under `data/poisoned/phase3_poc_budget_*/`.
  Verified **inert** — no study config and no Phase 5/6 code path reads them
  (D-030). They are historical artifacts; leave them alone.
- Colab/Kaggle environment parity is **still unverified** (Phase 0 tracker) and
  Phase 6 is where it finally gets exercised.

## On the horizon — check these first if something looks wrong later

1. **near/far sampling span (D-023) — check this FIRST if Phase 6 convergence
   disappoints.** `near`/`far` are 1.5 / 9.0, measured from the real scene
   (depth range 1.806–8.346 m). That is an **8.1 m span versus Lego's 4.0 m**,
   so at the same 64 coarse / 128 fine samples the per-sample spacing is roughly
   **2x coarser than the Phase 2 config that produced the validated 31.55 dB
   baseline**. It does not affect any frozen data — near/far are config values,
   changeable until training starts. If the final scene trains visibly worse than
   Phase 2's Lego run, **do not assume the poisoning pipeline or the scene is at
   fault**: check sample-count-vs-depth-span first, and consider raising
   `render.num_coarse_samples`/`num_fine_samples`. Log any change as a new entry.
2. **Phase 6's iteration count is an open decision.** D-018's 30,000 was scoped
   to the Lego PoC and sets **no precedent**; the final scene needs its own
   convergence check against its own reference range, analogous to D-013.
3. **Shadow retention is expected** (D-022 Finding 4, `METHODOLOGY.md` §3's
   "Scope of the edit"). Poisoned views show the mug gone but its shadow present.
   Report it; do not engineer it away.

## Immediate next steps

Per `docs/ROADMAP.md` **Phase 6 — Minimum Poisoning-Budget Sweep**: train all 8
conditions × 3 seeds (24 runs minimum), each with an `experiments/logs/` entry
and a `results.csv` row, computing metrics **only** against the frozen eval set.

**Interface note you will need immediately.** The condition configs carry
`dataset.val_path` and `dataset.test_path` (both `data/blender_scenes`), but
**`dataset.path` is deliberately not set**, because it varies per seed. To
resolve a run's training data, map (condition, seed) → `dataset_id` via
`data/poisoned/MANIFEST.csv`'s `dataset_id` / `selection_seed` columns:

- `C1`–`C6` at seed *s* → `data/poisoned/<condition_id>_seed<s>/`
- `control` and `C7` are **seed-invariant** — one directory each
  (`data/poisoned/control/`, `data/poisoned/C7/`), used by all three training
  seeds, with only model initialisation differing between those runs.

Also relevant: `scripts/train.py --skip-final-eval` (D-019) exists precisely for
back-to-back runs like this sweep, and avoids a repeat of D-016's VRAM-contention
stall.

**Gate:** all runs complete without divergence; loss curves inspected per
condition; `results.csv` has one row per (condition × seed).

---

**Before doing any work, read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is a
map, not a substitute.**
