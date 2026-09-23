# Handoff — 2026-09-16 (report written 2026-09-23), Phase 6 IN PROGRESS

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
>
> **This is the first "in progress" handoff in the project.** Every prior
> handoff marked a completed, gated phase. This one does not — Phase 6 is
> mid-flight, and the biggest open item (where the sweep actually runs) is
> explicitly NOT decided here. Do not read the existence of this file as
> permission to assume that decision has been made.

## Project one-line summary

An empirical security study measuring how the fraction of poisoned training
views affects selective target-object suppression in a NeRF, and the
collateral damage that poisoning causes to the rest of the scene — see
[README.md](../../README.md).

## Hardware / environment summary

Windows 11 laptop, RTX 5060 (Blackwell, sm_120), WSL2 Ubuntu, `uv` for Python
(D-009), Blender 5.2.1 LTS native on Windows (D-008). Setup:
[environment/SETUP.md](../../environment/SETUP.md). Colab/Kaggle parity is
**still unverified** (Phase 0 tracker) — see "Open, unresolved" below, this
is now directly relevant rather than a deferred formality.

## Current phase + status

From `docs/ROADMAP.md`'s status tracker at the time of writing:

| Phase | Status | Gate passed? |
|---|---|---|
| 0–5 | Complete | PASS (D-009–D-031) |
| 6 — Minimum poisoning-budget sweep | **In progress** — Step 0/1 done, sweep not started, compute platform undecided | Step 0 (investigation) and Step 1 (local control convergence probe) complete. Step 2 (sweep harness) not started. See `DECISION_LOG.md` D-032–D-035. |

## Locked decisions (Phase 6 Step 0/1)

- **D-032** — four latent config defects found and fixed before any sweep
  compute was spent: every condition config silently resolved to the
  *clean* dataset (fixed with a content-based assertion,
  `src/utils/dataset_id.py`); the final-scene config had no
  `training:`/`optimizer:` block at all (crash + 4x batch size, fixed to
  match Phase 2's validated hyperparameters); a YAML duplicate key silently
  deleted the Blender render section mid-edit (fixed the occurrence, closed
  the class with a duplicate-key-raises config loader); auto-resume was
  silently unsafe (fixed: `train.py`/`evaluate.py` now refuse a populated
  run directory without an explicit `--resume`).
- **D-033** — **`training.iterations = 200,000`** for the sweep, locked
  from real full-eval-set-PSNR evidence across 12 checkpoints (25k–200k),
  not the noisy 5-view TensorBoard proxy that misled the first two reads of
  this curve. `optimizer.lr_decay_steps = 500,000` and `training.batch_size
  = 1024` were already correct from D-032 and are confirmed, not re-decided,
  by this entry. **Seed-major run ordering** (all 8 conditions at seed 1,
  then seed 2, then seed 3) was decided earlier in planning and is
  unaffected by this investigation — reasoning: degrades gracefully if
  compute runs short, and surfaces condition-specific bugs in the first 8
  runs rather than at run 16+.
- **D-034** — this scene's control PSNR (**50.778 dB**) is **not comparable**
  to Lego's (31.550 dB, D-016) and must never be reported side-by-side with
  it. Binding for Phase 8/10: report relative degradation from this scene's
  own control, never absolute PSNR against published NeRF numbers.
- **D-035** — `evaluate.py` had the same dataset-resolution defect D-032
  fixed in `train.py`; already committed (`7d76c0d`) and verified working —
  it produced every number D-033 relies on.

## OPEN, UNRESOLVED — read this before Step 2

**Where the 24-run sweep executes is not decided.** Local (this laptop),
a rented single-GPU cloud instance, and a university HPC resource are all
still on the table. This is explicitly **not** assumed or defaulted by this
handoff — resolve it before Step 2 (sweep harness) begins, since the harness
design (checkpoint sync, environment parity checks, job submission) differs
materially by platform.

**If local is chosen:** projected cost at 200,000 iterations, measured local
rate (~5.9–6.5 it/s): **~226 GPU-hours (~9.4 days) for 24 runs.** Real, not
estimated from a shorter probe.

**If cloud/HPC is chosen: the data-transfer requirement below is mandatory,
not optional**, and Colab/Kaggle's cu128-floor-vs-cu130-pin gap (flagged in
D-032, unresolved) needs to be checked for whichever platform is actually
picked.

## CRITICAL — data must be transferred as literal files, never regenerated

D-027 established that re-rendering this scene is **not bit-exact**
(Cycles+OptiX GPU path tracing is not bit-deterministic across runs — a fixed
`cycles.seed` fixes the sample pattern, not the summation order). This means:
if the sweep moves to a different machine, `data/blender_scenes/`,
`data/masks/`, `data/background_plates/` and `data/poisoned/` **must be
copied as literal files** (tar/zip + scp, cloud storage, or physical media) —
**never regenerated from `scripts/build_scene.py` + `scripts/build_poison_set.py`
on the new machine and assumed equivalent.** A regenerated copy would not
match the frozen `eval_holdout`'s checksum, and Phase 6's results would not
be reproducible from a config + commit hash the way `PROJECT_STRUCTURE.md`
requires.

**What's git-tracked vs. what must be transferred manually — verified, not
assumed:**

| path | git-tracked | must transfer manually |
|---|---|---|
| `data/blender_scenes/cameras.json`, `transforms_{train,val,test}.json`, `eval_holdout_SHA256SUMS.txt` | yes | no |
| `data/blender_scenes/{train,val,eval_holdout}/*.png` | **no** | **yes** |
| `data/masks/`, `data/background_plates/` | **no** | **yes** |
| `data/poisoned/*/train/*.png` | **no** | **yes** |
| `data/poisoned/MANIFEST.csv` | **no — gitignored** | **yes, and easy to forget** |

`MANIFEST.csv` is the one most likely to be missed: it is not code, it looks
like it should be tracked, and `src/utils/dataset_id.py::resolve_dataset`
depends on it entirely for the (condition, seed) → dataset mapping. A clone
without it will fail every dataset resolution, loudly (by design, per D-032)
— but only after someone has already set up the new machine expecting it to
work.

**Total size to transfer:** measured directly, not estimated —

| directory | size |
|---|---|
| `data/blender_scenes/train` | 63 MB |
| `data/blender_scenes/val` | 6.3 MB |
| `data/blender_scenes/eval_holdout` | 47 MB |
| `data/masks` | 1.4 MB |
| `data/background_plates` | 107 MB |
| `data/poisoned` | 1.3 GB |
| **total** | **~1.5 GB** |

**Verification after transfer is mandatory, and confirmed working right
now:** `python scripts/freeze_eval_set.py --verify` recomputes every file's
SHA-256 against the frozen manifest and reports PASS/FAIL plus the aggregate
checksum. Run this immediately after any transfer, before any training
starts on the new machine — it will catch truncation, encoding corruption
(e.g. an FTP client mangling binary PNGs), or a missed subdirectory. Verified
just now on the current machine: `VERIFY: PASS — byte-for-byte identical`,
aggregate `211a5a59d85cf29447a7608e74bbe49ebe08717709d3c520bc3af549d65ab214`
(unchanged since the D-026 freeze).

No packaging or transfer was performed by this session — sizes and the
verification procedure are reported only, per instruction.

## Interface note carried forward from Phase 5 (still applies)

`dataset.path` is `null` on every condition config (D-032); resolve a run's
actual training data via `src/utils/dataset_id.py::dataset_id_for` /
`resolve_dataset`, backed by `data/poisoned/MANIFEST.csv`'s `dataset_id` /
`selection_seed` columns:

- `C1`–`C6` at seed *s* → `data/poisoned/<condition_id>_seed<s>/`
- `control` and `C7` are seed-invariant — one directory each, used by all
  three training seeds, with only model initialisation differing.

`resolve_dataset` VERIFIES this by content (byte-diffing against the clean
originals, not just checking the path resolves) before training or
evaluating — see D-032 for why a path-only check was judged insufficient.

## Known past issues/bugs already fixed (Phase 6, don't rediscover these)

- **D-032** — silent clean-dataset resolution, missing training block,
  YAML duplicate-key data loss, unsafe auto-resume. All fixed; see above.
- **D-035** — same dataset-resolution defect, second call site
  (`evaluate.py`). Fixed.
- One real interruption occurred during Step 1: the WSL2 VM was torn down
  mid-run (not an application crash — no OOM, no traceback, just the host
  disappearing), losing progress back to the last checkpoint. Recovered via
  `--resume`. Not a bug in this project's code; a reminder that anything
  run locally over many hours is exposed to host sleep/reboot, which is
  part of why the compute-platform decision above matters.

## Immediate next steps

1. **Resolve the compute-platform decision** (local / cloud / HPC) — see
   "Open, unresolved" above. Not this session's call.
2. Once resolved, **Step 2 — build the sweep harness**, per the approved
   Phase 6 plan: seed-major ordering, the (condition, seed) dataset
   assertion (already implemented in `src/utils/dataset_id.py`, reuse it —
   do not reimplement), `--skip-final-eval` always passed with evaluation
   as a separate process (`scripts/evaluate.py`, D-016/D-019), and the
   `results.csv` schema specified in the approved plan (one row per
   condition × seed, tracing to a config path + commit hash + log entry per
   `PROJECT_STRUCTURE.md`).
3. If the platform choice is cloud/HPC: verify environment parity
   (D-009's cu128-floor-vs-cu130-pin gap) and the data-transfer procedure
   above **before** Step 3 (running the sweep), not during it.

---

**Before doing any work, read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is a
map, not a substitute.**
