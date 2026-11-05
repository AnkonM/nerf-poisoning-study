# Handoff — 2026-11-05, laptop catch-up after the Phase 6 sweep (HPC -> laptop)

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

How much of a NeRF's training data must be poisoned to suppress a target
object, and what collateral damage results — see [README.md](../../README.md).

## Hardware / environment summary

Laptop: Windows 11 + WSL2 Ubuntu, RTX 5060 8 GB (capability (12, 0)), `uv`
env. The 24 Phase 6 runs were trained on the HPC machine (Quadro RTX 5000).
Setup: [environment/SETUP.md](../../environment/SETUP.md). Note: ~6 GB of the
laptop GPU's 8 GB was in use by non-WSL processes when checked (see below).

## Current phase + status

Written 2026-11-05. Authoritative status is the tracker in
[docs/ROADMAP.md](../ROADMAP.md) (Phase 6 row set by D-041). On this laptop:
all 24 sweep runs and their eval-view renders are present and
hash-verified; no §6 metric code exists yet (Phase 8 not started).

## What was synced (this session)

- **Git:** local master had 1 unpushed commit (`d5900a2` "docs: IA-1
  evidence") and had diverged from the HPC bundle's master. Backup branch
  `backup/pre-sync-master` -> `d5900a2`. No path overlap between `d5900a2`
  and the 17 HPC commits. Rebased: **`d5900a2` is now `cd8dbac`**; all 17 HPC
  commits (c5d6a47..5b6c6f0) kept their hashes and are ancestors of HEAD.
  `git diff --stat` vs HPC master touches only `docs/ia_evidence/ia1/`.
  Nothing pushed. Local master is "ahead 18" of origin/master.
- **Local-only excludes** (`.git/info/exclude`, not committed): `ia1.zip`,
  `ia1/IA1_RUN.zip`, `ia1/IA1_RUN (2).zip`.
- **Transfer:** `unzip -t` clean on all 3 zips. Extracted to
  `~/staging_transfer/`, merged into the repo with no-clobber rsync.
  62 staged paths already existed here (Phase 2/3/probe runs); all 62
  byte-identical, none overwritten. `experiments/runs/.sweep.lock` (HPC
  lock file, content `2166`, not in the manifest) was deliberately NOT
  copied. `~/staging_transfer/` still exists (4.7 GB); delete when happy.
- **sha256:** `sha256sum -c TRANSFER_MANIFEST.sha256` from repo root: 2296/2296
  OK, no non-OK lines.
- Temp ref `refs/tmp/hpc-master` deleted after verification.

## Verification results

- `uv sync --frozen` OK; `verify_env.py`: CUDA True, (12, 0), matmul PASSED.
- `pytest tests/`: **141 passed**, 0 skipped. HPC's D-040 figure was 130; the
  +11 are the closeout tests (`test_render_eval.py`, `test_sweep_summary.py`,
  `test_transfer_pack.py` = 11 tests, run separately).
- `freeze_eval_set.py --verify`: 226 files, 0 missing/unexpected/changed,
  aggregate `211a5a59…5ab214`, PASS.
- Permissions: 0 writable files among the 225 eval files; `transforms_test.json`
  is `-r--r--r--`. No chmod needed.
- `run_sweep.py status`: **24 COMPLETE**, harness not running, no hardware pin
  needed, started nothing.

## Phase 8 readiness inventory (read-only)

| item | state |
|---|---|
| results.csv | 24 rows, 8 conditions x 3 seeds, no duplicate run_ids |
| summary.json / eval JSON / 200000.tar, all 24 | present |
| final-checkpoint sha vs `phase6_final_checkpoints.csv` | 24/24 match (also match `render_meta.checkpoint_sha256`) |
| resumed runs | only `control_seed3` (interrupted at ~160.5k, resumed from 150k, rng_restored=true); retries: none anywhere |
| renders, 24 runs | all have `renders_fp16.npz`, 75 PNGs, `render_meta.json`; all record eval manifest sha `3e7d4ba9…253f63c6` and aggregate `211a5a59…`, equal to this laptop's frozen set |
| render validation | all 24 `ok`; max fp16 deviation vs harness PSNR 0.0062 dB (control_seed2), fp32 deviation 0.0; tolerance 0.01 |
| clean GT / masks / plates for 75 eval views | 75 / 75 / 75 present (800x800; masks uint16 {0,65535}) |
| `lpips` 0.1.4, `scikit-image` 0.26.0, torchvision | import OK |
| LPIPS linear weights (alex/vgg/squeeze) | shipped inside the `lpips` package |
| LPIPS backbone (torchvision alexnet/vgg16) | NOT cached (`~/.cache/torch/hub` absent); needs one-time download (alexnet ~244 MB from download.pytorch.org, reachable: HTTP 200). Not downloaded. |
| §6 metrics in code | none: `src/metrics/` holds only `.gitkeep`; `evaluate.py` computes mean whole-frame PSNR only |
| Phase 7 | C3 seeds 1-3 and C7 seeds 1-3 all COMPLETE (20% budget, hard erasure, random vs strategic) |
| disk | WSL volume 658 GB free; runs 3.0 GB, renders 1.8 GB, data 1.8 GB |

## IA branch inventory (report only; nothing tagged or deleted)

- Branches: `ia1` and `origin/ia1` at `d5900a2` (2026-10-23); `origin/ia2-final`
  at `15fe450` (2026-10-28). Existing tag: `ia1-snapshot` -> `efa8745`
  (2026-09-13, already an ancestor of master).
- Unique commits vs HPC master: `ia1`: `d5900a2` only (content now on master as
  `cd8dbac`). `ia2-final`: `15fe450` only (137 files, all under
  `docs/ia_evidence/ia2/`) — category (a), submission evidence. No category (b)
  project work on any IA branch. `docs/ia_evidence/ia2/` is not on this
  laptop's master.
- Proposed (NOT run): see final report in the session; tags
  `archive/ia1-2026-10` -> `d5900a2`, `archive/ia2-final-2026-11` -> `15fe450`.

## Open questions for the project lead

1. **Evaluation mask: dilated (3 px) or raw?** METHODOLOGY §3 defines dilation
   only for the *poisoning* mask ("raw Blender object-ID mask, one fixed
   dilation amount… 3 px"). §6 says only "target-object region" / "target
   region excluded" and does not say which mask. The stored eval masks are
   raw; dilation is applied only inside `scripts/build_poison_set.py`.
   Undecided here.
2. **Resolution.** Frozen GT, masks and plates are 800x800; renders are
   400x400 (`half_res`, and `load_blender_data` uses `cv2.resize(INTER_AREA)`).
   Metrics need GT/plates at 400x400 the same way, and a convention for
   resizing a binary mask (threshold after INTER_AREA? nearest?). Undecided.
3. **LPIPS backbone** (alex vs vgg) is not stated in the docs I grepped; the
   one-time weight download needs your OK.
4. **GPU headroom.** `nvidia-smi` showed ~6.0 of 8.15 GiB used with no WSL
   process using it (likely Windows-side). LPIPS-alex on 400x400, batch 1-4,
   is well under 2 GiB, so it fits; vgg is heavier but should still fit at
   batch 1. SSIM can run on CPU (scikit-image).
5. The render files were generated at commit `6be1b89` with 4 untracked
   closeout files (recorded in `render_meta.json`); harmless but noted.
6. Three IA zips remain untracked (excluded locally); `~/staging_transfer/`
   can be removed.

## Key locked decisions relevant right now

D-024 (3 px mask dilation, poisoning), D-027 (eval set never re-rendered),
D-026 (frozen-file permissions), D-032 to D-035 (sweep locks), D-040
(metrics gap), D-041 (Phase 6 closeout).

## Immediate next steps

Resolve open questions 1-3, then Phase 8 per `docs/ROADMAP.md`: build the §6
metrics from `renders_fp16.npz` against the frozen GT/masks/plates.

---

**Before doing any work, read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is a
map, not a substitute.**
