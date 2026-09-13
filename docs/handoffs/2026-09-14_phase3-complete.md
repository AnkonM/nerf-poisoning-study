# Handoff — 2026-09-14, Phase 3 complete

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

An empirical security study measuring how the fraction of poisoned
training views affects selective target-object suppression in a NeRF,
and the collateral damage that poisoning causes to the rest of the scene
— see [README.md](../../README.md).

## Hardware / environment summary

Local development on a Windows 11 laptop with an RTX 5060 laptop GPU
(Blackwell, sm_120), via WSL2 (Ubuntu), Python dependency management via
`uv` (not conda — see D-009), Blender installed natively on Windows (not
WSL2 — see D-008). Full setup/verification steps:
[environment/SETUP.md](../../environment/SETUP.md). `matplotlib` was
added as a real dependency during Phase 3 closeout (for TensorBoard
curve export); `pyproject.toml`/`uv.lock`/`environment/requirements.txt`
are all in sync as of this handoff.

## Current phase + status

(As of this handoff, from `docs/ROADMAP.md`'s status tracker:)

| Phase | Status | Gate passed? |
|---|---|---|
| 0 — Setup | Complete (local) | PASS locally; Colab/Kaggle parity still unverified (deferred to Phase 6) |
| 1 — Literature verification | Complete | PASS — novelty claim reconfirmed (D-012) |
| 2 — Lego clean-NeRF sanity check | Complete | PASS — clean-Lego test PSNR 31.550 dB, within the D-013 range (29–33 dB) (D-016) |
| 3 — Poisoning proof of concept (Lego) | **Complete** | **PASS** — monotonic PSNR degradation with poisoning budget (masked PSNR 23.6→19.4→11.6 dB at 0/20/50% budget), zero pipeline errors across all checks (D-017–D-020) |
| 4–10 | Not started | — |

## Key locked decisions relevant to picking up work right now

- **D-002** — poisoning budget is defined as % of *target-visible* views, not all views. Don't redefine this.
- **D-005** — vanilla NeRF (vendored in `src/nerf/`, from `yenchenlin/nerf-pytorch`, MIT), pure PyTorch, no custom CUDA extensions. Nerfacto is Phase 9 only.
- **D-009/D-010** — `uv` (`pyproject.toml`/`uv.lock`) is the single dependency source of truth; `environment/requirements.txt` is machine-generated from it, never hand-edited.
- **D-013** — the clean-Lego PSNR acceptance range (29–33 dB) was fixed *before* training and used as-is; don't move goalposts on future gate checks either.
- **D-016** — Phase 2's actual result and the VRAM-contention incident during its final eval (see "Known past issues" below).
- **D-017** — Phase 3's Lego mask/background-proxy/`V_target` protocol (alpha-channel mask thresholded at 127/255, fixed flat mid-gray `(128,128,128)` background proxy, `V_target` = all 100 train views). **Explicitly scoped to the Phase 3 Lego PoC only — this is NOT the real scene's mask/background-plate protocol.** Phase 4 must use real Blender-rendered object-ID masks and `background_plate` renders per `METHODOLOGY.md` §3, not this proxy.
- **D-018** — Phase 3 trained all 3 PoC conditions for 30,000 iterations (not full 200k). This number is scoped to this PoC only and **sets no precedent for Phase 6's core-sweep iteration count**, which is a separate, not-yet-made decision requiring its own convergence check against the final scene.
- **D-019** — `skip_final_test_eval` (`src/nerf/training.py`) / `--skip-final-eval` (`scripts/train.py`) added: opt-in flag to skip the automatic full-test-set render at the end of `train_from_config`, avoiding a repeat of D-016's VRAM-contention stall across back-to-back runs. **Default behavior is unchanged** (flag unset preserves the original unconditional eval). This is shared infrastructure — Phase 6's core sweep should use it too.
- **D-020** — **known debt:** `scripts/build_poison_set.py` currently only emits `train/` + `transforms_train.json`; Phase 3's `data/poisoned/phase3_poc_budget_*/val`, `.../test` (and their `transforms_*.json`) are manually-created symlinks to `data/nerf_synthetic/lego`'s originals, not script-generated. This violates `PROJECT_STRUCTURE.md`'s "fully reproducible from config alone" rule and **must be fixed in Phase 5** before the real 8-condition sweep relies on `build_poison_set.py`.

## Known past issues/bugs already fixed

- **D-011 / D-014** — a literal, unexpanded shell brace-glob directory (e.g. `src/{nerf,poisoning,...}` as one literal folder name) appeared twice in the original repo scaffold, under both `src/` and `data/`. Both fixed; if you ever see a folder name containing literal `{` `}` `,` characters, that's this bug pattern again.
- **D-015** — the top-level `.gitignore` had 3 leading spaces baked into every line, silently making every ignore rule a no-op (confirmed via `git check-ignore -v`). Fixed; git history audited afterward and confirmed clean (no large/generated files were ever actually committed while it was broken).
- **D-016** — the vendored training script's automatic post-training full-test-set render has no per-image progress logging by default, which combined with the process sitting near the 8GB VRAM cap made a genuinely-still-running (just very slow) process look identical to a hang for over 5 hours. Diagnosed via `/proc/<pid>/stat` CPU-tick deltas (not guessed), fixed by adding `show_progress` logging to `evaluate_psnr` (`src/nerf/training.py`) and building the standalone `scripts/evaluate.py` (checkpoint-only re-evaluation, no retraining needed). If a future training run's post-training phase looks stalled, check CPU ticks and GPU util/power before assuming it's hung, and consider running `scripts/evaluate.py` against the last checkpoint instead of waiting.
- **Phase 3 closeout — divergence-check false positive (statistical-comparison bug pattern, not a data bug):** a first-pass "did this run diverge" check compared each run's tail-mean training loss against the single lowest-ever-seen loss sample and flagged both poisoned conditions as "diverging." The single global minimum in a per-batch-random-image training loop is an outlier lucky batch (e.g. an easy, mostly-flat erased region), not a meaningful convergence baseline — comparing against it will false-positive almost any noisy-but-healthy run. Caught and fixed *before* any number was reported, by comparing first-half-vs-second-half mean loss instead (robust to single-batch noise). **Watch for this same class of mistake in Phase 6/8 analysis code**: any "is X anomalous" check built on a single extreme sample (min/max) rather than a distributional or windowed comparison is prone to exactly this false-positive pattern, especially with small per-condition sample sizes (Phase 6's N=3 seeds is squarely in this risk zone for downstream statistical checks).

## Immediate next steps

Per `docs/ROADMAP.md` Phase 4 — Final Scene + Frozen Evaluation Set:

- Build the custom multi-object Blender scene with a clear, distinct target object embedded in a larger environment (D-003 — custom scene over adapted real dataset).
- Render the full training-view set with camera poses.
- For every training view: render the matching `background_plate` (target toggled off) and `mask` (object-ID pass) — `METHODOLOGY.md` §3. **This replaces D-017's Lego alpha-channel/flat-color proxy — use real rendered plates/masks, not that PoC shortcut.**
- Render the held-out evaluation-view set (disjoint camera poses).
- Compute and record `|V_target|` (count of training views where the target mask exceeds the minimum-visibility threshold, `METHODOLOGY.md` §2) — gates every later poisoning-budget calculation.
- **Freeze** `data/blender_scenes/eval_holdout/` — set read-only, checksum it, record the checksum in `DECISION_LOG.md`.
- Freeze `METHODOLOGY.md` (remove the "status: not frozen" note, add a frozen-date entry to `DECISION_LOG.md`).
- **Gate:** scene has a clearly identifiable target object separable from background by mask; `|V_target|` is large enough that even the 5% condition corresponds to ≥1 whole poisoned view; eval set is frozen and checksummed.

**Before Phase 4's scene-building work gets far, resolve D-020's debt** (or explicitly re-schedule it, in writing, to a specific point before Phase 5/6 need it): extend `scripts/build_poison_set.py` to emit a fully self-contained, config-reproducible dataset directory (val/test included) instead of relying on manually-created symlinks, so the real 8-condition sweep's poisoned sets satisfy `PROJECT_STRUCTURE.md`'s reproducibility rule from the start.

---

**Before doing any work, read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is a
map, not a substitute.**
