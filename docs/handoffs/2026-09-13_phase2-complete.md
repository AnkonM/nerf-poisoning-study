# Handoff — 2026-09-13, Phase 2 complete

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
[environment/SETUP.md](../../environment/SETUP.md).

## Current phase + status

(As of this handoff, from `docs/ROADMAP.md`'s status tracker:)

| Phase | Status | Gate passed? |
|---|---|---|
| 0 — Setup | Complete (local) | PASS locally; Colab/Kaggle parity still unverified (deferred to Phase 6) |
| 1 — Literature verification | Complete | PASS — novelty claim reconfirmed (D-012) |
| 2 — Lego clean-NeRF sanity check | **Complete** | **PASS** — clean-Lego test PSNR 31.550 dB, within the D-013 range (29–33 dB) (D-016) |
| 3 — Poisoning proof of concept (Lego) | Not started | — |
| 4–10 | Not started | — |

## Key locked decisions relevant to picking up work right now

- **D-002** — poisoning budget is defined as % of *target-visible* views, not all views. Don't redefine this.
- **D-005** — vanilla NeRF (vendored in `src/nerf/`, from `yenchenlin/nerf-pytorch`, MIT), pure PyTorch, no custom CUDA extensions. Nerfacto is Phase 9 only.
- **D-009/D-010** — `uv` (`pyproject.toml`/`uv.lock`) is the single dependency source of truth; `environment/requirements.txt` is machine-generated from it, never hand-edited.
- **D-013** — the clean-Lego PSNR acceptance range (29–33 dB) was fixed *before* training and used as-is; don't move goalposts on future gate checks either.
- **D-016** — Phase 2's actual result and the VRAM-contention incident during its final eval (see "Known past issues" below).

## Known past issues/bugs already fixed

- **D-011 / D-014** — a literal, unexpanded shell brace-glob directory (e.g. `src/{nerf,poisoning,...}` as one literal folder name) appeared twice in the original repo scaffold, under both `src/` and `data/`. Both fixed; if you ever see a folder name containing literal `{` `}` `,` characters, that's this bug pattern again.
- **D-015** — the top-level `.gitignore` had 3 leading spaces baked into every line, silently making every ignore rule a no-op (confirmed via `git check-ignore -v`). Fixed; git history audited afterward and confirmed clean (no large/generated files were ever actually committed while it was broken).
- **D-016** — the vendored training script's automatic post-training full-test-set render has no per-image progress logging by default, which combined with the process sitting near the 8GB VRAM cap made a genuinely-still-running (just very slow) process look identical to a hang for over 5 hours. Diagnosed via `/proc/<pid>/stat` CPU-tick deltas (not guessed), fixed by adding `show_progress` logging to `evaluate_psnr` (`src/nerf/training.py`) and building the standalone `scripts/evaluate.py` (checkpoint-only re-evaluation, no retraining needed). If a future training run's post-training phase looks stalled, check CPU ticks and GPU util/power before assuming it's hung, and consider running `scripts/evaluate.py` against the last checkpoint instead of waiting.

## Immediate next steps

Per `docs/ROADMAP.md` Phase 3 — Poisoning Proof of Concept (still on Lego):

- Implement hard-erasure compositing (`src/poisoning/compositor.py`) against Lego's own background/mask (Lego has no natural background plate — a rough proxy is fine here, this phase is about pipeline mechanics, not measurement).
- Run 2–3 rough budget levels (e.g. 0/20/50% of Lego's own visible views) purely to see the poison-then-retrain mechanism visibly work.
- **Gate:** visibly degraded/suppressed reconstruction at the higher budget level, no pipeline errors (shape mismatches, pose misalignment, mask misregistration).
- **Explicit non-goal:** no scientific conclusions from these numbers — this phase exists to kill bugs before Phase 4 builds the real scene.

---

**Before doing any work, read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is a
map, not a substitute.**
