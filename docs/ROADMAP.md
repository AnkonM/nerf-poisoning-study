# Roadmap

Each phase has an **objective**, **tasks**, a **gate** (the quantitative or
concrete condition that must be true before moving on — this is what makes
progression data-driven rather than vibes-driven), **deliverables**, a rough
**effort/compute estimate**, and **risks**. Do not start a phase's tasks
before the previous phase's gate has actually been checked and passed —
that's the whole point of having gates.

## Status tracker

| Phase | Status | Gate passed? |
|---|---|---|
| 0 — Setup | Complete (local) | PASS locally — RTX 5060/WSL2 matrix-multiply + compute-capability (12, 0) check passes (`scripts/verify_env.py`). **Colab/Kaggle environment parity is unverified** and deferred until the first cloud run (expected Phase 6) — see `environment/SETUP.md` §2.1 and `DECISION_LOG.md` D-009. |
| 1 — Literature verification | Complete | PASS — novelty claim reconfirmed, no `METHODOLOGY.md` deviation needed. See `DECISION_LOG.md` D-012. |
| 2 — Lego clean-NeRF sanity check | Complete | PASS — clean-Lego test PSNR 31.550 dB, within the D-013 range (29–33 dB). See `DECISION_LOG.md` D-016 and `experiments/logs/phase2_lego_sanity.md`. |
| 3 — Poisoning proof of concept (Lego) | Complete | PASS — monotonic PSNR degradation with poisoning budget (masked PSNR 23.6→19.4→11.6 dB at 0/20/50% budget), zero pipeline errors across all checks. See `DECISION_LOG.md` D-017–D-020. |
| 4 — Final scene + eval set | Complete | PASS — `\|V_target\|` = 100/100 training views (mask area 1.47–2.18% of frame vs the 0.5% threshold locked before measurement; smallest margin 2.95x), so 5% → 5 poisoned views; eval set frozen read-only, 226 files, aggregate SHA-256 `211a5a59…5ab214`; `METHODOLOGY.md` frozen 2026-09-14. See `DECISION_LOG.md` D-021–D-026. |
| 5 — Poisoning pipeline build | Complete | PASS — manifest counts exact for all 20 datasets (0/5/10/20/30/50 = `round(b/100 × 100)`); outside-mask strict byte identity across 425 poisoned images, 0 deviations; 1,575/1,575 unpoisoned views byte-identical; 29 pytest tests pass; D-020 closed with byte-identical regeneration proven. See `DECISION_LOG.md` D-028–D-031. |
| 6 — Minimum poisoning-budget sweep | Not started | — |
| 7 — Ablation (random vs strategic) | Not started | — |
| 8 — Evaluation & analysis | Not started | — |
| 9 — Optional strengthening (conditional) | Not started | — |
| 10 — Write-up | Not started | — |

*(Update this table as phases close. This is the single source of truth for
"where are we", not a memory of what you did last week.)*

---

## Phase 0 — Environment & Tooling Setup

**Objective:** a working, reproducible local + cloud training environment
before any research code is written, so environment bugs don't get confused
with research findings later.

**Tasks:**
- [ ] Set up WSL2 (Ubuntu 24.04) on the Windows 11 laptop.
- [ ] Install PyTorch with explicit CUDA 12.8+ (`cu128`) wheels — verify the
  Blackwell (sm_120) RTX 5060 laptop GPU is actually used for compute, not
  silently falling back to CPU or an unsupported kernel path (see
  `environment/SETUP.md` for the verification script).
- [ ] Initialize git repo; adopt the structure in `docs/PROJECT_STRUCTURE.md`.
- [ ] Set up `configs/`, `pyproject.toml`/`uv.lock` (local) and
  `requirements.txt` (generated from the lockfile, for Colab/Kaggle), and
  confirm the exact same environment spec installs cleanly on Colab and
  Kaggle (this is what makes the local/cloud split in `environment/SETUP.md`
  actually usable, not aspirational).
- [ ] Choose and pin: vanilla NeRF implementation to adapt (record choice +
  source repo in `DECISION_LOG.md`).
- [ ] Set up experiment logging convention (`experiments/logs/` +
  `experiments/results/results.csv` schema).
- [ ] Set up Blender (headless-capable version) for scripted scene
  rendering.

**Gate:** a trivial PyTorch script runs a matrix multiply on the RTX 5060 in
WSL2 and reports the correct CUDA compute capability (12, 0) — *and* the same
environment file installs and runs a GPU op on both Colab and Kaggle without
modification.

**Deliverables:** working local + cloud environments; initialized repo
skeleton; `DECISION_LOG.md` entries for environment and NeRF-implementation
choice.

**Effort:** 0.5–1.5 days (higher end if Blackwell driver/CUDA issues surface —
budget for this explicitly rather than treating it as a rounding error).

**Risks:** tiny-cuda-nn / custom-CUDA-extension packages (relevant if
Nerfacto is later pursued) have had documented build issues on sm_120 as of
this writing. Mitigation: vanilla NeRF (pure PyTorch, no custom kernels) is
the Phase 1–8 default specifically to avoid this risk locally; anything
requiring custom CUDA extensions is pushed to cloud or deferred to Phase 9.

---

## Phase 1 — Literature Verification

**Objective:** confirm the novelty table in the feasibility assessment still
holds close to when experiments actually start, since this field moves fast.

**Tasks:**
- [ ] One focused search session re-running the queries from the original
  feasibility assessment (IPA-NeRF, NeRFool, 3DGS poisoning cluster,
  targeted object suppression, poisoning-budget sweeps in neural 3D
  representations).
- [ ] Confirm no new paper has published the exact combination: partial
  poisoning-budget sweep × single target object × suppression goal ×
  explicit collateral-damage control, on NeRF.
- [ ] Update `docs/DECISION_LOG.md` / `METHODOLOGY.md` intro if anything
  materially changed.

**Gate:** written confirmation (a short note, with links, in
`DECISION_LOG.md`) that the novelty claim from the feasibility assessment
still holds, or an explicit note of what changed and how the framing was
adjusted.

**Deliverables:** updated novelty-confirmation note.

**Effort:** half a day. This is a check, not a full systematic review —
don't let it expand.

---

## Phase 2 — Lego Clean-NeRF Sanity Check

**Objective:** de-risk the entire pipeline (data loading, camera poses,
training loop, rendering, metrics code) on the cheapest possible scene,
before any complexity is added.

**Tasks:**
- [ ] Train vanilla NeRF on the standard Blender Synthetic Lego dataset.
- [ ] Render novel views; compute PSNR against known held-out Lego test views.
- [ ] Verify camera-pose convention matches what Phase 4's custom Blender
  scene will use (same coordinate system, same JSON schema) — catching a
  convention mismatch here is far cheaper than catching it in Phase 4.

**Gate (quantitative):** clean Lego PSNR on held-out test views falls within
the range reported by the original NeRF paper / standard reproductions for
this dataset (roughly high-20s to low-30s dB, implementation-dependent —
record the exact reference figure and source in `DECISION_LOG.md` before
comparing). **Do not proceed to Phase 3 on a pipeline that hasn't hit this
number** — a poisoning result built on a broken clean baseline is not
interpretable.

**Deliverables:** working end-to-end train/render/metric pipeline; one
clean-Lego PSNR number with citation to the comparison source.

**Effort:** 1–2 days, mostly GPU time (small, cheap on local 8GB VRAM).

**Explicitly not a paper result** — one-line pipeline-verification mention
only, per `METHODOLOGY.md` framing.

---

## Phase 3 — Poisoning Proof of Concept (still on Lego)

**Objective:** confirm the poison-then-retrain mechanism visibly changes
reconstruction, and debug the masking/compositing code, before it's used on
data that actually matters for the paper.

**Tasks:**
- [ ] Implement hard-erasure compositing (`src/poisoning/compositor.py`)
  against Lego's own background/mask (Lego doesn't have a natural
  background plate — use a rough proxy here; this stage is about pipeline
  mechanics, not measurement).
- [ ] Run 2–3 rough budget levels (e.g. 0/20/50% of Lego's own visible
  views) purely to see the mechanism work.

**Gate:** visibly degraded/suppressed reconstruction at the higher budget
level, and no pipeline errors (shape mismatches, pose misalignment, mask
misregistration) across all conditions run.

**Deliverables:** working, tested compositing script; confirmation the
poisoning mechanism has an effect.

**Effort:** 1 day.

**Explicit non-goal:** no scientific conclusions are drawn from these
numbers — this phase exists to kill bugs, not to produce results.

---

## Phase 4 — Final Scene + Frozen Evaluation Set

**Objective:** build the actual scene and evaluation set the paper's results
come from.

**Tasks:**
- [x] Build the custom multi-object Blender scene — built *procedurally*
  from `configs/scenes/final_scene.yaml` via `scripts/build_scene.py`, not
  hand-modelled, so it is regenerable from config + commit (D-022).
- [x] Render the full training-view set with camera poses (100 train poses).
- [x] For every training view: matching `background_plate` and `mask`. Two
  real Cycles renders per view — the mask comes free from the same render
  as the original via the Object Index pass (D-022, D-025).
- [x] Render the held-out evaluation-view set (75 poses, disjoint by
  construction — all pools cut from one lattice, D-023).
- [x] Compute and record `|V_target|` = **100/100** training views, into
  `data/blender_scenes/cameras.json` (D-025).
- [x] **Freeze** the held-out set — read-only, checksummed, aggregate
  SHA-256 in `DECISION_LOG.md` D-026. Scope widened beyond this line's
  literal wording to include the holdout masks, plates and
  `transforms_test.json`, since every held-out metric is computed against
  those too.
- [x] Freeze `METHODOLOGY.md` (status now "FROZEN 2026-09-14"; D-026).

**Gate:** scene has a clearly identifiable target object separable from
background by mask; `|V_target|` is large enough that even the 5% condition
corresponds to a whole number of poisoned views ≥ 1 (if `|V_target|` is too
small, this is the point to add more training views, not to round awkwardly
later); eval set is frozen and checksummed.

**Deliverables:** final scene assets, full clean training set with
plates/masks, frozen eval set, frozen `METHODOLOGY.md`.

**Effort:** 2–4 days (scene building/rendering is the most variable-effort
step in the project — don't over-invest in photorealism beyond "one clear
target + legible context").

**Risk:** scope creep into a highly detailed scene. Mitigation: gate success
criteria above are deliberately minimal — a legible target + clean masks is
sufficient, extra detail buys nothing.

---

## Phase 5 — Poisoning Pipeline Build

**Objective:** implement the finalized attack-generation code against the
real scene.

**Tasks:**
- [x] Compositing implemented as ONE alpha-parameterized function (hard
  erasure is its `alpha=0` case), verified bit-identical to the Phase 3
  implementation over 10 cases. Soft-suppression `alpha` locked at **0.4**
  (D-029).
- [x] Random and strategic view-selection (`view_selection.py`), with an
  explicit `(-area, view_index)` tie-break (D-028).
- [x] `build_poison_set.py` rewritten — this is the **D-020 fix**: a
  condition dir now holds only `train/` + `transforms_train.json`, with
  val/eval_holdout reached via config paths instead of hand-made symlinks.
  Byte-identical regeneration from config alone proven (D-030).
- [x] **29 pytest tests** (`tests/`, previously empty): compositor pixel
  math, alpha=0 collapse, budget arithmetic pinned to the real function's
  outputs, tie-break determinism, mask bit-depth assertions.
- [x] Generated **20 datasets**, not 8 — `METHODOLOGY.md` §8 ties view
  selection to the same seed as model init, so C1–C6 get one set per seed;
  control and C7 are seed-invariant (D-028).

**Gate:** for every condition, `MANIFEST.csv` view counts match
`round(b/100 * |V_target|)` exactly; visual spot-check of a handful of
poisoned images per condition confirms only the target region differs from
the original.

**Deliverables:** poisoning pipeline code + tests; all 8 conditions'
poisoned image sets generated and manifested.

**Effort:** 2–3 days.

---

## Phase 6 — Minimum Poisoning-Budget Sweep

**Objective:** produce the paper's headline results.

**Tasks:**
- [ ] Train all 8 conditions (§5 of METHODOLOGY.md) × N seeds (minimum 3).
  That's **24 training runs minimum** for the core sweep — reconcile
  against available compute (local 8GB VRAM for iteration/debug runs,
  Colab/Kaggle for the bulk of the full-length training runs) before
  starting, and record the plan in `DECISION_LOG.md`.
- [ ] Every run gets an `experiments/logs/` entry (from the template) and a
  row in `experiments/results/results.csv` on completion.
- [ ] No metric is computed against anything other than the frozen eval set.

**Gate:** all runs complete without divergence; loss curves inspected for
each condition (a run that failed to converge is a bug to fix, not a data
point to report); `results.csv` has one row per (condition × seed).

**Deliverables:** complete `results.csv` for the core sweep; per-run logs.

**Effort:** the dominant compute cost of the project. Budget cloud GPU time
accordingly (see `environment/SETUP.md` for the local/cloud split
rationale).

---

## Phase 7 — Ablation (Random vs. Strategic View Selection)

**Objective:** answer whether *which* views are poisoned matters as much as
*how many*, at one informative budget level.

**Tasks:**
- [ ] Train condition C7 (20% budget, hard erasure, strategic selection) ×
  same N seeds as Phase 6.
- [ ] Compare directly against C3 (20% budget, hard erasure, random
  selection) from Phase 6 — same budget, same attack type, selection method
  is the only difference.

**Gate:** C7 runs complete cleanly; a direct C3-vs-C7 comparison table/plot
exists.

**Deliverables:** ablation results, comparison plot.

**Effort:** N additional training runs (small addition on top of Phase 6).

**Explicit non-goal:** this is one ablation, not a second sweep dimension —
do not expand into a full budget × selection-method grid unless Phase 8
analysis reveals a specific reason it's necessary (log that reasoning in
`DECISION_LOG.md` before doing it).

---

## Phase 8 — Evaluation & Analysis

**Objective:** turn `results.csv` into the paper's actual evidence.

**Tasks:**
- [ ] Compute masked/unmasked PSNR/SSIM/LPIPS per `METHODOLOGY.md` §6 for
  every run, from the frozen eval set only.
- [ ] Produce the core plots: target-suppression-vs-budget curve,
  collateral-damage-vs-budget curve (mean ± std across seeds, per
  §8 of METHODOLOGY.md), hard-erasure-vs-soft-suppression comparison,
  random-vs-strategic ablation comparison.
- [ ] Produce qualitative figures: rendered novel views at each budget
  level, target region close-ups vs. `background_plate` ground truth.
- [ ] Write up results neutrally — including if the effect is weak or
  threshold-shaped (see §8 negative-results handling below).

**Gate:** every plotted number traces to a `results.csv` row; every
`results.csv` row traces to an `experiments/logs/` entry and a git commit.

**Deliverables:** finished analysis notebook (promoted to `src/metrics/` +
`scripts/` for anything reused), all core plots and qualitative figures in
`paper/figures/`.

**Effort:** 2–3 days.

**Handling weak/negative results (decide the framing here, don't improvise
later):** if low poisoning budgets show negligible suppression, that is
itself a **reportable finding** — a threshold effect (suppression only
above budget X) is scientifically useful and should be framed as such, not
hidden. If soft suppression fails to produce meaningful suppression at all
at the tested alpha, report that as evidence about the alpha/attack-strength
boundary, and note it as the natural next experiment rather than re-running
with a new alpha inside this study (that would break the "alpha fixed
once" rule in `METHODOLOGY.md` §3).

---

## Phase 9 — Optional Strengthening Extensions (Conditional)

**Objective:** only pursued if time remains after Phase 8 is fully complete,
and only in this priority order (each addresses a specific named reviewer
objection — pick based on which objection is weakest after Phase 8, not by
default order):

1. Real-photo spot-check (Telea inpainting substituted for the background
   plate, per the earlier protocol discussion) — addresses "does this only
   work because you have perfect synthetic ground truth."
2. Nerfacto cross-model check — addresses "is this specific to vanilla
   NeRF." (Higher engineering risk on this hardware — see Phase 0 risk note
   on tiny-cuda-nn/Blackwell; plan to run this on cloud if pursued.)

**Gate to even start Phase 9:** Phase 8 is fully complete and written up.
**Do not start Phase 9 before Phase 8 is done** — this is the most common
way "research-paper worthy without crazy amounts of effort" projects blow
their budget.

---

## Phase 10 — Write-Up

**Tasks:**
- [ ] Draft using the structure already agreed for the paper.
- [ ] State the contribution at the precise, narrow scope justified by
  Phase 1's literature check — no broader claim.
- [ ] Explicitly address the reviewer objections already identified (naive
  erasure ≠ the whole contribution; object removal ≠ poisoning; generic
  NeRF-poisoning novelty is not claimed; single-scene limitation
  acknowledged).
- [ ] Include the negative/threshold-result framing from Phase 8 honestly.

**Gate:** complete draft with every quantitative claim traceable to
`experiments/results/results.csv`.

---

## What to Cut (if effort is running over budget)

In order of first-to-cut:

1. Phase 9 entirely (both extensions).
2. Phase 7 ablation's seed count down to N=1–2 (log this deviation).
3. Strategic-selection ablation reduced to a qualitative comparison instead
   of a full N-seed run, if compute is the binding constraint.
4. Number of seeds in the core sweep (Phase 6) down to N=2, minimum — never
   to N=1, since that removes the ability to report any variance and
   undermines the "data-driven" framing this whole project is built around.

Never cut: the frozen eval set discipline, the config-driven
reproducibility rules in `PROJECT_STRUCTURE.md`, or the soft-suppression
condition (C6) — it is the single highest-leverage item preempting the
"you just deleted the object" objection.
