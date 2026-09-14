# Decision Log

Append-only. Never edit or delete a past entry — if a decision is reversed,
add a new entry that supersedes it and says so. This is what lets someone
(including future you) understand *why* the project looks the way it does,
not just what it currently looks like.

**Entry format:**

```
## D-XXX — <short title>
- **Date:**
- **Decision:**
- **Alternatives considered:**
- **Rationale / evidence:**
- **Reversibility:** (how costly to undo, and what would trigger revisiting it)
```

---

## D-001 — Research niche and scope

- **Date:** project start
- **Decision:** focus on partial poisoning-budget sweep × target-object
  suppression × explicit collateral-damage control on vanilla NeRF, not a
  new attack paradigm.
- **Alternatives considered:** full IPA-NeRF-style viewpoint-conditioned
  backdoor reproduction; whole-scene untargeted degradation (NeRFool-style);
  3D Gaussian Splatting instead of NeRF.
- **Rationale / evidence:** literature review (feasibility assessment,
  Sections 2–3) found this exact intersection unpublished, while each
  individual component (poisoning-fraction ablations, NeRF as an attack
  surface, object-level scene editing) is separately established. Novelty
  classification: "Meaningfully differentiated empirical study," bordering
  on "Potentially novel research direction" — not "already done," not
  overstated as a new paradigm.
- **Reversibility:** low cost to revisit before Phase 4 (scene not yet
  built); high cost after Phase 6 (sweep already run against this framing).

## D-002 — Poisoning-budget definition anchored to target-visible views

- **Date:** project start
- **Decision:** budget = percentage of `V_target` (views where the target
  object is visible above a fixed area threshold), not percentage of all
  training views.
- **Alternatives considered:** percentage of all training views (original
  proposal); raw count of poisoned images; spatial coverage of the target
  across the scene.
- **Rationale / evidence:** percentage-of-all-views is not well-defined when
  target visibility is uneven across views — two "20%" runs could poison
  very different fractions of the views that actually contain the target,
  making cross-condition comparison invalid. This was identified as the
  single most important methodological fix in the feasibility assessment
  (Executive Verdict, point 2).
- **Reversibility:** must be locked before Phase 6 — changing it after the
  sweep invalidates the whole budget axis. See `METHODOLOGY.md` §2.

## D-003 — Custom Blender multi-object scene over adapted real-world dataset

- **Date:** project start
- **Decision:** build a custom synthetic multi-object Blender scene for the
  main study, rather than adapting Mip-NeRF 360 or Google Scanned Objects.
- **Alternatives considered:** Mip-NeRF 360 scenes; Google Scanned
  Objects-derived multiview scenes.
- **Rationale / evidence:** a custom synthetic scene gives free, perfect
  ground-truth target masks and background plates (via Blender's
  object-visibility toggle and object-ID pass), avoiding both the
  mask-annotation cost and the inpainting-artifact risk that a real
  photographic dataset would introduce. This directly enables the
  hard-erasure/soft-suppression compositing protocol in `METHODOLOGY.md`
  §3. Feasibility assessment flagged real-dataset mask annotation as the
  most expensive, least necessary part of the project.
- **Reversibility:** moderate — if scene-building proves harder than
  expected, falling back to an existing dataset is possible before Phase 4
  is complete, at the cost of losing the free ground-truth plates and
  needing to substitute inpainting-based erasure even for the main study
  (as currently planned only for the optional Phase 9 real-photo check).

## D-004 — Soft-suppression added as a near-mandatory second attack condition

- **Date:** project start
- **Decision:** include soft suppression (fixed-alpha blend toward the
  background plate) as condition C6, not as an optional extension.
- **Alternatives considered:** hard erasure only, treating soft suppression
  as a Phase 9 stretch goal.
- **Rationale / evidence:** the most predictable reviewer objection to a
  poisoning study that only deletes pixels is "of course the model can't
  reconstruct something it never saw" — this is a one-sentence dismissal
  that a soft, still-partially-visible degradation condition directly
  preempts. Feasibility assessment identified this as the cheapest,
  highest-leverage change available and treated it as close to mandatory.
- **Reversibility:** low cost to add/keep (compositing formula already
  generalizes hard erasure as the alpha→0 case); should not be cut except
  under the most severe time pressure (see ROADMAP.md "What to Cut").

## D-005 — Vanilla NeRF as the primary model; Nerfacto deferred to optional Phase 9

- **Date:** project start
- **Decision:** implement and run the entire core study (Phases 1–8) on a
  vanilla NeRF implementation (pure PyTorch, no custom CUDA extensions).
  Nerfacto (via Nerfstudio, which depends on the tiny-cuda-nn extension) is
  deferred to an optional Phase 9 cross-model check only.
- **Alternatives considered:** Nerfacto/Nerfstudio as the primary model from
  the start (faster training, more modern architecture).
- **Rationale / evidence:** the development laptop's GPU (RTX 5060 laptop,
  Blackwell architecture, sm_120 compute capability) has documented,
  currently-unresolved build/compatibility issues with tiny-cuda-nn and the
  Nerfstudio viewer, per open Nerfstudio GitHub issues as of late 2025.
  Vanilla NeRF has no custom CUDA extensions and works cleanly with
  standard PyTorch ≥2.7 cu128 wheels. Given the project's stated goal
  ("research-paper worthy without crazy amounts of effort"), avoiding a
  known environment-compatibility risk for the core study is the correct
  trade, at the cost of somewhat slower training.
- **Reversibility:** Nerfacto can be added later (Phase 9) without
  invalidating Phase 1–8 results, since it would be a separate, clearly
  labeled cross-model comparison, not a replacement of the core sweep.

## D-006 — Local (WSL2) + cloud (Colab/Kaggle) split development environment

- **Date:** project start
- **Decision:** develop and debug locally on WSL2 (Ubuntu) with PyTorch
  cu128+ on the RTX 5060 laptop GPU; run the bulk of full-length training
  runs (the Phase 6 sweep) on Colab/Kaggle cloud GPUs. No dual-boot native
  Linux.
- **Alternatives considered:** fully local (compute-constrained by 8GB
  VRAM and single-machine throughput); fully cloud (slower iteration loop
  for pipeline debugging); native Linux dual-boot (no meaningful compute
  advantage over WSL2 for this workload; adds partition-management and
  context-switching overhead).
- **Rationale / evidence:** WSL2 GPU compute passthrough is near-native for
  training workloads (as opposed to graphics-heavy workloads, where the gap
  matters more), giving a single, low-friction OS for all Linux-native NeRF
  tooling without reboot overhead. 8GB VRAM comfortably handles vanilla
  NeRF on compact scenes for iteration; cloud GPUs provide overflow
  capacity for the ~24-run core sweep without tying up the laptop for days,
  and Kaggle's free weekly GPU-hour allowance supplements Colab's.
- **Reversibility:** low cost — the environment spec (`environment.yml`) is
  designed to install identically in both contexts specifically so this
  split can be adjusted without re-doing setup work.

## D-007 — Statistical reporting: minimum 3 seeds per condition

- **Date:** project start
- **Decision:** every condition in the core sweep (Phase 6) is trained with
  a minimum of N=3 seeds; results reported as mean ± std, never a single
  run.
- **Alternatives considered:** single run per condition (cheaper, faster);
  N=5+ seeds (more statistically robust, more compute).
- **Rationale / evidence:** a poisoning-budget curve built from single runs
  per condition cannot distinguish a real budget effect from run-to-run
  training variance — this is exactly the kind of gap a reviewer would
  flag. N=3 is the minimum that allows any variance reporting at all, chosen
  as the floor that keeps the ~24-run core sweep compute-manageable per the
  feasibility assessment's estimate (~8–15 trainings for a "minimum
  defensible" to "stronger but manageable" range) once the ablation (Phase
  7) is added on top.
- **Reversibility:** can be reduced to N=2 as a documented last-resort cut
  (see ROADMAP.md "What to Cut") if compute runs out; should not go to N=1.

## D-008 — Blender installed natively on Windows, not inside WSL2

- **Date:** project start
- **Decision:** Blender (scene building + headless dataset rendering) runs
  as a native Windows install. Everything downstream of the raw renders
  (compositing, training, metrics) stays in the WSL2 Python environment.
- **Alternatives considered:** installing Blender inside WSL2/WSLg for a
  single-OS workflow.
- **Rationale / evidence:** Blender's Cycles/OptiX GPU rendering has mature,
  driver-based Blackwell support via the standard NVIDIA Studio driver —
  no sm_120-specific wheel-pinning or nightly-build fights like the
  PyTorch/tiny-cuda-nn situation in D-005. Running the GUI (needed for
  interactive scene building in Phase 4) through WSLg's graphics
  translation layer adds friction for no benefit when the native path is
  this straightforward. Headless rendering (`blender.exe --background ...
  --python ...`) works identically well from a Windows terminal, so there's
  no pipeline-automation reason to prefer WSL2 either.
- **Reversibility:** low cost either way — the render outputs are just
  image/JSON files; if WSL2/WSLg Blender support becomes preferable later,
  switching only affects Phase 4 tooling, not any already-frozen data or
  results.

## D-009 — uv (not conda) as local package manager; cu128→cu130 reconciliation

- **Date:** 2026-09-12
- **Decision:** two related corrections to Phase 0 setup, superseding the
  conda-based wording in D-006/`environment/SETUP.md` §1.5 and the exact
  cu128 pin implied by D-005/`environment/SETUP.md` §1.3 (neither D-005 nor
  D-006 is edited — this entry supersedes the relevant parts of both, per
  this log's append-only rule):
  1. **Package manager:** the project standardizes on `uv`
     (`pyproject.toml` + `uv.lock`) for the local environment, not conda.
     `environment/environment.yml` is deleted; `environment/SETUP.md` §1.5
     now documents `uv sync` instead of `conda env create`.
  2. **CUDA wheel version:** cu128 was always intended as a *minimum
     floor* (the first cu-tagged wheel line with native sm_120/Blackwell
     support), not a required exact pin. The environment actually in use
     (`pyproject.toml`'s `cu130` index, `torch==2.14.0+cu130` per
     `uv.lock`) has been verified working on the reference dev machine
     (RTX 5060 laptop) — `scripts/verify_env.py` reports compute
     capability `(12, 0)` and a passing GPU matrix multiply. cu130 is now
     the documented pinned version; cu128 remains the floor for anyone
     building on older wheels.
- **Alternatives considered:** (a) actually installing conda/miniconda
  locally to match the original `environment.yml` plan; (b) rolling the
  existing `pyproject.toml`/`uv.lock` back to an exact `cu128` pin to match
  the original wording literally.
- **Rationale / evidence:** (a) was rejected because conda was never
  actually installed or needed on this machine — a working `uv`-managed
  `.venv` with a functioning CUDA-enabled torch already existed before
  Phase 0 setup began, and installing a second, redundant package manager
  just to match documentation that predated that `.venv` would be pure
  overhead with no functional benefit. (b) was rejected because cu130 is
  already verified working on the exact reference hardware this project
  targets, and downgrading a working, verified install to match a
  conservative floor chosen before any hardware verification happened
  would trade a real, tested result for an untested "safer-looking" one.
- **Reversibility:** low cost. Package manager: switching back to conda
  would only mean re-adding `environment/environment.yml` and reverting
  `environment/SETUP.md` §1.5 — no code depends on the choice of manager.
  CUDA wheel version: `pyproject.toml`'s `[[tool.uv.index]]` URL and the
  version pins are the only places this is encoded; falling back to cu128
  would be a one-line index change if cu130 ever proves unstable. Colab/
  Kaggle parity (cu128 floor, since those platforms are not Blackwell) is
  unaffected either way and remains untested until the first cloud run
  (Phase 6) — see `docs/ROADMAP.md` status tracker.

## D-010 — pyproject.toml completed as single dependency source of truth; torchaudio dropped

- **Date:** 2026-09-12
- **Decision:**
  1. `pyproject.toml` (via `uv add`) now declares the full pipeline
     dependency set — `numpy`, `imageio`, `opencv-python`, `pyyaml`,
     `tqdm`, `tensorboard`, `scikit-image`, `lpips` — in addition to the
     existing torch stack, closing the gap flagged after D-009 where these
     lived only in `requirements.txt`. `environment/requirements.txt` is
     now machine-generated from `uv.lock` (`uv export --no-hashes --no-dev
     --no-editable --no-emit-project --emit-index-url`) rather than
     hand-maintained, so the two can no longer drift apart by construction.
  2. `torchaudio` is removed from `pyproject.toml`/`uv.lock` (via `uv
     remove`) and from `requirements.txt`.
- **Alternatives considered:** continuing to hand-maintain
  `requirements.txt` as a second, manually-synced list; keeping
  `torchaudio` in case it becomes useful later.
- **Rationale / evidence:** hand-maintaining two dependency lists is
  exactly the kind of untracked-divergence risk `PROJECT_STRUCTURE.md`
  warns against generally — generating one from the other removes the
  failure mode entirely instead of relying on discipline to keep them in
  sync. `torchaudio` has no role in this vision-only NeRF pipeline (no
  audio data, model, or metric anywhere in `METHODOLOGY.md`); it was dead
  weight pulled in by the original project scaffold, not a deliberate
  future dependency, so it's removed now rather than carried forward
  unused.
- **Reversibility:** low cost either way. Re-adding `torchaudio` is a
  single `uv add torchaudio`; regenerating `requirements.txt` after any
  future `pyproject.toml` change is the one documented command in the
  file's own header comment.

## D-011 — Fixed a scaffold bug blocking `uv`: literal brace-glob directory under `src/`

- **Date:** 2026-09-12
- **Decision:** removed an empty, incorrectly-named directory at
  `src/{nerf,poisoning,data_pipeline,metrics,utils}` (a literal directory
  name, not five separate directories — the result of an unexpanded shell
  brace-glob from the original repo scaffolding, e.g. `mkdir` run under a
  shell/context where `{a,b,c}` doesn't expand). This directory predates
  this session's work and was never git-tracked (git does not track empty
  directories, and it appears nowhere in `git log --all`). In its place,
  created the actual five directories: `src/nerf/` (with a placeholder
  `__init__.py`, since `pyproject.toml`'s `uv_build` backend requires
  `src/nerf/__init__.py` to build the declared `nerf` package — this was
  the only one of the five the build system actually required) and
  `src/poisoning/`, `src/data_pipeline/`, `src/metrics/`, `src/utils/`
  (each with a `.gitkeep` placeholder — not required by the build system,
  just the documented `PROJECT_STRUCTURE.md` layout ahead of Phase 5+ code).
- **Alternatives considered:** marking the project non-buildable via
  `[tool.uv] package = false` in `pyproject.toml` to sidestep the build
  requirement entirely, rather than fixing `src/`; leaving the bug and
  routing around it with `uv add --frozen` (which would have added
  dependency declarations without actually locking/syncing them, silently
  leaving the environment in a state that only looks correct).
- **Rationale / evidence:** the bug was completely silent until `uv
  add`/`uv sync` was attempted for the first time (D-010's dependency
  work) — nothing before that point exercised the build path, so it sat
  undetected since initial scaffolding. Fixing the actual `src/` layout
  (rather than disabling packaging) keeps `pyproject.toml`'s existing
  intent — a real, buildable `nerf` package plus a working `nerf` console
  script — intact, and is the more honest fix given the placeholder
  `__init__.py` costs nothing and unblocks the declared build config
  exactly as originally intended, rather than quietly changing what
  `pyproject.toml` claims about itself. This is exactly the kind of
  silent-failure-turned-loud-discrepancy `DECISION_LOG.md` exists to
  catch, per this project's ground rules in `README.md`.
- **Reversibility:** trivial — the placeholder `src/nerf/__init__.py` and
  `.gitkeep` files are removed/replaced the moment real code lands in
  Phase 2+; no design decision is locked in by this fix beyond "the src/
  layout now matches what `pyproject.toml` already declared."

## D-012 — Phase 1 literature verification: novelty claim reconfirmed

- **Date:** 2026-09-12
- **Decision/finding:** the novelty claim from the original feasibility
  assessment still holds. No published work combines (a) a
  poisoning-budget sweep defined over target-visible views, (b) a
  NeRF-specific attack, (c) an object-suppression/removal goal, and (d) an
  explicit collateral-damage-vs-budget curve as the central result.
- **Evidence reviewed:**
  - **IPA-NeRF** (ECAI 2024, arXiv:2407.11921) — bi-level optimization,
    single backdoor-viewpoint illusion injection, ablates distortion
    budget (epsilon) and angle constraints, not poisoned-view-fraction.
  - **StealthAttack** (arXiv:2510.02314, Oct 2025) — density-guided
    Gaussian injection for 3DGS (not NeRF), illusion-injection goal,
    ablates number of poisoned viewpoints (2/3/4) as "how many views show
    the same illusion," reports innocent-view (V-TEST) fidelity alongside
    attack success — closest adjacent pattern to a budget/collateral
    framing, but wrong representation and wrong attack goal.
  - **Poison-splat** (arXiv:2410.08190, 2024) — the one paper found with
    an actual poisoning-ratio sweep (20/40/60/80%, randomly selected
    views) on 3DGS, but the attack goal is a compute-cost/memory
    denial-of-service, unrelated to visual/object integrity.
  - **Shielding the Unseen** (arXiv:2310.03125, 2023) — untargeted
    whole-scene spatial-deformation degradation (privacy motivation), not
    object-selective.
  - **Generalizable Targeted Data Poisoning** (arXiv:2412.03908, 2024) and
    the "Checkerboard" clean-label backdoor line of work —
    poisoning-budget-vs-success-rate curves, but for 2D image classifiers,
    not 3D scene reconstruction.
- **Conclusion:** no deviation to `METHODOLOGY.md` required. Novelty
  classification unchanged from the original feasibility assessment
  ("Meaningfully differentiated empirical study," bordering "Potentially
  novel research direction").
- **Reversibility:** n/a (a finding, not a design decision) — revisit if a
  focused re-check before Phase 6 write-up surfaces a new publication
  combining all four elements above.

## D-013 — Phase 2 clean-Lego PSNR comparison basis, fixed before training

- **Date:** 2026-09-12
- **Decision:** Phase 2's gate is satisfied if this project's clean-Lego
  PSNR on held-out test views falls within approximately **29–33 dB**.
  Below that range is a pipeline bug to fix, not a result to report.
- **Primary reference:** the original NeRF paper (Mildenhall et al. 2020)
  reports Lego PSNR = 32.54 dB, per the comparison table in Mip-NeRF
  (Barron et al. 2021, arXiv:2103.13415), which directly cites the
  original NeRF numbers.
- **Reproduction variance context:** independent reproductions of vanilla
  NeRF on Lego have reported PSNR as low as ~29.5 (per a GL-NeRF paper's
  own reproduction, arXiv:2410.19831) and as high as ~31.65 (per a
  Rip-NeRF paper's reproduction, arXiv:2405.02386), confirming a realistic
  acceptance range of roughly 29.5–32.5 dB rather than requiring an exact
  match to 32.54.
- **Rationale / evidence:** fixing the comparison range *before* running
  anything (per `ROADMAP.md`'s Phase 2 gate instructions) is what makes
  the gate a real check rather than a post-hoc rationalization — if the
  range were picked after seeing a number, it would stop being a gate.
- **Reversibility:** must be locked before training per the above; a
  training run that misses this window is a documented pipeline bug to
  fix (see Phase 2 gate outcome in `ROADMAP.md`'s status tracker), not
  grounds for redefining the range afterward.

## D-014 — Fixed the same literal brace-glob bug under `data/`

- **Date:** 2026-09-12
- **Decision:** removed an empty, incorrectly-named directory at
  `data/{raw,blender_scenes,poisoned,masks}` (the same class of bug as
  D-011, in `data/` instead of `src/` — an unexpanded shell brace-glob
  from the original repo scaffolding). Never git-tracked, confirmed empty.
  Replaced with the real directories per `PROJECT_STRUCTURE.md`:
  `data/raw/`, `data/blender_scenes/`, `data/masks/`,
  `data/background_plates/`, `data/poisoned/`. Also added
  `data/nerf_synthetic/` — an undocumented-until-now directory holding the
  external Blender Synthetic dataset used for Phase 2's sanity check only;
  it is separate from `data/raw/` (reserved for this project's own
  `.blend` scene files) and `data/blender_scenes/` (reserved for the
  Phase 4 main-study scene, subject to the eval-holdout freeze rule) —
  documented in `PROJECT_STRUCTURE.md` and excluded from git via
  `.gitignore` (241MB of downloaded PNGs, not generated by any script
  here).
- **Rationale / evidence:** found while setting up Phase 2's dataset
  directory; fixing it now (rather than routing around it) keeps
  `PROJECT_STRUCTURE.md`'s documented tree accurate, consistent with how
  D-011 handled the identical bug pattern in `src/`.
- **Reversibility:** trivial — these are empty placeholder directories
  (aside from the downloaded, gitignored Lego data) with no design
  decision locked in.

## D-015 — Fixed a silently-broken `.gitignore` (every pattern had 3 leading spaces)

- **Date:** 2026-09-12
- **Decision:** stripped the 3 leading spaces baked into every line of the
  top-level `.gitignore` (from the original scaffold — `sed -n l` / `cat
  -A` showed every pattern, including comments, prefixed with `   `).
  Confirmed via `git check-ignore -v` that this made every single pattern
  in the file a silent no-op: `__pycache__/`, `*.pth`, `data/blender_scenes/`,
  and even `.venv` were **not actually being ignored** by this file (`.venv`
  merely never showed up in `git status` because it carries its own nested
  `.venv/.gitignore`, unrelated to the project's). After stripping the
  leading whitespace, `git check-ignore -v` confirms all of the patterns
  exercised so far (`__pycache__/`, `.venv`, `data/nerf_synthetic/`,
  `experiments/runs/`) now match correctly.
- **Rationale / evidence:** found while adding vendored code under `src/`
  for Phase 2 and noticing `__pycache__/` directories showing as untracked
  in `git status` despite `.gitignore` listing that exact pattern. This is
  the same class of issue as D-011/D-014 (a scaffold-generated file with a
  formatting defect that silently did nothing until something finally
  exercised it) — fixed immediately rather than routed around, since a
  non-functional `.gitignore` risks accidentally committing large
  generated/checkpoint files later in the study.
- **Reversibility:** trivial — whitespace-only change, no ignore rules
  were added, removed, or reworded, just made to actually take effect.

## D-016 — Phase 2 gate result: clean-Lego PSNR PASS (31.55 dB)

- **Date:** 2026-09-13
- **Decision/finding:** Phase 2's gate **PASSES**. Final held-out
  test-set PSNR is **31.550 dB** over all 200 `transforms_test.json`
  views, against the D-013 acceptance range of **29–33 dB**. 31.55 dB
  falls solidly inside that range, not near either edge.
- **Training details:** `configs/scenes/lego_sanity.yaml`, vendored
  vanilla NeRF (`src/nerf/`, per D-005), seed 0, 200,000/200,000
  iterations completed (no early stop, no divergence). Trained locally on
  WSL2, RTX 5060 laptop GPU, PyTorch 2.14.0+cu130. Training-loop wall time
  11.91h; final train-batch PSNR 33.36 dB, val-subset PSNR 31.63 dB at
  iteration 200,000 (both consistent with the independently-recomputed
  200-view test PSNR below). Git commit at training start:
  `d1906094dd7437f5c0bc582d989a35991940df43`.
- **Incident during this run (full detail in the conversation record, summarized
  here for the audit trail):** the vendored training script's built-in
  post-training behavior (`src/nerf/training.py`) automatically renders
  the full test set before returning, with no per-image progress logging
  at the time. Combined with the run sitting at ~7.3–7.6GB/8.15GB VRAM,
  this full-test-set render took **5.5+ hours** on the original run (vs.
  an expected ~40–70 min) and was indistinguishable from a hang from the
  log alone — GPU stayed at 100% util and the process kept accumulating
  CPU ticks throughout, confirming it was genuinely still computing, just
  starved for VRAM headroom. Diagnosed via `/proc/<pid>/stat` CPU-tick
  deltas across repeated samples (a hang would show flat ticks; this
  didn't), not by guessing. Resolved by: (1) killing that process
  (weights were already checkpointed at iteration 200,000 in
  `200000.tar`, so no retraining was lost), (2) adding per-image
  progress logging to `evaluate_psnr` (`show_progress` argument,
  `src/nerf/training.py`), (3) adding `scripts/evaluate.py` — a
  standalone, previously-planned-but-unbuilt entry point
  (`PROJECT_STRUCTURE.md`) that loads a checkpoint and re-evaluates
  without retraining, and (4) re-running the test-set evaluation alone
  with full VRAM headroom, which completed in 38.2 minutes — closely
  matching the original estimate and confirming the VRAM-contention
  diagnosis. The 31.550 dB figure above is from that clean, standalone
  re-evaluation, not the original contended run.
- **Sample renders:** 4 novel test views (indices 200, 266, 333, 399),
  each saved as a ground-truth | rendered side-by-side PNG, at
  `experiments/results/phase2_lego_sanity/`. Visual check: structurally
  correct (recognizable Lego model, correct colors/geometry, no floaters
  or missing geometry), somewhat softer than ground truth on fine detail
  (tread patterns, thin structural bars) — consistent with a
  properly-converged vanilla NeRF at this quality level, not a pipeline
  bug.
- **Rationale / evidence:** this is the reference-basis comparison locked
  in D-013, applied to the actual measured number, per `ROADMAP.md`
  Phase 2's gate ("do not proceed to Phase 3 on a pipeline that hasn't
  hit this number").
- **Reversibility:** n/a (a measured result, not a design decision).
  Phase 3 may now begin per `ROADMAP.md`.

## D-017 — Phase 3 Lego PoC: mask/background-proxy/`V_target` choices (scoped to this PoC only)

- **Date:** 2026-09-13
- **Decision:** for the Phase 3 pipeline-mechanics PoC on the Blender
  Synthetic Lego dataset only (`ROADMAP.md` Phase 3), fix these three
  choices, all separate from and non-binding on the Phase 4 final-scene
  protocol:
  1. **Mask source:** each Lego RGBA PNG's own alpha channel, binarized at
     `alpha > 127` (foreground/target) vs. `alpha <= 127` (background).
     Verified against the actual downloaded files
     (`data/nerf_synthetic/lego/train/*.png`, spot-checked `r_0`, `r_1`,
     `r_50`, `r_99`): all are RGBA, `uint8`, alpha ranges over the full
     `[0, 255]`, but a per-image histogram shows alpha is overwhelmingly
     bimodal at exactly 0 or 255 (e.g. `r_0.png`: 487176 px at 0, 145940 px
     at 255, out of 640000) with a thin anti-aliased edge band — sampling
     every 10th training view (10 images), only **1.45%** of pixels have an
     intermediate alpha value. So the alpha channel is a near-perfect but
     not exactly binary segmentation mask; a threshold is needed at the
     silhouette edge, not because the mask is generally ambiguous.
     `alpha > 127` (standard 8-bit midpoint) was used to binarize it rather
     than a soft/continuous blend, since hard erasure is specified as a
     hard-edged operation in `METHODOLOGY.md` §3 (in contrast to soft
     suppression's explicit blending).
  2. **Background-proxy method:** Lego has no rendered `background_plate`
     (no object-toggled-off pass exists for this external dataset). Use a
     fixed flat proxy color, RGB `(128, 128, 128)` (mid-gray), applied
     identically to every poisoned pixel in every image/condition — no
     per-image sampling or hand-picking. Chosen over alternatives because
     it requires zero per-image judgment (the non-negotiable rule in
     `CLAUDE.md`) and is visually distinct from both the dataset's
     white-background compositing convention (`configs/base.yaml`
     `render.white_background: true`) and the Lego model's own color
     palette, making the erasure effect unambiguous to eyeball during
     validation.
  3. **`V_target` = all 100 `transforms_train.json` training views.**
     Every Lego training view shows the object (there is only one object
     in frame, always at least partially visible), so under `METHODOLOGY.md`
     §2's definition (`V_target` = views where the target's mask exceeds a
     fixed minimum-area threshold), the threshold is trivially satisfied by
     every view here. This keeps `D-002`'s target-visible-view definition
     intact rather than special-casing it away for the PoC.
  - **Alternatives considered:** (mask) using Blender's object-ID pass
    instead of alpha — not applicable, these are pre-rendered external PNGs
    with no accompanying ID pass; (mask) no thresholding, treat alpha as a
    continuous mask — rejected because hard erasure is defined as a hard
    replacement, and a continuous blend at 1.45% of pixels would silently
    turn hard erasure into a mild soft-suppression variant at every edge.
    (background) sampling each image's own transparent-background fill
    color — rejected, that fill is arbitrary per-pixel noise under
    zero-alpha (unpremultiplied), not a meaningful "scene background" to
    replicate; (background) pure white/black — rejected as visually
    confusable with the white-background compositing convention (white) or
    the model's own dark parts (black), which would make erasure harder to
    visually verify. (`V_target`) restricting to a subset of "best" views —
    rejected, no such distinction exists for a single-object turntable
    dataset, and it would be an unjustified per-condition judgment call.
- **Rationale / evidence:** direct inspection of the downloaded dataset
  files (channel counts, alpha histograms) rather than assumption, per this
  step's explicit instruction to confirm rather than assume the alpha
  channel's behavior.
- **Reversibility:** trivial to reverse/ignore — these choices are
  explicitly scoped to the Phase 3 Lego PoC (`condition_id`s
  `phase3_poc_budget_*`) only. They are **not** the final scene's
  mask/background-plate protocol; that is a Phase 4 deliverable per
  `ROADMAP.md` Phase 4 ("render the matching `background_plate` ... and
  `mask` ... for every training view") and must come from real
  Blender-rendered plates per `METHODOLOGY.md` §3, not an alpha-channel/
  flat-color proxy. No existing config, decision, or dataset is
  overwritten by this entry — `configs/poisoning/budget_20.yaml` (the real
  condition C3) and `configs/scenes/final_scene.yaml` are untouched.

## D-018 — Phase 3 PoC training: 30,000-iteration budget (does not set Phase 6 precedent)

- **Date:** 2026-09-13
- **Decision:** train all three Phase 3 PoC conditions
  (`phase3_poc_budget_00/20/50`) for **30,000 iterations** each, same
  seed (0), same `lego_sanity.yaml`-derived hyperparameters otherwise
  (only the training image set differs per condition).
- **Alternatives considered:** the full 200,000-iteration Phase 2 budget
  — rejected as disproportionate to Phase 3's ~1-day effort estimate
  (`ROADMAP.md`) and its explicit non-goal of drawing conclusions from
  these numbers (3x ~12h runs is a multi-day cost for a mechanism check);
  a much smaller budget (e.g. 5,000 iterations) — rejected as
  under-converged even for coarse shape, given `lego_sanity.yaml`'s
  `precrop_iters: 500` warmup and 500,000-step LR decay schedule (5,000
  iterations is only 1% into that schedule and well within the
  precrop-dominated early phase).
- **Rationale / evidence:** 30,000 iterations is standard "coarse
  structure visible, fine detail not converged" territory for vanilla
  NeRF on this dataset (consistent with the qualitative behavior reported
  for the original NeRF codebase at this stage of training). Using the
  measured Phase 2 rate (200,000 iterations / 11.91h training-loop time =
  ~4.66 iter/s on this GPU), 30,000 iterations is a ~1.8h/run estimate,
  ~5.4h for all three conditions sequentially — proportionate to Phase
  3's effort budget while still giving each condition tens of thousands
  of gradient steps to visibly separate a fully-erased-object condition
  from a clean one. Also aligned to
  `configs/base.yaml`'s `logging.checkpoint_every: 25000`, so each run
  gets one interim checkpoint (25,000) plus the final one (30,000).
- **Reversibility:** trivial — this is a per-run config value
  (`configs/poisoning/phase3_poc_budget_*.yaml`'s `training.iterations`
  override), scoped to these three `condition_id`s only. It has no
  bearing on Phase 6's core-sweep iteration count, which is a separate,
  not-yet-made decision (that sweep trains on the final scene, not Lego,
  and needs its own convergence check against that scene's own
  reference PSNR range, analogous to D-013).

## D-019 — `skip_final_test_eval` / `--skip-final-eval` added to `src/nerf/training.py` / `scripts/train.py`

- **Date:** 2026-09-13
- **Decision:** added an opt-in `skip_final_test_eval: bool = False`
  parameter to `train_from_config` (`src/nerf/training.py`) and a
  matching `--skip-final-eval` CLI flag to `scripts/train.py`. When set,
  `train_from_config` skips the full-test-set render it otherwise runs
  unconditionally right before returning, and returns `test_psnr: None`
  in its summary dict instead. **Default (flag unset) is unchanged** —
  the original unconditional full-test-set eval still runs, preserving
  exact behavioral parity with the Phase 2 run this code was validated
  against (D-016).
- **Alternatives considered:** (a) leave `train_from_config` as-is and
  just also call `scripts/evaluate.py` afterward — rejected, this does
  **not** avoid the problem: `train_from_config` would still run its own
  full-test-set eval first, in-process, while the training run's CUDA
  allocations are still held, before ever returning control to the
  caller. That in-process eval is exactly the D-016 stall pattern (a
  cheap operation under full VRAM headroom, but slow/contended right
  after a long training loop) — D-016 fixed this for the *standalone
  re-evaluation* use case by adding `scripts/evaluate.py`, but never
  removed or gated the original in-training auto-eval that caused the
  problem in the first place, so the same class of run (train
  immediately followed by needing its test PSNR) was still exposed to
  it. (b) unconditionally remove the in-training auto-eval — rejected,
  it's used by existing callers (e.g. anyone re-running the Phase 2
  config as originally written) and removing it outright would be a
  silent behavior change to already-validated code, not a scoped
  addition. (c) make skipping the default — rejected for the same
  reason; a new run's default behavior should match what was already
  validated unless a caller opts out.
- **Rationale / evidence:** built for and used by this step's 3-condition
  Phase 3 PoC training sweep (`scripts/train.py ... --skip-final-eval`,
  each run's `test_psnr: null` in `experiments/runs/<run_id>/summary.json`
  confirms it took effect) specifically to avoid a repeat of D-016's
  5.5h VRAM-contention stall across three back-to-back runs. This is
  shared infrastructure (`src/nerf/training.py`, `scripts/train.py`), so
  it will also be available to Phase 6's core sweep, which trains many
  more runs back-to-back and has the same exposure.
- **Reversibility:** trivial — additive, opt-in, default-preserving. No
  existing config or run is affected unless it explicitly passes the new
  flag.

## D-020 — Known debt: Phase 3 PoC's poisoned `val`/`test` splits are manually symlinked, not `build_poison_set.py`-generated

- **Date:** 2026-09-13
- **Decision/finding:** `data/poisoned/phase3_poc_budget_{00,20,50}/val`,
  `.../test`, `.../transforms_val.json`, and `.../transforms_test.json`
  are filesystem symlinks pointing at `data/nerf_synthetic/lego`'s
  original (unmodified) files, created by hand (`ln -s`) during this
  step's training pass — not produced by `scripts/build_poison_set.py`,
  which currently only writes `train/` and `transforms_train.json`
  (per its Step-1 scope: build+validate the poisoned training set, no
  training yet). `load_blender_data` requires all three splits present
  under one `dataset.path`, so training needed *something* there; since
  val/test are never poisoned (`METHODOLOGY.md` only poisons training
  views), symlinking the unmodified source in was a non-destructive way
  to unblock training without touching the already-validated `train/`
  content.
  - **This is scoped to the Phase 3 Lego PoC only, and is logged here as
    known debt, not fixed now:** `build_poison_set.py` should be
    extended to emit a fully self-contained, reproducible-from-config
    dataset directory (`val`/`test` included, however that's decided to
    be sourced) as part of Phase 5's poisoning-pipeline build, before
    it's relied on for the real 8-condition sweep (`METHODOLOGY.md` §5).
    Relying on manually-created symlinks for that sweep would violate
    `PROJECT_STRUCTURE.md`'s "a poisoned condition must be fully
    reproducible by rerunning `build_poison_set.py` against its config
    alone" rule — the symlinks are outside that script's control and
    wouldn't be recreated by a fresh run of it.
- **Alternatives considered:** having `build_poison_set.py` copy (not
  symlink) val/test at Step-1 time — deferred rather than rejected; a
  real design question (copy vs. symlink vs. reference-by-path,
  and how this generalizes to the Phase 4 final scene, which has its own
  frozen `eval_holdout/` and won't use Lego's val/test at all) that
  belongs to Phase 5's actual pipeline design, not a decision to make
  informally while unblocking one PoC training run.
- **Rationale / evidence:** found and fixed-around (not fixed) while
  setting up this step's training runs; flagged per this step's
  instruction to log it as debt rather than solve it now.
- **Reversibility:** trivial to remove (the symlinks affect nothing
  outside the three `phase3_poc_budget_*` directories); the underlying
  gap in `build_poison_set.py` must be closed before Phase 5/6, not
  reversed.

## D-021 — Phase 4 Step 0: verified Blender/WSL2 bridge, file-location strategy, and pose-pool design

- **Date:** 2026-09-14
- **Context:** D-008 established the *rationale* for the
  Windows-Blender/WSL2-Python split but never actually exercised a
  WSL2-orchestrated headless Blender invocation. This entry records the
  results of doing so for real before any Phase 4 scene work, per this
  project's verify-don't-assume precedent (D-009's `verify_env.py`,
  D-016's CPU-tick diagnosis). All probe work was confined to a scratch
  directory and a Windows temp folder (since removed); no repo file was
  written by the investigation itself.

- **Decision 1 — Blender invocation method (verified working):**
  Blender **5.2.1 LTS** (build date 2026-08-25, Windows release), at
  `/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe`.
  Blender is **not** on the inherited Windows PATH from a WSL2 shell, so
  the explicit `/mnt/c/...` path is required. The canonical invocation
  for this project is:
  ```bash
  "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" \
      --background --factory-startup --python "$(wslpath -w script.py)" -- <args>
  ```
  A WSL2-resident script is passed by converting its path with
  `wslpath -w` into `\\wsl.localhost\...` UNC form; Blender reads it
  without issue. `--factory-startup` is included deliberately so renders
  cannot be contaminated by whatever GUI preferences happen to be saved
  on this particular machine — without it, the render is a function of
  untracked local state, which would break "reproducible from a config +
  a commit hash."

- **Decision 2 — OptiX confirmed engaged (not a CPU fallback):** with
  `compute_device_type='OPTIX'` and `scene.cycles.device='GPU'`, the
  enabled device is `'NVIDIA GeForce RTX 5060 Laptop GPU'`, `type=OPTIX`.
  Available device types on this machine: `NONE, CUDA, OPTIX, HIP,
  ONEAPI`. **Note for `render_scene.py`:** this machine also exposes an
  AMD Radeon 780M iGPU under HIP and the RTX 5060 a second time under
  CUDA, so the device-enable loop must filter on `d.type == 'OPTIX'`
  explicitly rather than enabling every listed device — otherwise work
  can land on the iGPU.
  **Measured timing** (default cube, 800×800, 64 samples, OptiX
  denoising): **~1.8–2.2 s/frame** steady state. Two identical 10-frame
  batches gave means of 1.875 s and 2.236 s — GPU boost/thermal state
  moves the mean by ~20%, which is *larger than any filesystem effect
  measured below*. Phase 4 wall-clock estimates use the slower figure.

- **Decision 3 — file-location strategy: Option A (UNC direct).**
  Blender (the Windows process) writes output directly into the WSL2 repo
  via its `\\wsl.localhost\Ubuntu-24.04\...` path. No separate
  copy/rsync step.
  **Timing evidence.** A first pass comparing whole-render totals
  suggested Option B (Windows-native) was *slower* than Option A, which
  is backwards; re-running A reproduced B's number exactly, showing that
  apparent difference was the thermal effect above, not the filesystem.
  Isolating pure write cost instead — 20 identical 503 KB PNG writes with
  `fsync`, issued from the Windows side:
  | Destination | mean/file | min | max |
  |---|---|---|---|
  | **A:** `\\wsl.localhost\Ubuntu-24.04\...` | **9.6 ms** | 7.3 ms | 14.5 ms |
  | **B:** `C:\Windows\Temp\...` (native) | **1.6 ms** | 1.1 ms | 4.1 ms |
  UNC therefore costs **~8 ms extra per file**. Phase 4 writes ~545
  files, so Option A's total penalty **across the entire phase is ~4
  seconds** — 0.4% of a single frame's render cost, and far below the
  ~20% thermal variance already present in the measurement.
- **Alternatives considered (file location):** Option B, render to a
  Windows-native scratch folder then `robocopy`/`rsync` into the repo —
  rejected. It is ~6× faster per write in isolation but that advantage is
  ~4 seconds in absolute terms across all of Phase 4, and it is bought by
  adding a hand-run step that sits outside any script's control. That is
  precisely the untracked-provenance failure mode already logged as debt
  in D-020 (manually-created symlinks that `build_poison_set.py` would
  not recreate). Paying 4 seconds to keep the render fully
  script-reproducible is the correct trade under this project's ground
  rules.

- **Decision 4 — pose-pool design: three disjoint pools (train 100 / val
  10 / eval_holdout 75).** Investigation of the vendored loader
  established:
  1. `load_blender_data` (`src/nerf/datasets/blender.py`) opens all three
     of `transforms_{train,val,test}.json` **unconditionally** in a single
     `basedir` — there is no "no-val" code path, so a val entry must exist.
  2. Val-subset PSNR monitoring (`src/nerf/training.py`) draws a fixed
     deterministic slice `i_val[:5]` — the first 5 frames of
     `transforms_val.json`. It does **not** structurally require a
     disjoint pose pool; a re-pointed subset of `train/` would work at
     zero extra render cost.
  3. The loader requires **RGBA** input (`images[..., -1:]` is indexed
     for the white-background composite; 3-channel data would crash).
     Verified on a probe render that Blender PNG output with
     `color_mode='RGBA'` yields `(800,800,4)` uint8 with alpha == 255
     everywhere (Film>Transparent off), making the white-background
     composite a mathematical no-op here — so `render.white_background`
     is don't-care for this scene, but `color_mode='RGBA'` is mandatory.
  4. `camera_angle_x` is read from the **last** split iterated (`test`),
     so all three transforms files must agree on it or focal length
     silently comes from the wrong file.
  Despite (2) permitting a two-pool design, **three disjoint pools were
  chosen**: if val frames were drawn from `train/`, then for every
  poisoned condition the val-PSNR monitoring signal would be computed
  partly on poisoned images, making the one live convergence signal
  during a long training run misleading exactly where it matters most.
  ~10 extra renders (~40 s) is a trivial price for a monitoring signal
  that means the same thing in every condition.
- **Alternatives considered (pose pools):** (a) val as a deterministic
  subset of `train/` — rejected per the contamination reasoning above;
  (b) val as a subset of `eval_holdout/` — rejected because it would have
  the training loop reading the frozen holdout on every validation
  interval, which weakens the "the holdout is only ever touched by
  `evaluate.py`" discipline (`METHODOLOGY.md` §4) even though the access
  is read-only.

- **Finding 5 — camera-pose convention: already verified, no conversion
  needed.** Phase 2 explicitly recorded the convention in two places
  (`src/nerf/datasets/blender.py`'s module docstring and
  `experiments/logs/phase2_lego_sanity.md`): right-handed world
  coordinates, per-frame `transform_matrix` is a 4×4 camera-to-world
  matrix, camera looks down its own local −Z with +Y up and +X right.
  **This is Blender's own native camera convention**, so Phase 4's rig
  can write `camera.matrix_world` into `transform_matrix` verbatim with
  no axis conversion. This retires what would otherwise have been Phase
  4's single highest-risk item, and is exactly what `ROADMAP.md` Phase 2's
  "verify the convention matches Phase 4's scene" task existed to buy.

- **Finding 6 — D-020 debt explicitly rescheduled, not resolved here.**
  The Phase 3 handoff asked that D-020 (`build_poison_set.py` does not
  emit a self-contained dataset dir; Phase 3's val/test are hand-made
  symlinks) be resolved *or explicitly rescheduled in writing* before
  Phase 4 got far. **It is hereby rescheduled to Phase 5's first task.**
  Rationale: Phase 4 does not invoke `build_poison_set.py` at any point —
  it produces the clean scene, plates, masks and frozen holdout, all of
  which are upstream of poisoning — so fixing it now would be speculative
  work against a pipeline whose Phase 4 inputs do not yet exist. Phase 5
  is both the first point the script is actually used on real data and
  the phase whose own task list already owns it.

- **Rationale / evidence:** every number above is measured on this
  machine in this session, not assumed — per the explicit instruction to
  verify rather than assume the Windows/WSL2 Blender bridge works.
- **Reversibility:** low cost throughout. The invocation method and
  device filter are a few lines in `scripts/render_scene.py`. The file
  strategy is a single output-path choice; switching to Option B later
  would mean adding a sync step, not re-rendering anything. The pose-pool
  design is the one item with real switching cost: once
  `eval_holdout/` is frozen at Phase 4 Step 6 it cannot change, and
  dropping the separate val pool afterward would mean either regenerating
  the train pool or accepting contaminated monitoring — so it is settled
  now, before any render, deliberately.

## D-022 — Phase 4 Step 2: final scene locked; background-plate method; shadow behaviour of mask-limited erasure

- **Date:** 2026-09-14
- **Status:** these are the **locked** scene choices, recorded after the
  Step 2 human preview review passed (`ROADMAP.md` Phase 4). Everything
  below is defined in `configs/scenes/final_scene.yaml` and built by
  `scripts/build_scene.py` → `src/data_pipeline/blender_build_scene.py`;
  nothing here is hardcoded in a `.py` file, and
  `data/raw/tabletop_diorama.blend` is a regenerable build artifact, not a
  hand-modelled asset (README.md's reproducibility ground rule, which is
  why the scene brief mandated a procedural build over GUI modelling).

### Decision 1 — locked scene content

Tabletop diorama, all objects opaque diffuse Principled BSDF (metallic 0,
transmission 0 — the brief forbids glass/mirror/metal because they break
object-ID mask cleanliness and make soft suppression ambiguous):

| Role | Shape | Colour | Placement |
|---|---|---|---|
| **Target** `Target_Mug` | tapered cylinder body (r 0.105→0.120, h 0.260) + torus handle (major 0.075, minor 0.030) | red | origin, `pass_index = 1` |
| `Distractor_Book` | box 0.22×0.16×0.06, 15° | blue | ring r 0.70, 40° |
| `Distractor_Ball` | sphere r 0.110 | green | ring r 0.70, 130° |
| `Distractor_Cone` | cone r 0.130, h 0.260 | yellow | ring r 0.70, 215° |
| `Distractor_Ring` | torus (major 0.100, minor 0.035) lying flat | tan | ring r 0.70, 310° |

Table: disc r 6.0, albedo 0.34. Backdrop: open-topped cylinder r 5.0,
h 4.0, albedo 0.52. Key light: area, size 1.4, **90 W**, at
(1.60, −1.30, 2.60). World ambient strength 0.12. Camera: 50 mm lens /
36 mm sensor (39.6° FOV), **radius 2.9 m**, elevation 25°–75°.

Three choices here are derived rather than picked, and are recorded so a
later reader does not "simplify" them back into a broken state:

1. **Camera radius 2.9 m, not 4.0 m.** At 4.0 m the target subtends only
   ~0.76% of frame area — *below* the visibility threshold in Decision 3,
   which would have made `V_target` collapse. 2.9 m is the closest radius
   that still frames the full 1.70 m diorama.
2. **Handle tube thickness (minor_radius 0.030) is forced, not aesthetic.**
   With `body_r` = body radius at handle height, `overlap = body_r −
   (offset − major − minor)` and `hole_gap = (offset − major + minor) −
   body_r`, so **`overlap + hole_gap ≡ 2 × minor_radius`**. Both must be
   adequate: too little overlap and the handle renders as a detached
   floating ring; too little hole_gap and the hole is buried inside the
   body and there is no handle hole at all. No choice of `major_radius` or
   `offset` can rescue a thin tube. At minor 0.030 with
   `offset = major + body_r`, both are 0.030 m (~11 px), so the hole
   (diameter ~34 px) survives `METHODOLOGY.md` §3's 3–5 px dilation.
3. **Backdrop albedo 0.52, not 0.72.** The backdrop encloses the scene, so
   a bright value turns it into a giant softbox whose interreflection
   erases the key light's contact shadows entirely.

- **`Distractor_Can` (cylinder) was replaced by `Distractor_Ring` (torus)**
  during this review: a cylinder is silhouette-identical to the target's
  mug body, so from poses where the handle is occluded or foreshortened
  the two differed only in hue. The scene brief requires the target be
  visually/semantically distinct from the distractors; a flat-lying torus
  is unambiguous from every pose in the rig.

### Decision 2 — background-plate method: `hide_render`

The plate is rendered with the target's **`hide_render = True`** (removed
from camera rays, shadow rays and indirect/AO bounces), *not* with
camera-visibility-only toggling. A camera-visibility plate leaves the
target's cast shadow and contact AO on the table with no object above
them, which is a compositing artifact rather than ground truth: a plate is
supposed to represent the scene as if the target had never been there
(D-003).

### Decision 3 — thresholds (locked BEFORE `V_target` is computed)

- **Per-pixel mask binarisation:** not a tunable. Blender's ID Mask pass
  with anti-aliasing off is **exactly binary** — measured on real preview
  masks, exactly 2 unique values and no intermediate edge band at all.
  This is strictly better than D-017's Lego alpha proxy, which had 1.45%
  intermediate-alpha pixels and needed a real threshold decision.
- **Per-view minimum visibility (`METHODOLOGY.md` §2):**
  **0.005 (0.5% of frame pixels = 3,200 px of 640,000).**
  Measured target area across the rig extremes is **1.62%–2.35%**, i.e. a
  **3.2×–4.7× margin** — structural, not marginal. A 1.0% candidate was
  **rejected** because the same geometry would clear it by only ~1.4×,
  leaving `V_target` sensitive to per-pose foreshortening. Locked now, in
  config, before `V_target` is computed, and never tuned afterwards (the
  D-013 discipline).

### Finding 4 — shadows survive erasure; this is mask-limited compositing, not a defect

**This is the substantive methodological finding of Step 2 and the reason
Decision 2 alone is not sufficient.** Recorded in full so it is not
rediscovered as a surprise during Phase 8 write-up.

`METHODOLOGY.md` §3's formulas read the plate **only inside the mask**:

```
poisoned = mask * background_plate + (1 - mask) * original
```

The target's cast shadow falls on the table **outside** the target's
silhouette, therefore outside the mask, therefore the poisoned pixel there
is taken from `original` and the shadow is retained **regardless of which
plate method is used**. Switching the plate to `hide_render` changes only
pixels *inside* the silhouette; it cannot remove a shadow lying outside
it. So "object gone, shadow remains" is a property of **mask-limited
compositing**, not of the plate's content.

Measured (preview triple, az 60° / el 40°): the original-vs-plate
difference region is ~3.7% of frame — **larger than the 2.1% target
itself** — and it does not decay with dilation (3.74% of outside-mask
pixels differ at 0 px dilation, still 1.67% at 45 px, max magnitude 82/255).
Dilation is structurally the wrong instrument: the difference is a cast
shadow stretching away across the table, not a rim around the silhouette.

- **Decision: Option A — `METHODOLOGY.md` §3 is kept EXACTLY as written.**
  No deviation-log entry is required, because nothing in §3 changes.
- **Why this does not affect either metric category as currently defined:**
  the primary metrics (§6) are *masked* PSNR/SSIM/LPIPS inside the target
  region and *unmasked* metrics over the rest of the frame, both computed
  against the **true clean render**. The shadow is present in the clean
  render and present in the poisoned training images alike, so it is
  simply consistent scene content that the model is expected to
  reconstruct; it is not an error signal in the masked (suppression)
  region, and in the unmasked (collateral) region it is identical between
  control and poisoned conditions, so it cannot inflate or deflate the
  collateral-damage curve. The secondary plate-distance metric is computed
  in the target region only, which is exactly where the plate *is* used.
- **What it does mean scientifically:** hard erasure leaves a residual
  cue — the object is gone but its shadow is not. This is an honest and
  realistic limitation of any mask-limited attack (a real attacker
  inpainting a masked region faces the same constraint), and should be
  reported as such rather than engineered away.
- **Alternatives considered:** (b) redefining the mask as the union of the
  object-ID mask and the shadow-difference region — rejected for the core
  study: it would change §3's "raw Blender object-ID mask" definition
  after the protocol was written, converting a pre-registered choice into
  a post-hoc one, and it would make the attack strictly stronger than the
  one specified. (c) lighting the scene to suppress shadows — rejected:
  contact shadows were added deliberately, they are needed for the plate
  method to be meaningful at all, and a shadowless scene gives the NeRF
  less geometric grounding.
- **Required follow-up before Step 6's freeze:** add a short clarifying
  note to `METHODOLOGY.md` §3 (or §9) stating that shadow retention under
  mask-limited erasure is a known, expected property of the specified
  formulas and not a pipeline defect — so that it reads as pre-registered
  rather than discovered after the fact.

### Finding 5 — the target's shadow does touch distractors, negligibly

Checked explicitly before locking, over a **48-pose sample** (16 azimuths
× elevations 25°/50°/75°) at production settings, by giving all
distractors a shared `pass_index = 2` so one ID Mask yields their combined
footprint, then testing which pixels *belonging to a distractor* change
between the original and the `hide_render` plate.

**Answer: yes, but negligible.**

| difference magnitude | total px across all 48 poses | poses affected |
|---|---|---|
| > 8 / 255 (faint indirect bounce) | 2,651 | 42 / 48 |
| > 20 / 255 | 178 | 8 / 48 |
| > 30 / 255 (genuine shadow contact) | 118 | 8 / 48 |
| > 50 / 255 | 43 | 5 / 48 |

Worst single pose (az 112°, el 25°): 125 crossing pixels = **0.56% of the
visible distractor area** and **0.025% of the frame**, max magnitude
68/255. So the near-ubiquitous case (42/48 poses) is faint indirect colour
bleed at ~10/255, and true shadow contact occurs in 8 of 48 poses totalling
118 px across the entire sample.

Documented rather than treated as blocking, for the same reason as
Finding 4: these pixels lie outside the target mask, so the compositor
never writes to them and poisoned images are unaffected. It is recorded so
that a later observer comparing a plate against its original does not
mistake this for misregistration.

### Decision 6 — two settings promoted into config, one constraint on Step 4

- **View transform `Standard`** (not AgX/Filmic) is now
  `render.view_transform` in `configs/scenes/final_scene.yaml` and applied
  by the build script, rather than living only in a preview script. It
  changes every pixel of every training image and every metric computed
  from one, so it belongs in the config per `PROJECT_STRUCTURE.md`.
- **Masks are 16-bit BW PNG** (values 0 and 65535), recorded as
  `render.mask_color_depth`. **Step 4's binarisation code must assert the
  bit depth and threshold at half of the dtype max.** D-017's Lego
  `alpha > 127` assumes uint8 and would silently classify *every* pixel of
  a 16-bit mask as foreground — a failure that produces a plausible-looking
  full-frame mask rather than an error.

### Three bugs found and fixed during Step 1/2

Logged because each failed silently rather than raising, which is the
failure class this project's rules exist to catch (cf. D-011, D-014, D-015):

1. **Key light aimed upward.** The hand-rolled Euler aim used
   `acos(dz/r)` where aiming at the origin requires `acos(-dz/r)` — 141.6°
   instead of 38.4°. Every render was lit purely by world ambient and
   backdrop bounce. It presented as "flat, shadowless lighting" and barely
   responded to energy changes (900 → 9000 W), which is what exposed it.
   Replaced with `Vector.to_track_quat('-Z','Y')`. **The correct energy is
   ~10× lower (90 W) than the drafts that were unknowingly compensating
   for it** — anyone reading the config's small energy value should not
   "correct" it upward.
2. **Blanket `shade_smooth`** on every object rounded the cone tip,
   cylinder caps and box edges into blobs. Replaced with angle-based
   `shade_auto_smooth`.
3. **Handle buried inside the body** — see Decision 1, item 2.

- **Reversibility:** the scene is regenerable from config + commit at any
  time, so pre-freeze changes are cheap. After Step 6 freezes
  `data/blender_scenes/eval_holdout/`, changing scene content would
  invalidate the eval set and every downstream result — so these choices
  are locked here, deliberately, before the rig and batch render are
  built on top of them.

## D-023 — Phase 4 Step 3: camera rig, pose-pool partition, and measured near/far

- **Date:** 2026-09-14
- **Decision:** the rig is **185 poses** on a Fibonacci lattice over the
  elevation band 25°–75° at radius 2.9 m, partitioned by a seeded shuffle
  (`partition_seed: 0`) into **train 100 / val 10 / eval_holdout 75**.
  Generated by `scripts/build_camera_rig.py` →
  `src/data_pipeline/blender_build_rig.py`, which writes
  `data/blender_scenes/cameras.json` plus the three
  `transforms_{train,val,test}.json` files.
- **Pose generation:** `sin(elevation)` is sampled uniformly over the band
  (equal-area, so the rig is not pole-heavy) with a golden-angle azimuth
  (low-discrepancy, no repeated azimuths). **All three pools are cut from
  one lattice**, so `METHODOLOGY.md` §4's requirement that eval views be
  disjoint from every training view holds **by construction**, not by
  post-hoc checking. Verified anyway: 185/185 unique positions, zero
  overlap between any pair of splits.
- **Why the rig runs inside Blender:** `transform_matrix` is taken verbatim
  from `camera.matrix_world`. Phase 2 established that the loader's
  convention *is* Blender's native camera convention (D-021 finding 5), so
  reading the matrix straight out is correct by construction; re-deriving
  the same matrix in numpy would only create an opportunity to get it
  subtly wrong. Validated: all 185 rotation matrices orthonormal with
  det +1, all radii exactly 2.900000 m, and every camera's local −Z
  forward vector aligns with the look-at direction to 1e−7.
- **`camera_angle_x` = 0.6911112 rad (39.60°)**, written **identically**
  into all three transforms files — required because the loader takes this
  value from whichever split it happens to read last (D-021 finding 4).
- **`cameras.json` schema:** none existed anywhere in the repo —
  `PROJECT_STRUCTURE.md` lists the file and `METHODOLOGY.md` §2 says
  `|V_target|` and the threshold live in its metadata, but the format was
  undefined. Defined here as `schema_version: 1`, with blocks for
  `camera` (including an explicit written-out statement of the pose
  convention), `rig` (method, radius, band, counts, seed), `depth_probe`,
  `v_target`, and a `poses` array carrying per-pose split, file_path,
  azimuth/elevation, location and transform_matrix. The `v_target.count`
  field is deliberately **null** at this point: the threshold is locked
  now, the count is a Step 5 deliverable computed from real data.
- **Naming note:** the loader requires `transforms_test.json` while
  `PROJECT_STRUCTURE.md` names the directory `eval_holdout/`. Both are
  honoured — the file is `transforms_test.json`, its `file_path` entries
  point into `eval_holdout/`.

### near/far are MEASURED, and the inherited values were both wrong

- **Decision:** `dataset.near: 1.5`, `dataset.far: 9.0`.
- **Evidence:** a Cycles Z-pass probe over **all 185 rig poses**
  (`--measure-depth`) gives an actual scene depth range of
  **1.806 m .. 8.346 m**.
- **Why this matters:** the config had been carrying Lego's `near: 2.0,
  far: 6.0`, and **both** are wrong for this scene. `near: 2.0` sits
  *inside* the measured minimum, so it would have clipped the nearest
  table surface; `far: 6.0` falls well short of the 8.346 m far backdrop
  wall, so the NeRF would have had no sampling range in which to place the
  background at all. Neither failure would have raised an error — they
  would have shown up as unexplained reconstruction artifacts during
  Phase 6, after the eval set was already frozen.
- **Alternatives considered:** shrinking the backdrop cylinder (radius
  5.0 → ~3.5) to tighten the depth range — rejected, it would put the wall
  visibly close behind the subject at low elevations and change the locked
  scene content from D-022 for no scientific gain. Accepting a wider
  sampling range (8.1 m here vs Lego's 4.0 m, so coarser spacing per
  sample) is the better trade; if Phase 6 convergence proves poor, this is
  a documented place to look first.
- **Reversibility:** near/far are config values with no effect on rendered
  data, so they can be changed up until Phase 6 training begins without
  touching the frozen eval set. The rig itself, once Step 6 freezes
  `eval_holdout/`, cannot change.

## D-024 — Phase 4 Step 4a pilot: denoiser bleed cleared, mask dilation set to 3 px

- **Date:** 2026-09-14
- **Scope:** this entry closes the two questions the Step 4a pilot existed
  to answer, and is written **before** the full Step 4 batch render, as
  required. It sets the last unfilled value in
  `configs/scenes/final_scene.yaml`
  (`render.background_plate.mask_dilation_px`).
- **Evidence base:** rather than the planned 2-view pilot, the measurement
  ran over a **48-pose set** (16 azimuths × elevations 25°/50°/75°) already
  rendered at production settings (800×800, 128 samples, OptiX denoising,
  fixed `cycles.seed = 0`, `hide_render` plates) during the Finding-5 check
  in D-022. More poses for no extra render cost.

### 1. OptiX denoiser bleed: NOT a problem

Measured max `|original − plate|` at more than 150 px from the target mask,
across all 48 poses: **12/255**, and that residual is real cast shadow, not
denoiser spill. With an identical `cycles.seed` between a view's original
and plate render, the denoiser does **not** smear the plate's differing
light paths across the frame. No mitigation needed; denoising stays ON and
the sample count stays at 128. (The plan's fallback — disabling denoising
and raising samples — is not required.)

### 2. Mask dilation: **3 px** (the low end of `METHODOLOGY.md` §3's
pre-registered 3–5 px candidate range, so this is NOT a deviation)

Two independent measurements set this, and they push in opposite
directions — which is what makes 3 the answer rather than an arbitrary pick.

**(a) Lower bound — the anti-aliased silhouette fringe.** The ID mask is
exactly binary, but the *rendered image* is anti-aliased, so silhouette
pixels are object/background blends. Measuring residual target **chroma**
(`R − (G+B)/2`, which isolates leftover red object pixels from shadow, a
luminance-only change) just outside the dilated mask after a hard-erasure
composite:

| dilation | max residual chroma | mean over poses |
|---|---|---|
| 0 px | 93.0 | 65.42 |
| 1 px | 22.5 | 18.02 |
| 2 px | 21.0 | 16.51 |
| 3 px | 21.0 | 15.95 |
| 5 px | 21.0 | 15.14 |
| 8 px | 18.0 | 13.85 |

The AA fringe is gone by 1–2 px (93 → 22.5 → 21). **Everything beyond that
is a plateau that dilation cannot remove**, because it is not an edge
artifact: residual chroma decays smoothly with distance from the mask
(mean 4.01 at 3–6 px, 3.50 at 6–12, 2.66 at 12–25, 1.65 at 25–50, 0.82 at
50–100), which is the signature of **diffuse red colour bleed** — the red
mug tinting the nearby table via indirect light. Removing the target
removes that tint. This is the same phenomenon class as D-022 Finding 4
(shadow retention) and is accepted under the same Option A reasoning: it
lies outside the mask, so the compositor never writes there.

**(b) Upper bound — the handle hole.** D-022 committed to verifying that
the handle hole survives dilation. **It does not survive 5 px**, and the
D-022 estimate ("~34 px hole, ~24 px after 5 px dilation") was wrong: it
assumed a face-on circular hole, whereas at most poses the handle is
foreshortened and the hole is a narrow ellipse. Measured over all 48 poses
(the hole is geometrically visible in 20 of them; at the other 28 the
handle is edge-on or occluded, which is expected):

| dilation | poses with hole still open | poses where it CLOSES | median hole px |
|---|---|---|---|
| 0 px | 20/20 | 0 | 336 |
| 2 px | 16/20 | 4 | 161 |
| 3 px | 16/20 | 4 | 92 |
| 4 px | 14/20 | 6 | 38 |
| 5 px | 10/20 | **10** | 12 |

At 5 px the hole closes in **half** of the poses where it exists; at 3 px,
in 4 of 20.

- **Decision: 3 px.** It is the smallest value that fully covers the AA
  fringe while staying inside the pre-registered range, and it preserves
  the handle hole in 16/20 poses versus 10/20 at 5 px. Going above 3 px
  buys nothing measurable (residual chroma is identical at 3 and 5 px)
  and costs hole fidelity.
- **Alternatives considered:** 1–2 px — outside `METHODOLOGY.md` §3's
  pre-registered range, so choosing it would be a protocol deviation
  requiring its own justification, for a benefit (4 more poses keeping an
  open hole) that is marginal. 5 px — rejected on the hole evidence above.
- **Documented residual:** in 4 of 20 poses the 3 px dilation closes the
  handle hole, meaning a small patch of table seen *through* the handle
  (30–90 px, under 0.015% of frame) is treated as target and replaced by
  plate. Recorded so it is not later mistaken for misregistration.
- **Reversibility:** a single config value with no effect on rendered data
  — the masks and plates on disk are unchanged by it, since dilation is
  applied at compositing time (Phase 5), not at render time. It can be
  revisited up until the first poisoned set is built, without re-rendering.

## D-025 — Phase 4 Step 5: batch render complete, |V_target| = 100, gate PASS

- **Date:** 2026-09-14
- **Batch render:** 360 renders (185 originals + 175 masks + 175 background
  plates = **535 files**) in **14.6 minutes**, zero errors. This is inside
  the 25–50 min estimate, and the per-render rate (~2.4 s) matches the
  Step 0 probe on the default cube — the real scene costs no more than the
  trivial one at these settings. File counts exact: train 100 / val 10 /
  eval_holdout 75 originals; masks and plates for train + eval_holdout
  only (val is monitoring-only, D-021).
- **`|V_target|` = 100 of 100 training views.** Per-view target mask area
  over the training set: **min 1.47%, median 1.85%, max 2.18%** of frame,
  against the threshold of 0.50% locked in D-022 before any of this was
  computed. **Smallest margin over threshold: 2.95×.**
  Every training view clears it, which is the intended structural outcome
  of placing a moderately-sized target centrally and aiming every rig pose
  at it — not a coincidence, and not the result of tuning the threshold.
  Recorded in `data/blender_scenes/cameras.json` under `v_target`
  (count, per-view area fractions for both train and eval_holdout, and the
  threshold), per `METHODOLOGY.md` §2's requirement that it live there and
  never be recomputed mid-study.

### Phase 4 gate (ROADMAP.md): **PASS**

| check | result |
|---|---|
| target separable from background by mask | **PASS** — all 175 masks binary (uint16, exactly {0, 65535}), none empty, none full-frame |
| `round(0.05 × \|V_target\|) ≥ 1` | **PASS** — equals **5** |
| ...and lands on several views, not exactly 1 | **PASS** — 5, per the brief's requirement |
| all validation checks clean | **PASS** — zero errors across 185 views |

Resulting budget ladder (`METHODOLOGY.md` §5), all whole numbers with no
awkward rounding: 5% → 5 views, 10% → 10, 20% → 20, 30% → 30, 50% → 50.

### Validation detail

- **Mask bit depth asserted, not assumed** — all masks `uint16` with
  exactly two unique values. The reader raises on any other dtype or on a
  non-binary mask (D-024's concern about inheriting D-017's uint8 `>127`).
- **Far-field original-vs-plate agreement: max 13/255** (worst view
  `eval_holdout/r_065`), against a bound of 20. This is the check that
  replaced Phase 3's strict outside-mask pixel identity, which is **not**
  valid under `hide_render` plates: the target's shadow and colour bleed
  legitimately differ outside the mask (D-022 Finding 4, D-024). Far-field
  agreement is what actually proves the fixed `cycles.seed` is holding and
  that the two renders of a view are registered to each other.
- **Handle hole under the locked 3 px dilation:** open in 68 views, closed
  in 23 (of the 91 views where the hole is geometrically visible at all).
  Consistent with D-024's 48-pose sample and well better than the 5 px
  behaviour that ruled out the top of the range.
- **Loader round-trip on the real scene:** `load_blender_data` returns
  `(185, 400, 400, 4)` float32 RGBA at `half_res`, split counts exactly
  **100 / 10 / 75**, focal **555.556** — matching the value derived
  independently from the 50 mm lens / 36 mm sensor / 800 px geometry
  (`0.5·800/tan(0.5·camera_angle_x)/2`) to three decimals. The alpha
  channel is present and fully opaque, so the loader's white-background
  composite is the no-op predicted in D-021.
- **Reversibility:** none of this is a design decision — it is the measured
  outcome of the choices locked in D-022/D-023/D-024. `|V_target|` must not
  be recomputed after this point (`METHODOLOGY.md` §2); if the scene or rig
  ever changed, every downstream budget number would change with it, which
  is precisely what the Step 6 freeze exists to prevent.

## D-026 — Phase 4 Step 6: held-out evaluation set FROZEN; METHODOLOGY.md frozen

- **Date:** 2026-09-14
- **Decision:** the held-out evaluation set is frozen and `METHODOLOGY.md` is
  locked. **This is the point of no return for Phase 4.** It was done only
  after the Step 5 gate returned a genuine PASS on all four checks (D-025),
  not on a "mostly worked".

### What is frozen — deliberately wider than ROADMAP.md's literal wording

`ROADMAP.md` Phase 4 says "freeze `data/blender_scenes/eval_holdout/`".
Taken literally that would freeze the held-out *images* while leaving
mutable the masks, plates and poses that every held-out metric is actually
computed against — satisfying the rule's letter while defeating its
purpose. Frozen set is therefore:

| path | why it must be frozen |
|---|---|
| `data/blender_scenes/eval_holdout/` | the held-out rendered views |
| `data/masks/eval_holdout/` | `METHODOLOGY.md` §6's **masked** metrics are computed with these |
| `data/background_plates/eval_holdout/` | §6's secondary plate-distance metric is computed against these |
| `data/blender_scenes/transforms_test.json` | the camera poses that *define* the held-out set |

**226 files total.**

### Checksums

- **Aggregate SHA-256:
  `211a5a59d85cf29447a7608e74bbe49ebe08717709d3c520bc3af549d65ab214`**
- Per-file manifest: `data/blender_scenes/eval_holdout_SHA256SUMS.txt`
  (tracked in git, so the checksums live in version control rather than only
  alongside the data they describe).
- Re-checkable at any time with
  `python scripts/freeze_eval_set.py --verify`, which reports missing,
  unexpected and changed files as well as the aggregate.
- `scripts/freeze_eval_set.py --freeze` **refuses to run if the manifest
  already exists**, so the freeze cannot be silently re-taken over modified
  data at a later date — re-freezing would otherwise produce a fresh,
  self-consistent checksum that hid the change.

### Write protection — verified by attempting writes, not asserted

All frozen files are mode `r--r--r--` and the frozen directories are
`dr-xr-xr-x`. Five write attempts were made and **all five failed**:

| attempt | result |
|---|---|
| append to a held-out view | blocked (Permission denied) |
| append to a held-out mask | blocked (Permission denied) |
| append to `transforms_test.json` | blocked (Permission denied) |
| create a new file inside the frozen directory | blocked |
| delete a held-out view | blocked |

The directories are made read-only as well as the files specifically so
that **deletion** is blocked: POSIX delete permission comes from the parent
directory, so read-only files alone would still have allowed `rm`. After
the attempts, the set still contains its 75 held-out views and
`--verify` reports byte-for-byte identity.

### METHODOLOGY.md frozen

- Status line changed from "to be frozen at the end of Phase 4" to
  **"FROZEN 2026-09-14"**, carrying the aggregate checksum of the eval set
  it is measured against.
- §10's deviation log now reads "none — followed exactly", with an explicit
  note that the §3 "Scope of the edit" paragraph is **not** a deviation: it
  was added pre-freeze during Step 2 and states a property the §3 formulas
  always had, without changing the formulas.
- Two values that were still written as *candidates* were replaced with the
  decided values before freezing, so the frozen protocol is unambiguous:
  the mask dilation (**3 px**, D-024) and the minimum-visibility threshold
  and resulting count (**0.005**, **`|V_target|` = 100**, D-022/D-025).
  Leaving a frozen protocol reading "candidate 3–5 px" would have left a
  free parameter inside a document whose entire purpose is to have none.
- **From this date, any change to §1–§7 requires both a `DECISION_LOG.md`
  entry and a line in §10's deviation log.**

- **Reversibility: none, by design.** Unfreezing and re-rendering the
  held-out set would invalidate every metric computed against it and, per
  `README.md`'s ground rules, would mean the experiment must be rerun. If a
  defect is ever found in the eval set, the correct response is to document
  it and restart the affected phases — not to quietly regenerate the data.

## D-027 — Phase 4 closeout: end-to-end reproducibility verified; renders are NOT bit-exact

- **Date:** 2026-09-14
- **What was done:** the closeout reproducibility check was **actually
  executed**, not asserted. From the committed config alone, into a scratch
  data root (so it could not touch the frozen set): rebuilt the `.blend`,
  regenerated the camera rig, re-rendered all 360 images, then compared
  against the frozen manifest and re-ran the gate.

### Result: reproducible in every number that matters, but NOT byte-for-byte

| artifact | result |
|---|---|
| `.blend` rebuild | PASS |
| camera rig | PASS — 185 poses, train 100 / val 10 / eval_holdout 75 |
| `transforms_test.json` | **byte-identical** |
| masks (75 held-out) | **pixel-identical** — max abs diff **0** |
| originals (75 held-out) | **NOT identical** — max abs diff **1/255**, on 0.0028% of pixels |
| background plates (75) | **NOT identical** — max abs diff **1/255**, on 0.0028% of pixels |
| SHA-256 manifest vs regenerated copy | **FAIL — 225 of 226 files differ** |
| loader round-trip on regenerated copy | PASS — 100/10/75, focal 555.556 |
| recomputed `\|V_target\|` | **PASS — 100, margin 2.95x, gate PASS. Identical to the frozen value.** |

- **The plan's stated success criterion — "the checksum manifest reproduces
  byte-for-byte" — is FALSE and is corrected here.** Cycles + OptiX path
  tracing with GPU denoising is not bit-deterministic across runs:
  floating-point reduction order across parallel threads varies, so a
  re-render lands within one 8-bit quantisation step on a small fraction of
  pixels. A fixed `cycles.seed` fixes the *sample pattern*, not the
  *summation order*. This was assumed rather than checked when the plan was
  written; running it is what exposed it.

- **Magnitude, in context:** ~18 pixels of 640,000 per image differ, each by
  exactly 1/255. That is below the quantisation floor of the stored 8-bit
  PNGs and orders of magnitude below any PSNR/SSIM/LPIPS difference this
  study reports (Phase 3's effect sizes were measured in whole dB).

- **What this does and does not mean:**
  - It does **not** weaken the freeze. `scripts/freeze_eval_set.py --verify`
    against the actual frozen data still reports byte-for-byte identity —
    the manifest's real job is detecting modification or corruption of the
    frozen copy, and it does that exactly.
  - It does **strengthen** the case for freezing rather than regenerating:
    since a re-render is not bit-identical, the frozen set is the single
    authoritative artifact and must never be regenerated, which is already
    `METHODOLOGY.md` §4's rule. Had this study relied on "just re-render it
    if needed", every regeneration would have silently shifted the
    evaluation data underneath the results.
  - Everything the paper actually reports as a number reproduces **exactly**:
    `|V_target|`, the per-view mask areas, the mask pixels themselves, the
    camera poses, and the gate outcome.

- **Honest claim for the write-up:** the pipeline is reproducible from a
  config file + a git commit hash to within **±1/255 per pixel on ~0.003% of
  pixels** for RGB renders, and **exactly** for poses, masks and every
  derived quantity. It is not bit-exact, and should not be described as such.
- **Reversibility:** n/a — a measured property of the toolchain, not a
  decision. Revisit only if Blender/OptiX ever guarantees bit-exact output,
  which would allow tightening the claim.
