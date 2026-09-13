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
