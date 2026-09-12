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
