# Methodology (Locked Protocol)

**Status: FROZEN 2026-09-14** (end of Phase 4 — final scene built, eval set
rendered, checksummed and set read-only; see `DECISION_LOG.md` D-026).
From this date, any change to §1–§7 requires a dated entry in
`DECISION_LOG.md` explaining what changed and why, **and** a corresponding
line in §10's deviation log below — silent changes here are how "the
poisoning-budget curve" quietly turns into "whatever curve happened to look
good," and that's exactly the failure mode this document exists to prevent.

The frozen held-out evaluation set this protocol is measured against has
aggregate SHA-256
`211a5a59d85cf29447a7608e74bbe49ebe08717709d3c520bc3af549d65ab214`
(226 files; manifest at `data/blender_scenes/eval_holdout_SHA256SUMS.txt`,
re-checkable with `python scripts/freeze_eval_set.py --verify`).

This document exists because the single biggest threat to this project's
credibility is not weak results — negative or moderate results are fine and
are discussed in §8 — it's **an unprincipled, moving protocol** that lets a
reviewer argue the numbers were fitted to a story after the fact. Everything
below is decided *before* the budget sweep is run, using the reasoning
already validated by literature review (see `DECISION_LOG.md` entries D-002,
D-003, D-004), not adjusted afterward based on how a given condition turns out.

## 1. Research question

> What is the minimum fraction of poisoned training views required to cause
> reliable target-object suppression in a NeRF, and what collateral
> degradation does that poisoning introduce to non-target scene content?

Corollary question answered by the ablation (§7): does *which* views are
poisoned (random vs. strategically chosen) matter as much as *how many*?

## 2. Poisoning-budget definition

**The independent variable is: percentage of *target-visible* training views
that are poisoned — not percentage of all training views.**

Rationale (do not revisit without new evidence): if the target object appears
in only a subset of training images, "poisoning 20% of all views" is an
ill-defined quantity — it could mean anywhere from 0% to a much larger
fraction of the views that actually show the object, depending on which views
happen to be sampled. Two runs both labeled "20%" would not be measuring the
same intervention. Anchoring to target-visible views means every number on
the x-axis means the same thing across conditions and across any future
scene.

Formally, for a training set with views `V`, target-visible views
`V_target ⊆ V` (any view in which the target object's mask covers ≥ a fixed
minimum pixel-area threshold, set once in `configs/scenes/final_scene.yaml`
and never tuned per condition), and poisoning budget `b ∈ {0, 5, 10, 20, 30,
50}` (percent):

```
num_poisoned = round(b / 100 * |V_target|)
poisoned_views = sample(V_target, num_poisoned)   # method fixed per condition, see §7
```

`|V_target|` and the minimum-visibility threshold are recorded in
`data/blender_scenes/cameras.json` metadata once the final scene is built and
are never recomputed mid-study.

**Frozen values for this study:** minimum-visibility threshold **0.005**
(0.5% of frame pixels), locked before the count was computed; **`|V_target|`
= 100** of 100 training views (per-view target mask area 1.47%–2.18% of
frame, smallest margin 2.95x over threshold). The resulting budget ladder is
5% → 5 views, 10% → 10, 20% → 20, 30% → 30, 50% → 50. See `DECISION_LOG.md`
D-022 (threshold) and D-025 (measured count).

## 3. Attack conditions

Two poisoning mechanisms, applied by a single deterministic script
(`src/poisoning/compositor.py`) so there is zero manual judgment per image:

**Hard erasure** (baseline attack — explicitly framed as a weak baseline in
the paper, not the contribution):

```
poisoned_pixel = mask * background_plate + (1 - mask) * original
```

**Soft suppression** (the condition that answers "you just deleted the
object" — the object remains faintly present, only degraded):

```
poisoned_pixel = mask * (alpha * original + (1 - alpha) * background_plate) + (1 - mask) * original
```

- `alpha` is fixed once (candidate range 0.4–0.5; exact value recorded in
  `configs/poisoning/soft_suppression_20.yaml` when set) and held constant
  across every soft-suppression image in the study. It is never varied per
  image or per budget level. Intensity-as-a-variable is an explicitly
  out-of-scope future ablation (§9).
- Mask source is identical for every view: raw Blender object-ID mask, one
  fixed dilation amount decided once and applied uniformly. **Decided value:
  3 px** (`render.background_plate.mask_dilation_px` in
  `configs/scenes/final_scene.yaml`), set from the Step 4a pilot's evidence
  before any poisoned set was built — see `DECISION_LOG.md` D-024. The
  Blender object-ID mask is exactly binary (no anti-aliased edge band), so
  no per-pixel binarisation threshold is needed or used.
- Both formulas are implemented as pure pixel arithmetic on three
  Blender-rendered, pose-aligned images (`original`, `background_plate`,
  `mask`) — no inpainting, no generative fill, for the synthetic scene.

**Scope of the edit (expected property, not a defect).** Both formulas read
`background_plate` **only inside the mask**; outside it, the poisoned pixel is
the original, unchanged. Effects the target has on the scene *beyond its own
silhouette* — its cast shadow, its contact ambient occlusion, and the diffuse
colour it bounces onto nearby surfaces — therefore survive poisoning, because
those pixels lie outside the mask. A poisoned view shows the object removed but
its shadow still present. This is an inherent, intended consequence of defining
the attack as a mask-limited edit, not a rendering bug, a misregistration, or a
plate-generation error, and it is deliberately **not** engineered away: a real
attacker editing a masked region faces exactly the same constraint, so removing
it would model an unrealistically strong adversary. It is also why the attack
is framed (§3, above) as a deliberately weak baseline rather than the
contribution. The residual shadow is a legitimate detection cue and should be
reported as such. Measurements of the effect on this study's scene, and the
reasoning behind keeping §3 unchanged, are in `DECISION_LOG.md` D-022
(Finding 4) and D-024.

## 4. Evaluation views

A fixed set of held-out camera poses, disjoint from all training views (clean
and poisoned), is rendered once during Phase 4 and used for every condition's
evaluation, unmodified. This set is what "held-out" means in every results
table in this project — it is never re-rendered, re-sampled, or adjusted after
the first poisoning run.

## 5. Poisoning conditions matrix (minimum defensible experiment)

| Condition | Budget (% of `V_target`) | Attack type | View selection |
|---|---|---|---|
| Control | 0% | — | — |
| C1 | 5% | Hard erasure | Random |
| C2 | 10% | Hard erasure | Random |
| C3 | 20% | Hard erasure | Random |
| C4 | 30% | Hard erasure | Random |
| C5 | 50% | Hard erasure | Random |
| C6 | 20% | Soft suppression | Random |
| C7 (ablation) | 20% | Hard erasure | Strategic (max target visibility) |

7 conditions + control = **8 training runs minimum**, per §6 of ROADMAP.md.

## 6. Metrics

**Primary:**
- Masked PSNR/SSIM/LPIPS in the target-object region on held-out views
  (target-suppression signal) — computed against the *true clean render*,
  not the poisoned/edited image, so the metric measures reconstruction
  fidelity to ground truth, not similarity to the attack.
- Unmasked (whole-frame, target region excluded) PSNR/SSIM/LPIPS on held-out
  views (collateral-damage signal).

**Secondary:**
- Distance of the poisoned model's rendered target region to the known
  `background_plate` ground truth (a direct, interpretable "did suppression
  actually replace the object with plausible background" signal, not just a
  generic quality drop).
- Training-loss curves, for sanity-checking convergence per condition.

No new custom metric is introduced without being justified against a named
gap in the metrics above.

## 7. View-selection methods

- **Random:** uniform sample without replacement from `V_target`, seeded
  (see §8).
- **Strategic** (ablation only, C7): views selected by a fixed, documented
  rule (e.g., views with the largest target mask area first) — the rule is
  written into `src/poisoning/view_selection.py` and referenced by name in
  the condition's config, not chosen ad hoc per run.

## 8. Runs, seeds, statistical reporting

- Each condition is trained with **N seeds** (minimum N=3 for the core sweep,
  given compute constraints — see ROADMAP.md Phase 6 for the compute budget
  this implies). Report **mean ± std** across seeds per condition, not single
  runs.
- Seeds are fixed and recorded in the condition's config file before
  training, not chosen after seeing results.
- Random-view-selection sampling uses the same seed set as model
  initialization, and both are recorded per run in
  `experiments/logs/`.

## 9. Explicitly out of scope for the core study

(Full list and rationale in ROADMAP.md "What to Cut" and DECISION_LOG.md.)
Alpha-intensity sweep, inpainting-based real-photo spot-check, full IPA-NeRF
reproduction, online/inference-time attacks, physical-world validation,
Nerfacto cross-model comparison. Each may become a Phase-9 optional
extension only if time remains after the core sweep is complete and
evaluated — never before.

## 10. Deviation log

Any change to §1–§7 after this document is frozen must be logged here with a
date and a link to the corresponding `DECISION_LOG.md` entry. If this section
is empty, the protocol as originally frozen was followed exactly.

*(none — the protocol as frozen on 2026-09-14 has been followed exactly.)*

Note: the §3 "Scope of the edit" paragraph (shadow / ambient-occlusion /
colour-bleed retention under mask-limited erasure) is **not** a deviation. It
was added before the freeze, during Phase 4 Step 2, to state explicitly a
property the §3 formulas always had; the formulas themselves are unchanged.
See `DECISION_LOG.md` D-022 Finding 4 and D-024.
