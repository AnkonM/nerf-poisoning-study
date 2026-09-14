# Handoff — 2026-09-14, Phase 4 complete

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

Windows 11 laptop, RTX 5060 (Blackwell, sm_120), WSL2 Ubuntu, `uv` for
Python (D-009), Blender **5.2.1 LTS** installed natively on Windows (D-008).
Phase 4 established that headless Blender can be **driven from a WSL2
shell** and writes straight into the repo over the UNC path, with no copy
step (D-021). Setup and verification commands:
[environment/SETUP.md](../../environment/SETUP.md) (§3.4/§3.5 now record the
verified invocation).

## Current phase + status

As of this handoff, from `docs/ROADMAP.md`'s status tracker:

| Phase | Status | Gate passed? |
|---|---|---|
| 0 — Setup | Complete (local) | PASS locally; Colab/Kaggle parity still unverified (deferred to Phase 6) |
| 1 — Literature verification | Complete | PASS (D-012) |
| 2 — Lego clean-NeRF sanity check | Complete | PASS — 31.550 dB (D-016) |
| 3 — Poisoning proof of concept (Lego) | Complete | PASS (D-017–D-020) |
| 4 — Final scene + eval set | **Complete** | **PASS** — `|V_target|` = 100/100 training views; eval set frozen (226 files, aggregate SHA-256 `211a5a59…5ab214`); `METHODOLOGY.md` frozen 2026-09-14 (D-021–D-026) |
| 5 — Poisoning pipeline build | **Next** | — |

## What Phase 4 produced

- **Scene:** a tabletop diorama built **procedurally** by
  `scripts/build_scene.py` from `configs/scenes/final_scene.yaml`. Red mug
  (target, `pass_index = 1`) at the origin; four distractors (blue box,
  green sphere, yellow cone, tan flat-lying torus) on a 0.70 m ring.
  `data/raw/*.blend` is a **build artifact and is gitignored** — the config
  is the source of truth, never the `.blend` (D-022).
- **Rig:** 185 Fibonacci-lattice poses, elevation 25–75°, radius 2.9 m, cut
  by a seeded shuffle into **train 100 / val 10 / eval_holdout 75** —
  disjoint by construction (D-023).
- **Data:** 535 files — 185 originals, plus masks and background plates for
  train + eval_holdout (val is monitoring-only). 360 renders in 14.6 min.
- **`|V_target|` = 100 of 100** training views. Budget ladder is whole
  numbers throughout: 5% → 5 views, 10% → 10, 20% → 20, 30% → 30, 50% → 50.

## Key locked decisions relevant to picking up work right now

Phase 0–3 entries still apply (especially **D-002** budget = % of
target-visible views, **D-005** vanilla NeRF, **D-009/D-010** `uv` as the
dependency source of truth). New in Phase 4:

- **D-021** — verified WSL2→Blender invocation, OptiX device filtering
  (filter on `type == OPTIX`; this machine also exposes an AMD iGPU), UNC
  direct-write, and the three-disjoint-pool design.
- **D-022** — locked scene. Also **Finding 4**, the important one: the §3
  formulas read the plate **only inside the mask**, so the target's cast
  shadow, AO and colour bleed **survive poisoning**. Resolved as Option A —
  §3 unchanged, property documented. Don't "fix" this.
- **D-023** — rig + `cameras.json` schema + **measured** near/far.
- **D-024** — mask dilation **3 px** (not 5: 5 closes the mug's handle hole
  in half the poses). Denoiser bleed cleared.
- **D-025** — `|V_target|` = 100, gate PASS.
- **D-026** — the freeze. **`METHODOLOGY.md` is now FROZEN** — changing
  §1–§7 requires both a `DECISION_LOG.md` entry *and* a line in §10's
  deviation log.

## Known past issues/bugs already fixed

Phase 0–3: **D-011/D-014** brace-glob directories, **D-015** whitespace-broken
`.gitignore`, **D-016** the VRAM-contention stall that looked like a hang
(check CPU ticks before assuming a hang). New in Phase 4:

- **Key light aimed upward** (D-022) — a sign error in hand-rolled Euler aim
  math meant the scene was lit only by ambient and bounce. It failed
  *silently* as "flat, shadowless lighting". Fixed with `to_track_quat`.
  **The config's key-light energy of 90 W is correct** and is ~10x lower
  than the pre-fix drafts that were compensating — do not "correct" it up.
- **Blanket `shade_smooth`** rounded cones and cylinders into blobs; now
  angle-based auto-smooth.
- **Blender 5.2 compositor API breaks** (all documented in
  `src/data_pipeline/blender_render_views.py`): compositor lives in
  `scene.compositing_node_group`; no `CompositorNodeComposite`; the socket
  is `"Object Index"` not `IndexOB` and only exists under Cycles; ID Mask's
  index/anti-alias are *input sockets*; and File Output's
  `format.media_type` defaults to `MULTI_LAYER_IMAGE`, hard-locking output
  to EXR until set to `"IMAGE"`. Expect these if writing new bpy code.
- **Masks are 16-bit** (`uint16`, {0, 65535}). D-017's Lego `alpha > 127`
  assumes uint8 and would mark *every* pixel foreground — assert the dtype.
  `src/data_pipeline/scene_validation.py::read_mask` does this.

## Reproducibility: ±1 LSB, not bit-exact (D-027)

The full closeout check was executed for real: rebuild the `.blend`,
regenerate the rig, re-render all 360 images from the committed config, then
compare. **Poses, masks and every derived number reproduce exactly**
(`|V_target|` = 100, gate PASS, masks pixel-identical, `transforms_test.json`
byte-identical). **RGB renders do not** — Cycles + OptiX is not
bit-deterministic (a fixed seed fixes the sample pattern, not the
floating-point summation order), so a re-render differs by **1/255 on ~0.003%
of pixels** and the SHA-256 manifest does **not** match a regenerated copy.

Two consequences for whoever picks this up:

1. **Do not re-render the frozen eval set to "check" it.** Use
   `python scripts/freeze_eval_set.py --verify`, which compares the frozen
   copy against the manifest (that still passes byte-for-byte). A
   regeneration will always appear to fail.
2. **Do not describe this pipeline as bit-exact reproducible.** The honest
   claim is: exact for poses/masks/derived numbers, ±1/255 on ~0.003% of
   pixels for renders.

## On the horizon — check these first if something looks wrong later

1. **near/far sampling span (D-023) — check this FIRST if Phase 6
   convergence disappoints.** `near`/`far` are now **1.5 / 9.0**, measured
   from the real scene (actual depth range 1.806–8.346 m). That is an
   **8.1 m span versus Lego's 4.0 m**, so with the same 64 coarse / 128 fine
   samples the per-sample spacing is roughly **2x coarser than the Phase 2
   configuration that produced the validated 31.55 dB baseline**. This is
   the correct bracket for this scene's geometry and it does **not** affect
   the frozen data — near/far are config values, changeable up until Phase 6
   training starts. But if the final scene trains to a visibly worse PSNR
   than Phase 2's Lego run, **do not assume the poisoning pipeline or the
   scene is at fault**: check sample-count-vs-depth-span first, and consider
   raising `render.num_coarse_samples`/`num_fine_samples` before anything
   else. Log any change as a new `DECISION_LOG.md` entry.
2. **D-020 debt is now due.** `scripts/build_poison_set.py` currently emits
   only `train/` + `transforms_train.json`; Phase 3's val/test were manual
   symlinks. D-021 explicitly rescheduled this to **Phase 5's first task**.
   It must be fixed before the real 8-condition sweep relies on that script,
   or the poisoned sets violate `PROJECT_STRUCTURE.md`'s
   "reproducible-from-config-alone" rule.
3. **Phase 6 iteration count is an open decision.** D-018's 30,000 was
   scoped to the Lego PoC and sets **no precedent**. The final scene needs
   its own convergence check against its own reference range, analogous to
   D-013.
4. **Colab/Kaggle environment parity is still unverified** (Phase 0 status
   tracker), and Phase 6 is where it finally gets exercised.
5. **Shadow retention is expected, not a bug** (D-022 Finding 4). When
   qualitative figures are produced in Phase 8, poisoned views will show the
   mug gone but its shadow present. This is pre-registered in
   `METHODOLOGY.md` §3's "Scope of the edit" paragraph and should be
   *reported*, not engineered away.

## Immediate next steps

Per `docs/ROADMAP.md` **Phase 5 — Poisoning Pipeline Build**: implement the
finalized hard-erasure and soft-suppression compositing against the real
`background_plate`/`mask` triples, implement random and strategic view
selection, finish `build_poison_set.py` (including the D-020 fix above),
write the unit tests, and generate all 8 conditions' poisoned sets.

Two values Phase 5 needs, already locked and non-negotiable: mask dilation
**3 px** (D-024) and `|V_target|` = **100**, so condition view counts are
5/10/20/30/50. `alpha` for soft suppression (C6) is **still unset** —
`METHODOLOGY.md` §3 gives a candidate range of 0.4–0.5 and requires it be
fixed once, recorded in `configs/poisoning/soft_suppression_20.yaml`, and
never varied. Note `METHODOLOGY.md` is now frozen, so setting alpha within
the pre-registered range is fine, but going outside it is a deviation.

**Gate:** for every condition, `MANIFEST.csv` view counts must equal
`round(b/100 × 100)` exactly, and a visual spot-check must confirm only the
target region differs from the original.

---

**Before doing any work, read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full. This file is a
map, not a substitute.**
