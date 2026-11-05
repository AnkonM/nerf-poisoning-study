# Phase 8 results draft (descriptive only)

Status: DRAFT for the project lead. Plan of record: `DECISION_LOG.md` D-043 (analysis), D-042 (metric definitions).
Input of record: `experiments/results/metrics/per_run.csv`, sha256
`ba10c20623a59e9b97ef6da494d6b94146ed818101dbaf8d0f740aa8a242ac40` (24 runs). Tables: `experiments/results/phase8/`.
Figures: `paper/figures/` (each has a sidecar JSON with the hash, run ids, columns and commit).
Language rule: this document describes numbers. Where a hypothesis is mentioned it is labelled as one. With N = 3 seeds
there are no p-values; "separated" means the per-seed [min, max] ranges do not overlap, "overlapping" means they do.
All PSNR values are dB, data_range 1. Absolute values from this scene must not be compared with published NeRF numbers
(D-034). Poisoning is expressed as % of `V_target` (D-002).

Seed-spread caveat (D-043 A): `control` and `C7` train on ONE dataset, so their seed spread is initialisation-only
variance. C1-C6 each draw a different set of poisoned views per seed, so their spread mixes view-selection and
initialisation variance. The two are not like for like.

## 1. Claims table

| # | Claim (what the numbers show) | Source rows | Figure |
|---|---|---|---|
| 1 | Target-region PSNR (M_dil) falls from 38.61 (control) through 35.12, 31.45, 25.90, 22.56 to 18.08 at 0, 5, 10, 20, 30, 50% budget. Adjacent per-seed ranges overlap only for 20/30% (C3/C4); every other adjacent budget pair is separated. | `condition_summary.csv` (tgt_psnr; control, C1-C5) | Fig 1 |
| 2 | The same ordering appears in target SSIM (0.987 -> 0.709) and target LPIPS (0.0037 -> 0.245). | `condition_summary.csv` (tgt_ssim, tgt_lpips) | Fig 1 |
| 3 | Collateral-region PSNR is 51.90 (control), 49.31 (C1), then 45.39, 45.66, 46.41, 47.59 for C2-C5. C2-C5 per-seed ranges overlap one another; control is separated from C1-C6 (it overlaps C7), and C1 is separated from C2 and C3 but overlaps C4 and C5. | `condition_summary.csv` (col_psnr) | Fig 2 |
| 4 | Plate PSNR rises from 12.11 (control) to 16.62 (C5); the clean-GT-vs-plate reference over the same region is 12.12. | `condition_summary.csv` (plate_psnr), `reference_gt_vs_plate.csv` | Fig 3 |
| 5 | Hard erasure (C3) vs soft suppression at alpha 0.4 (C6), both 20%: target PSNR 25.90 vs 29.69 (ranges overlap); target SSIM and LPIPS ranges are separated; collateral PSNR 45.66 vs 47.98 (overlap); plate PSNR 13.69 vs 13.04 (overlap). | `pair_comparisons.csv` (hard_vs_soft) | Fig 4 |
| 6 | Random (C3) vs strategic (C7), both 20% hard erasure: target PSNR 25.90 vs 29.50 (separated), collateral PSNR 45.66 vs 50.87 (separated), plate PSNR 13.69 vs 15.38 (separated); target SSIM ranges overlap. | `pair_comparisons.csv` (random_vs_strategic) | Fig 5 |
| 7 | Whole-frame PSNR from the new code equals each run's eval-JSON PSNR within 0.0062 dB (24/24). | `per_run.csv` (`whole_psnr_dev_vs_eval_json`) | - |
| 8 | In poisoned training views the cast shadow outside the mask is unchanged, as expected from a mask-limited edit. | `data/poisoned/C5_seed1`, training view 85 | Fig 7 |

## 2. Primary results (mean +/- sample std, ddof = 1, n = 3; per-seed values in `condition_summary.csv`)

| Condition | Target PSNR | Collateral PSNR | Plate PSNR | Per-seed target PSNR |
|---|---|---|---|---|
| control (0%) | 38.607 +/- 0.132 | 51.899 +/- 0.325 | 12.108 +/- 0.055 | 38.745, 38.591, 38.483 |
| C1 (5%) | 35.116 +/- 1.364 | 49.310 +/- 0.949 | 12.584 +/- 0.391 | 36.421, 35.227, 33.700 |
| C2 (10%) | 31.445 +/- 2.074 | 45.395 +/- 2.248 | 13.035 +/- 0.461 | 33.466, 29.321, 31.549 |
| C3 (20%) | 25.898 +/- 2.596 | 45.656 +/- 2.288 | 13.691 +/- 0.639 | 23.438, 28.611, 25.644 |
| C4 (30%) | 22.557 +/- 1.189 | 46.413 +/- 1.620 | 14.409 +/- 0.373 | 21.389, 22.515, 23.766 |
| C5 (50%) | 18.082 +/- 0.435 | 47.589 +/- 1.293 | 16.625 +/- 0.195 | 17.927, 18.573, 17.746 |
| C6 (20%, soft, alpha 0.4) | 29.693 +/- 2.313 | 47.983 +/- 1.266 | 13.041 +/- 0.586 | 27.027, 31.169, 30.882 |
| C7 (20%, strategic; init-only variance) | 29.503 +/- 0.182 | 50.872 +/- 0.737 | 15.375 +/- 0.041 | 29.691, 29.490, 29.328 |

What the tables show, in plain numbers:

- **Budget curve (control, C1-C5).** Target-region PSNR decreases at every step of the ladder; there is no budget at
  which it is flat. Per-seed ranges of adjacent budgets overlap only for C3/C4 (sample std is largest at 10-20%:
  2.07 and 2.60 dB). Target SSIM and LPIPS show the same ordering. No threshold-shaped (flat-then-drop) pattern is
  visible in the target-region metrics at the five budgets tested; the smallest budget, 5%, already lowers target PSNR
  by 3.5 dB on average (range 33.70-36.42 vs control 38.48-38.75, separated).
- **Collateral-region PSNR across C1-C5.** The means are 49.31, 45.39, 45.66, 46.41, 47.59. This is not monotone: it falls
  from control to C2 and then rises slightly with budget. The per-seed ranges of C2, C3, C4 and C5 overlap
  ([42.97, 47.41], [43.72, 48.18], [45.39, 48.28], [46.15, 48.66]), and the sample stds (1.3-2.3 dB) are as large as the
  differences between neighbouring means. **No trend across C2-C5 is claimed.** Collateral LPIPS means are
  nearly equal across C2-C5 (0.019-0.020) and collateral SSIM means differ in the third decimal (0.9954-0.9962).
- **Plate PSNR.** Rises with budget; it is within 0.5 dB of the GT-vs-plate reference at control and C1, and 4.5 dB above it
  at C5. Reminder (D-042 G): the reference is the scale for reading this number, not a target.
- **Hard (C3) vs soft (C6).** Soft suppression at alpha 0.4 has higher target-region PSNR (+3.8 dB, ranges overlap),
  higher target SSIM and lower target LPIPS (both separated). Collateral PSNR +2.3 dB (overlap), collateral LPIPS lower
  (separated). Plate PSNR is lower for C6 (-0.65 dB, overlap). Only one alpha was tested.
- **Random (C3) vs strategic (C7).** C7's target PSNR is 3.6 dB higher than C3's, its collateral PSNR 5.2 dB higher, its
  plate PSNR 1.7 dB higher (all three separated); target SSIM ranges overlap and target LPIPS is higher for C7 (separated).
  C7's seed spread is initialisation-only (std 0.18 dB target PSNR) and C3's includes view selection (2.60 dB); the two
  spreads measure different things, so the narrower C7 range is not evidence that strategic selection is "more stable".
- **Mean-of-PSNR vs mean-of-MSE.** For C3 vs C7 plate PSNR is separated but plate MSE (averaged over views) overlaps. The
  two aggregates weight views differently (a PSNR mean is dominated by low-error views); both are in `pair_comparisons.csv`.

## 3. Exploratory results (post-hoc-motivated; NOT headline claims; D-043 C)

These diagnostics were chosen after the primary table had been seen. Thresholds used below for E2 (10 and 20 degrees) were
chosen after looking at Fig 8b and are descriptive only.

- **E1 (per-view distributions, 225 views per condition).** Poisoned conditions have wide per-view spreads: target-PSNR
  minimum 7.4-15.0 dB and maximum 30.8-41.0 dB in C1-C7, against 33.0-41.6 dB for control. The median is above the mean
  for C1-C4, C6 and C7 (e.g. C7: median 33.32, mean 29.50, IQR 22.73-37.61); C5's median (17.71) is below its mean (18.08). Plate PSNR for C7 has a long upper tail (max 39.9 dB; mean 15.38, median 12.55). It shows
  that the condition means hide per-view heterogeneity; it does not show why views differ. (`E1_per_view_distribution.csv`,
  Fig 8a.)
- **E2 (C3 vs C7 geometry).** Camera angles are measured from the scene origin. The median angular distance from an eval
  view to its nearest poisoned training view is 9.8-14.0 deg for C3 (per seed) and 18.8 deg for C7; C7's poisoned views
  leave 35 of 75 eval views more than 20 deg from any poisoned view (C3: 11-13). In both conditions, views far from every
  poisoned view have higher target PSNR (mean target PSNR, views > 20 deg vs <= 10 deg: C3 29.7-31.3 vs 19.9-24.7; C7
  36.7-37.0 vs 18.7-19.4). Fig 8b shows that the lowest-PSNR C7 views cluster at azimuths of roughly -100 to -60 and +50 to +100 deg.
  This is an association within one scene; it does not settle why C3 and C7 differ in the aggregate, and it is a
  hypothesis (not a finding) that the distance to the nearest poisoned view matters. (`E2_per_view_geometry.csv`, Fig 8b.)
- **E3 (collateral error by distance from the mask).** Mean squared error render-vs-GT (x1e-5, mean over seeds) in the
  0-10 px band: control 0.94, C1 3.96, C2 37.0, C3 15.0, C4 16.7, C5 22.9, C6 6.9, C7 6.5. In the > 100 px band: control
  0.76, C1 1.59, C2 3.80, C3 3.42, C4 5.61, C5 1.84, C6 1.69, C7 0.94. The 0-10 px band carries the largest errors in
  every poisoned condition; the C2 mean in the two inner bands is driven by seed 1 (81.9 and 50.8 vs 10.8/18.3 and
  3.2/9.3 for seeds 2/3). C4's > 100 px mean is driven by seed 1 (11.1 vs 2.4/3.3). Bands are a diagnostic, not a metric.
  (`E3_collateral_band_mse.csv`, Fig 8c.)
- **E4 (loss and validation vs collateral).** Final training loss is 4.8e-5 to 2.6e-4 across the 24 runs and decreases with
  seed index within every condition (seed 1 > seed 2 > seed 3 in all 8 conditions; this is an observation about the 24
  runs, not an explanation). Final
  validation-subset PSNR (clean val views) is 31.1-49.2 dB. Nothing in the table is interpreted: the 24 pairs are in
  `E4_loss_val_vs_collateral.csv`. Source of the validation number: last point of the `val/psnr_subset` TensorBoard scalar
  (`summary.json` carries none), step 200000 for every run.
- **E5 (resumed run).** `control_seed3` (resumed from the 150,000-iteration checkpoint with RNG restored) lies inside the
  [min, max] of control seeds 1-2 for whole PSNR, target SSIM and collateral PSNR, and outside it for 11 of the 14 metric
  columns (e.g. target PSNR 38.483 vs 38.591-38.745; plate PSNR 12.045 vs 12.138-12.142). With only two reference seeds the
  reference range is very narrow, so "outside" here neither shows nor rules out an effect of the resume (for some metrics
  the distance from the seeds 1-2 range is similar to the seed 1-seed 2 difference, for others it is larger, e.g. plate PSNR). (`E5_resumed_run.csv`.)

## 4. Limitations

- One scene (a single synthetic tabletop scene with one target object) and one Blender-rendered eval set (75 views).
- One GPU model (Quadro RTX 5000, driver 580.173.02, torch 2.14.0+cu130) for all 24 runs; no cross-GPU variation measured.
- N = 3 seeds per condition; `control` and `C7` have a single dataset each, so their variance is initialisation-only.
- Soft suppression tested at one alpha (0.4) (METHODOLOGY §3 fixes alpha once). Hard erasure: random and one strategic rule
  (largest target visibility).
- Shadow and contact effects outside the mask are retained by design (D-022 Finding 4; METHODOLOGY §3 "Scope of the
  edit"); they are counted as collateral error by the M_dil partition.
- `control_seed3` was interrupted at ~160,500 iterations and resumed from 150,000 with RNG restored (D-039); bit-equivalence
  to an uninterrupted run was not shown. Its recorded train wall time covers only the resumed segment (D-041).
- The 24 render sets were produced at commit `6be1b89` with four then-untracked closeout files (recorded in each
  `render_meta.json`); render checkpoint hashes and eval-set hashes were verified (D-042).
- Renders are stored as float16; the effect on whole-frame PSNR is at most 0.0062 dB (D-041, and `per_run.csv` deviations).
- Colab/Kaggle environment parity is unverified and unexercised.
- Region-pooled SSIM and LPIPS partly reflect context outside the region (receptive field / window; D-042 E, F); PSNR is the
  region-pure metric. The whole-frame SSIM includes the 5 px border (D-042 E).
- Loss-curve human sign-off (D-041) is still pending.

## 5. Open questions for the project lead (not answered here)

1. How should the non-monotone collateral PSNR across C2-C5 (overlapping ranges, large seed spread) be framed in the paper
   (e.g. as "no detectable trend at N = 3")? Is more seeds at C2-C5 wanted, or is that Phase 9?
2. C7 differs from C3 in target, collateral and plate PSNR (all separated). Is a follow-up on the view-geometry hypothesis
   (E2) wanted, and if so does it need its own decision entry before any claim is made?
3. Should per-view heterogeneity (E1) be part of the paper's main text, given that condition means hide skewed or bimodal
   per-view distributions?
4. Is `rawtgt_` (sensitivity) to be reported in the paper beyond Fig 1? It is within about 1-2 dB of the M_dil numbers
   and shows the same ordering.
5. Do you want the mean-of-PSNR vs mean-of-MSE difference (section 2, last bullet) addressed by choosing one aggregate?
6. Should Phase 7 be marked complete in the ROADMAP now that the C3-vs-C7 table and figure exist?
