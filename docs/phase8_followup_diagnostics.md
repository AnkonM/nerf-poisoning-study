# Phase 8 follow-up diagnostics (D-044)

Descriptive or exploratory (post-hoc-motivated) only. Plan of record: `docs/DECISION_LOG.md` D-044, committed before anything here
was computed. Primary aggregates are as pre-registered in D-043; nothing here replaces them. No p-values, no causal
language. Definitions D-042/D-043, `per_run.csv`, the existing metrics outputs, renders, checkpoints and frozen data are
unchanged. All parameters are in `configs/analysis/phase8_followup_d044.yaml`.

Run: `python scripts/followup_diagnostics.py configs/analysis/phase8_followup_d044.yaml` (Part 2, tables in
`experiments/results/phase8/followup/`, figures 9 and 10) and
`python scripts/depth_opacity_diagnostics.py configs/analysis/phase8_followup_d044.yaml` (Part 3, GPU, figure 11).
Figures carry sidecar JSON files with the `per_run.csv` sha256 (`ba10c206...ac40`), run ids, columns and git commit.

## 2a. C3 vs C6, per-seed paired differences (C6 minus C3)

Precondition checked by the script and found true: at each seed, `C3_seed<s>` and `C6_seed<s>` poison the identical set of
20 training views (`MANIFEST.csv`); they also share the seed (init and batch order, by code reading, not bit for bit).
Figure 9.

| metric | seed 1 | seed 2 | seed 3 | mean | sd (ddof=1) | seeds > 0 | seeds < 0 | one sign in all 3 |
|---|---|---|---|---|---|---|---|---|
| tgt_psnr | 3.589 | 2.558 | 5.238 | 3.795 | 1.352 | 3 | 0 | yes |
| tgt_ssim | 0.07374 | 0.03478 | 0.06299 | 0.05717 | 0.02012 | 3 | 0 | yes |
| tgt_lpips | -0.06758 | -0.02945 | -0.05386 | -0.0503 | 0.01931 | 0 | 3 | yes |
| col_psnr | 1.258 | 3.663 | 2.06 | 2.327 | 1.225 | 3 | 0 | yes |
| col_ssim | 0.0009605 | 0.001636 | 0.001256 | 0.001284 | 0.0003388 | 3 | 0 | yes |
| col_lpips | -0.006465 | -0.01661 | -0.009173 | -0.01075 | 0.005255 | 0 | 3 | yes |
| plate_psnr | -0.6927 | -0.4266 | -0.8297 | -0.6497 | 0.205 | 0 | 3 | yes |
| plate_mse | 0.002827 | 0.002585 | 0.002764 | 0.002726 | 0.0001253 | 3 | 0 | yes |

Counts of positive / negative seeds are over three seeds only.

## 2b. Seed-by-condition grid

Values (per-run) and rank within the seed (1 = largest value) for the three PSNR columns; the other metrics are in
`2b_seed_condition_grid.csv`. control and C7 show initialisation-only variation (one dataset); C1-C6 show view-selection
plus initialisation variation.

**tgt_psnr** (value, rank in seed; rank 1 = largest)

| seed | control | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|---|
| 1 | 38.75 (1) | 36.42 (2) | 33.47 (3) | 23.44 (6) | 21.39 (7) | 17.93 (8) | 27.03 (5) | 29.69 (4) |
| 2 | 38.59 (1) | 35.23 (2) | 29.32 (5) | 28.61 (6) | 22.51 (7) | 18.57 (8) | 31.17 (3) | 29.49 (4) |
| 3 | 38.48 (1) | 33.70 (2) | 31.55 (3) | 25.64 (6) | 23.77 (7) | 17.75 (8) | 30.88 (4) | 29.33 (5) |

**col_psnr** (value, rank in seed; rank 1 = largest)

| seed | control | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|---|
| 1 | 51.57 (1) | 48.22 (6) | 42.97 (8) | 48.18 (7) | 48.28 (5) | 48.66 (4) | 49.44 (3) | 51.16 (2) |
| 2 | 52.22 (1) | 49.72 (3) | 47.41 (4) | 43.72 (8) | 45.57 (7) | 46.15 (6) | 47.38 (5) | 51.43 (2) |
| 3 | 51.91 (1) | 49.98 (3) | 45.81 (6) | 45.07 (8) | 45.39 (7) | 47.96 (4) | 47.13 (5) | 50.04 (2) |

**plate_psnr** (value, rank in seed; rank 1 = largest)

| seed | control | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|---|
| 1 | 12.14 (8) | 12.30 (7) | 12.64 (6) | 14.41 (4) | 14.76 (3) | 16.65 (1) | 13.72 (5) | 15.42 (2) |
| 2 | 12.14 (8) | 12.42 (7) | 13.54 (4) | 13.18 (5) | 14.44 (3) | 16.42 (1) | 12.75 (6) | 15.34 (2) |
| 3 | 12.05 (8) | 13.03 (5) | 12.92 (6) | 13.48 (4) | 14.02 (3) | 16.81 (1) | 12.65 (7) | 15.37 (2) |

Conditions whose rank is identical in all three seeds:

| metric | conditions with the same rank in all 3 seeds (of 8) |
|---|---|
| tgt_psnr | 5 |
| tgt_ssim | 2 |
| tgt_lpips | 2 |
| col_psnr | 2 |
| col_ssim | 1 |
| col_lpips | 2 |
| plate_psnr | 4 |
| plate_mse | 2 |

Mean residual from the condition mean, per seed (averaged over the 8 conditions):

| metric | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| tgt_psnr | -0.3495 | 0.3246 | 0.02486 |
| tgt_ssim | -0.005079 | 0.002994 | 0.002084 |
| tgt_lpips | 0.005607 | -0.003809 | -0.001798 |
| col_psnr | 0.419 | -0.1898 | -0.2292 |
| col_ssim | 4.178e-06 | 0.0001929 | -0.0001971 |
| col_lpips | -0.0005418 | 0.0002971 | 0.0002447 |
| plate_psnr | 0.1459 | -0.07828 | -0.06759 |
| plate_mse | -0.001505 | 0.001072 | 0.000433 |

Pearson correlation, over the 8 conditions, between the residual vectors of two seeds (residual = value minus that
condition's mean over the three seeds):

| metric | r(seed1,seed2) | r(seed1,seed3) | r(seed2,seed3) | mean of the 3 |
|---|---|---|---|---|
| tgt_psnr | -0.859 | -0.555 | 0.050 | -0.455 |
| tgt_ssim | -0.732 | -0.438 | -0.291 | -0.487 |
| tgt_lpips | -0.663 | -0.605 | -0.195 | -0.488 |
| col_psnr | -0.926 | -0.703 | 0.383 | -0.415 |
| col_ssim | -0.885 | -0.663 | 0.239 | -0.436 |
| col_lpips | -0.961 | -0.683 | 0.455 | -0.396 |
| plate_psnr | -0.722 | -0.680 | -0.018 | -0.473 |
| plate_mse | -0.485 | -0.727 | -0.248 | -0.487 |

Reading note (arithmetic, not a finding): residuals taken from a three-seed condition mean sum to zero over the seeds
within each condition. With three seeds and independent noise, the expected pairwise correlation of these residual
vectors is about -0.5, so values near -0.5 are what no shared seed effect would give, and values clearly above it would
indicate shared variation. The column "mean of the 3" is the number to compare with -0.5.

## 2c. Mean-of-MSE aggregates beside the D-043 mean-of-PSNR

For every run the mean of the per-view PSNRs reproduces the `per_run.csv` value to 1e-9 dB (asserted by the script).
"Pooled MSE, pixel-weighted": per-view MSE weighted by that view's region pixel count, summed over the 75 views, then
converted to PSNR. "Pooled MSE, view-mean": unweighted mean of the 75 per-view MSEs, then converted. Plate distance uses
the target-region pixel count as weight. Cells are mean over seeds [per-seed min, max], dB.

**target (M_dil)** — mean [per-seed min, max], dB

| condition | mean of per-view PSNR (D-043) | pooled MSE, pixel-weighted | pooled MSE, view-mean |
|---|---|---|---|
| control | 38.61 [38.48, 38.75] | 38.32 [38.16, 38.44] | 38.33 [38.18, 38.44] |
| C1 | 35.12 [33.70, 36.42] | 28.85 [23.73, 33.54] | 28.91 [23.75, 33.56] |
| C2 | 31.45 [29.32, 33.47] | 23.48 [19.49, 26.68] | 23.67 [19.67, 26.81] |
| C3 | 25.90 [23.44, 28.61] | 21.04 [18.97, 23.09] | 21.14 [18.97, 23.35] |
| C4 | 22.56 [21.39, 23.77] | 18.85 [17.04, 20.51] | 18.91 [17.12, 20.56] |
| C5 | 18.08 [17.75, 18.57] | 15.30 [15.05, 15.54] | 15.33 [15.02, 15.56] |
| C6 | 29.69 [27.03, 31.17] | 25.76 [23.86, 27.22] | 25.85 [23.90, 27.38] |
| C7 | 29.50 [29.33, 29.69] | 18.60 [18.40, 18.70] | 19.10 [18.91, 19.21] |

**collateral** — mean [per-seed min, max], dB

| condition | mean of per-view PSNR (D-043) | pooled MSE, pixel-weighted | pooled MSE, view-mean |
|---|---|---|---|
| control | 51.90 [51.57, 52.22] | 50.96 [50.73, 51.42] | 50.96 [50.73, 51.42] |
| C1 | 49.31 [48.22, 49.98] | 47.89 [46.09, 48.88] | 47.89 [46.09, 48.88] |
| C2 | 45.39 [42.97, 47.41] | 42.89 [39.23, 45.66] | 42.89 [39.23, 45.66] |
| C3 | 45.66 [43.72, 48.18] | 44.33 [42.68, 47.23] | 44.33 [42.68, 47.23] |
| C4 | 46.41 [45.39, 48.28] | 43.29 [40.95, 44.71] | 43.29 [40.95, 44.71] |
| C5 | 47.59 [46.15, 48.66] | 46.53 [45.11, 47.53] | 46.53 [45.11, 47.53] |
| C6 | 47.98 [47.13, 49.44] | 47.00 [45.80, 48.58] | 47.00 [45.80, 48.58] |
| C7 | 50.87 [50.04, 51.43] | 49.81 [48.71, 50.72] | 49.81 [48.71, 50.72] |

**plate distance (over M_dil)** — mean [per-seed min, max], dB

| condition | mean of per-view PSNR (D-043) | pooled MSE, pixel-weighted | pooled MSE, view-mean |
|---|---|---|---|
| control | 12.11 [12.05, 12.14] | 11.97 [11.91, 12.01] | 11.97 [11.90, 12.00] |
| C1 | 12.58 [12.30, 13.03] | 12.35 [12.14, 12.62] | 12.34 [12.13, 12.60] |
| C2 | 13.04 [12.64, 13.54] | 12.66 [12.44, 13.00] | 12.63 [12.42, 12.95] |
| C3 | 13.69 [13.18, 14.41] | 12.87 [12.49, 13.49] | 12.83 [12.44, 13.46] |
| C4 | 14.41 [14.02, 14.76] | 12.94 [12.81, 13.15] | 12.93 [12.82, 13.15] |
| C5 | 16.62 [16.42, 16.81] | 14.36 [13.71, 15.00] | 14.35 [13.70, 15.00] |
| C6 | 13.04 [12.65, 13.72] | 12.64 [12.27, 13.23] | 12.61 [12.23, 13.20] |
| C7 | 15.38 [15.34, 15.42] | 12.91 [12.86, 12.96] | 12.78 [12.73, 12.83] |

Pair statements (range overlap or separation, D-043 rule) that differ between the D-043 aggregate and at least one
alternative, over all 28 condition pairs and three regions. Count per region: tgt: 9 of 28; col: 6 of 28; plate: 7 of 28.

22 of 84 (region x pair) statements differ between the D-043 aggregate and at least one alternative.

| region | pair | mean-of-PSNR (D-043) | pooled MSE, pixel-weighted | pooled MSE, view-mean | D-043 pre-registered pair |
|---|---|---|---|---|---|
| tgt | C1 vs C2 | separated | overlap | overlap |  |
| tgt | C1 vs C6 | separated | overlap | overlap |  |
| tgt | C2 vs C3 | separated | overlap | overlap |  |
| tgt | C2 vs C4 | separated | overlap | overlap |  |
| tgt | C2 vs C7 | overlap | separated | separated |  |
| tgt | C3 vs C6 | overlap | separated | separated | yes |
| tgt | C3 vs C7 | separated | separated | overlap | yes |
| tgt | C4 vs C7 | separated | overlap | overlap |  |
| tgt | C6 vs C7 | overlap | separated | separated |  |
| col | C1 vs C3 | separated | overlap | overlap |  |
| col | C1 vs C4 | overlap | separated | separated |  |
| col | C1 vs C7 | separated | overlap | overlap |  |
| col | C2 vs C6 | overlap | separated | separated |  |
| col | C4 vs C5 | overlap | separated | separated |  |
| col | C4 vs C6 | overlap | separated | separated |  |
| plate | C1 vs C3 | separated | overlap | overlap |  |
| plate | C2 vs C4 | separated | overlap | overlap |  |
| plate | C2 vs C7 | separated | overlap | overlap |  |
| plate | C3 vs C7 | separated | overlap | overlap | yes |
| plate | C4 vs C6 | separated | overlap | overlap |  |
| plate | C4 vs C7 | separated | overlap | overlap |  |
| plate | C6 vs C7 | separated | overlap | overlap |  |

## 2d. Per-view: farther from both the clean GT and the plate than they are from each other

Target-region (M_dil) MSE; flagged when d_GT > d_ref and d_plate > d_ref (strict), d_ref = MSE(clean GT, plate) for that
view. Cells: number of views flagged of 75 (in brackets, for context: views with d_GT > d_ref only, views with
d_plate > d_ref only). Total flagged over the 15 runs: 0.

| condition | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| control | 0/75 (GT-only 0, plate-only 35) | 0/75 (GT-only 0, plate-only 22) | 0/75 (GT-only 0, plate-only 75) |
| C3 | 0/75 (GT-only 3, plate-only 16) | 0/75 (GT-only 1, plate-only 20) | 0/75 (GT-only 1, plate-only 33) |
| C5 | 0/75 (GT-only 11, plate-only 18) | 0/75 (GT-only 10, plate-only 24) | 0/75 (GT-only 8, plate-only 11) |
| C6 | 0/75 (GT-only 0, plate-only 14) | 0/75 (GT-only 0, plate-only 30) | 0/75 (GT-only 0, plate-only 32) |
| C7 | 0/75 (GT-only 8, plate-only 27) | 0/75 (GT-only 8, plate-only 27) | 0/75 (GT-only 10, plate-only 24) |

For context, the largest value over views of min(d_GT, d_plate) / d_ref:

| condition | largest value over its 3 runs and 75 views of min(d_GT, d_plate) / d_ref |
|---|---|
| control | 0.007 |
| C3 | 0.330 |
| C5 | 0.504 |
| C6 | 0.294 |
| C7 | 0.280 |

## 2e. Spread of per-view target-region PSNR

Figure 10. IQR = Q3 - Q1 (numpy linear interpolation); range = max - min; dB. Pooled = 3 seeds x 75 views = 225 values.

| condition | pooled IQR | pooled range | pooled median | IQR / control IQR | per-run IQR (seeds 1,2,3) | per-run range (seeds 1,2,3) |
|---|---|---|---|---|---|---|
| control | 1.81 | 8.56 | 38.78 | 1.00 | 1.78, 1.81, 1.90 | 8.13, 6.88, 8.32 |
| C1 | 3.87 | 30.16 | 37.06 | 2.13 | 2.59, 3.21, 7.18 | 18.03, 23.59, 30.16 |
| C2 | 9.27 | 33.15 | 34.22 | 5.12 | 6.04, 11.93, 9.63 | 24.34, 31.29, 27.33 |
| C3 | 9.79 | 27.38 | 27.33 | 5.40 | 9.63, 6.61, 9.27 | 26.49, 23.67, 24.62 |
| C4 | 8.41 | 26.20 | 23.22 | 4.64 | 7.84, 8.61, 8.03 | 25.45, 22.43, 21.33 |
| C5 | 8.47 | 22.82 | 17.71 | 4.67 | 8.12, 9.88, 6.95 | 18.20, 21.20, 20.26 |
| C6 | 9.22 | 23.93 | 30.73 | 5.09 | 8.12, 7.19, 9.33 | 23.34, 20.86, 21.74 |
| C7 | 14.88 | 30.98 | 33.32 | 8.21 | 14.81, 14.82, 14.93 | 30.05, 30.20, 30.38 |

## 3. Depth and opacity (EXPLORATORY; control, C3, C5, C7; seed 1 only)

Procedure as in D-044 (existing render path, each run's final checkpoint, fine network, perturb 0, fp32, 400x400, all 75
eval views, GPU only, ray chunk 2048 with no out-of-memory halving needed). Statistics use the float32 arrays; the
float16 copies are in `experiments/renders_diagnostics/<run>/depth_acc_fp16.npz` (git-ignored). Expected depth is
`1 / disp_map` (scene units; near 1.5, far 9.0). Figure 11. The newly rendered RGB matches the stored renders to within
float16 storage precision, so the checkpoints and render path are the ones that produced the stored renders:

| run | ray chunk used | peak VRAM reserved (GiB) | render minutes | new RGB vs stored render: mean PSNR (dB) / min PSNR (dB) / max abs diff | depth min / max | acc min / max |
|---|---|---|---|---|---|---|
| control_seed1 | 2048 | 0.61 | 16.2 | 79.36 / 78.73 / 0.0002 | 1.840 / 6.690 | 0.999975 / 1.000001 |
| C3_seed1 | 2048 | 0.64 | 14.9 | 79.34 / 78.69 / 0.0002 | 1.830 / 7.543 | 0.895949 / 1.000001 |
| C5_seed1 | 2048 | 0.64 | 14.9 | 79.34 / 78.70 / 0.0002 | 1.839 / 7.524 | 0.992452 / 1.000001 |
| C7_seed1 | 2048 | 0.64 | 14.6 | 79.34 / 78.72 / 0.0002 | 1.861 / 8.050 | 0.850821 / 1.000001 |

Accumulated opacity over M_dil and expected depth over M_dil (per view, then summarised over 75 views). Opacity is
displayed as 1 - opacity because every value is 1 to within about 1e-7:

| condition | quantity | mean | median | Q1 | Q3 | min | max |
|---|---|---|---|---|---|---|---|
| control | 1 - mean accumulated opacity over M_dil | 6.51e-08 | 6.36e-08 | 5.71e-08 | 7.19e-08 | 4.59e-08 | 8.85e-08 |
| C3 | 1 - mean accumulated opacity over M_dil | 1.87e-08 | 1.8e-08 | 1.64e-08 | 2.04e-08 | 1.34e-08 | 2.96e-08 |
| C5 | 1 - mean accumulated opacity over M_dil | 1.3e-08 | 1.29e-08 | 1.18e-08 | 1.43e-08 | 8.81e-09 | 1.8e-08 |
| C7 | 1 - mean accumulated opacity over M_dil | 1.64e-08 | 1.6e-08 | 1.52e-08 | 1.77e-08 | 1.15e-08 | 2.19e-08 |
| control | mean expected depth over M_dil (scene units) | 2.8038 | 2.8043 | 2.7902 | 2.8167 | 2.7708 | 2.8422 |
| C3 | mean expected depth over M_dil (scene units) | 2.8460 | 2.8450 | 2.8289 | 2.8629 | 2.7962 | 2.8891 |
| C5 | mean expected depth over M_dil (scene units) | 2.8509 | 2.8519 | 2.8326 | 2.8647 | 2.8102 | 2.8964 |
| C7 | mean expected depth over M_dil (scene units) | 2.8500 | 2.8510 | 2.8317 | 2.8637 | 2.8102 | 2.8951 |

Depth difference from the control model's depth, mean absolute over M_dil, per view, summarised over 75 views:

| condition | quantity | mean | median | Q1 | Q3 | min | max |
|---|---|---|---|---|---|---|---|
| C3 | mean abs depth difference vs control (scene units) | 0.04366 | 0.04375 | 0.04084 | 0.04798 | 0.03021 | 0.05121 |
| C5 | mean abs depth difference vs control (scene units) | 0.04847 | 0.04817 | 0.04518 | 0.05218 | 0.04054 | 0.05693 |
| C7 | mean abs depth difference vs control (scene units) | 0.0467 | 0.04675 | 0.04362 | 0.04998 | 0.03874 | 0.0558 |

Relation to the angular distance to the nearest poisoned training view (E2 function; the 10 and 20 degree thresholds were
chosen post hoc in D-043 and are used only for comparability; control has no poisoned views):

| condition | Spearman rho (depth diff vs nearest poisoned angle, 75 views) | nearest-angle range (deg) | bin <10: n, mean | bin 10-20: n, mean | bin >=20: n, mean |
|---|---|---|---|---|---|
| C3 | 0.409 | 7.1 - 31.4 | 40, 0.0423 | 22, 0.0435 | 13, 0.0480 |
| C5 | 0.174 | 7.1 - 16.9 | 58, 0.0481 | 17, 0.0498 | 0, - |
| C7 | 0.094 | 7.1 - 53.8 | 20, 0.0467 | 20, 0.0465 | 35, 0.0468 |

## Limitations

- Accumulated opacity cannot distinguish the mug from the opaque surface behind it; a ray ending on the table gives
  opacity near 1 whether or not the mug is represented.
- No ground-truth depth exists for the plate scene. Depth is compared with the control model only, so a difference is a
  difference between two models, not an error against geometry.
- One seed per condition in Part 3, so no between-seed statement can be made. There is no baseline for how much the
  expected depth of two control-like models differs from each other (for example control seed 1 against control seed 2),
  so the size of the numbers above cannot be placed on a scale of model-to-model variation.
- Part 2 tables use three seeds per condition. Sign counts and ranges over three values are descriptive only.
- The two MSE-pooling variants are both alternatives to the pre-registered aggregate; neither is claimed to be correct.
- 2b residual correlations are constrained by the three-seed centring (see the reading note).
- The E2 angle thresholds are post hoc.
- Per-seed pairing in 2a rests on identical poisoned views and a shared seed by code reading, and GPU runs are not
  bitwise reproducible (D-027).
- Part 3 ran on the laptop GPU (RTX 5060), the training and stored renders on a Quadro RTX 5000; the RGB agreement above
  shows no practical difference for these checkpoints.

## Open questions (not answered here)

1. Which aggregate should the paper report for the target and plate regions, given the statements in 2c that change with
   the choice? Does that need its own decision entry?
2. Should a depth baseline between two independently trained control models be measured before depth differences are
   interpreted at all (needs a decision entry and another control render)?
3. Is a shared seed effect across conditions (2b) worth stating, and with what wording at N = 3?
4. Should per-view spread (2e) and the 2d zero count be reported in the main text or in an appendix?
5. Is a multi-seed version of Part 3 wanted (Phase 9 scope)?

## Assumptions

1. Per-view MSE for the target and collateral regions is recovered from the stored per-view PSNR (10^(-PSNR/10)); plate
   MSE is taken directly from `plate_mse`. No metric was recomputed from images.
2. The 2c pixel weights are the 400x400 region pixel counts in the per-view CSV; plate distance uses the target-region count.
3. The collateral region includes all 75 views; target and plate use views flagged included (all 75 are included).
4. Rank 1 = largest value for every metric, including those where smaller is better (D-044).
5. The 75 eval poses were taken from the clean scene's `transforms_test.json` through the training loader (half
   resolution), not from the poisoned run's dataset; the RGB agreement with the stored renders supports this.
6. Each run's local config was resolved by the basename of the HPC config path in its `summary.json`, under `configs/poisoning/`.
7. Poisoned views for C7 come from the `C7` dataset id in `MANIFEST.csv`; C3 and C5 from `C3_seed1` and `C5_seed1`.
8. The nearest-poisoned angle uses camera positions relative to the origin (0,0,0), as in E2.
9. Part 3 statistics were computed from the float32 arrays in memory, not from the stored float16 copies.
10. Opacity and expected-depth values were not clipped; opacity slightly above 1 (up to 1.0000009) occurs from float32 summation.

## Corrections made during this step

- The first Part 3 run stopped after the control model because the RGB-sanity callback was not wired in; it was fixed and
  the whole script re-run from the start.
- The second Part 3 run did not compute the C7 angle relation (its dataset id is `C7`, not `C7_seed1`) and the opacity
  panel of Fig 11 used an offset axis that was hard to read. Both were fixed and Part 3 was re-run in full; the tables
  and figure above are from the final run.

## Addendum (Phase 9 task, Part 5): depth noise floor — EXPLORATORY, descriptive only

Procedure: the D-044 Part 3 render path (final checkpoints, fine network, perturb 0, fp32, 400x400, 75 views, GPU, ray chunk 2048,
no out-of-memory halving needed; new RGB matched the stored renders at 79.3 dB, max abs diff 0.0002) for seeds 2 and 3 of
control, C3, C5 and C7; seed 1 from the D-044 float16 arrays. All differences below are computed on float16-rounded depth;
recomputing the D-044 quantities that way gives 0.04366 / 0.04847 / 0.04670 (C3 / C5 / C7 vs control, seed 1), equal to the
D-044 float32 values to the digits shown. Quantity: mean over M_dil of |depth_a - depth_b|, summarised over the 75 views
(scene units). Table: `experiments/results/phase8/followup/part5_depth_noise_floor.csv` (script `scripts/depth_noise_floor.py`,
config `configs/analysis/phase9_depth_noise_floor.yaml`).

| comparison | seeds 1-2 | seeds 1-3 | seeds 2-3 | what differs between the two models |
|---|---|---|---|---|
| control vs control | 0.0200 | 0.0193 | 0.0056 | initialisation / batch order (one dataset) |
| C7 vs C7 | 0.0068 | 0.0067 | 0.0055 | initialisation only (one dataset) |
| C3 vs C3 | 0.0460 | 0.0351 | 0.0232 | poisoned view set AND initialisation |
| C5 vs C5 | 0.0204 | 0.0155 | 0.0181 | poisoned view set AND initialisation |

Beside these, the D-044 values (condition at seed 1 vs control at seed 1) were C3 0.0437, C5 0.0485, C7 0.0467.

Descriptive observations: the control-vs-control differences (0.0056-0.0200) and the C3 / C5 same-condition seed-pair differences
(0.0155-0.0460) are of the same order as, or in the C3 case equal to, the D-044 condition-vs-control values; C7 same-condition
pairs are smaller (0.0055-0.0068). The seed-1 control differs from the seed-2 and seed-3 controls by about 3.5x the 2-3 difference.
Limits: three seeds, one scene, one pair per cell, no ground-truth depth; a depth difference is a difference between two models.
Not answered here: how much of the D-044 depth differences is attributable to the poisoning rather than to seed-to-seed variation.
