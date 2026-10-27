# Phase 6 sweep: results

**24 of 24 runs complete (8 conditions x 3 seeds); all 200000 iterations.** control_seed3 was interrupted once and resumed from iteration 150000 (resume_events=1).

**Metric source:** `experiments/results/results.csv` (copied to `snapshot/results.csv`), column `test_psnr_mean_whole_frame_db` = whole-frame PSNR over the 75 frozen eval-holdout views (from `scripts/evaluate.py`). Std is the sample std (ddof=1).

**Not available:** results.csv contains no masked/target-region suppression metric and no collateral-damage metric (the per-run eval JSON has only `test_psnr`). Therefore no suppression or collateral numbers are reported.

| condition | budget (% of V_target) | attack | view selection | seeds completed | whole-frame test PSNR, dB (mean ± std) |
|---|---|---|---|---|---|
| control | 0 | - | - | 3/3 (1,2,3) | 50.23 ± 0.19 (n=3) |
| C1 | 5 | hard_erasure | random | 3/3 (1,2,3) | 46.71 ± 0.50 (n=3) |
| C2 | 10 | hard_erasure | random | 3/3 (1,2,3) | 42.53 ± 0.69 (n=3) |
| C3 | 20 | hard_erasure | random | 3/3 (1,2,3) | 39.85 ± 0.81 (n=3) |
| C4 | 30 | hard_erasure | random | 3/3 (1,2,3) | 37.89 ± 0.72 (n=3) |
| C5 | 50 | hard_erasure | random | 3/3 (1,2,3) | 34.34 ± 0.18 (n=3) |
| C6 | 20 | soft_suppression | random | 3/3 (1,2,3) | 43.31 ± 0.91 (n=3) |
| C7 | 20 | hard_erasure | strategic | 3/3 (1,2,3) | 43.55 ± 0.42 (n=3) |

## Per-run whole-frame test PSNR (dB)

| run | PSNR |
|---|---|
| control_seed1 | 50.078 |
| C1_seed1 | 46.791 |
| C2_seed1 | 41.918 |
| C3_seed1 | 39.099 |
| C4_seed1 | 37.251 |
| C5_seed1 | 34.364 |
| C6_seed1 | 42.282 |
| C7_seed1 | 43.761 |
| control_seed2 | 50.452 |
| C1_seed2 | 47.166 |
| C2_seed2 | 42.387 |
| C3_seed2 | 40.707 |
| C4_seed2 | 37.745 |
| C5_seed2 | 34.496 |
| C6_seed2 | 43.996 |
| C7_seed2 | 43.828 |
| control_seed3 | 50.171 |
| C1_seed3 | 46.168 |
| C2_seed3 | 43.284 |
| C3_seed3 | 39.758 |
| C4_seed3 | 38.676 |
| C5_seed3 | 34.149 |
| C6_seed3 | 43.649 |
| C7_seed3 | 43.061 |

## Observations (numbers only)

- Hard-erasure random, whole-frame PSNR by budget 0/5/10/20/30/50%: 50.23 / 46.71 / 42.53 / 39.85 / 37.89 / 34.34 dB; strictly decreasing across the 6 means: True.
- Whole-frame PSNR falls with budget, but whole-frame PSNR mixes target-region and non-target-region error; it cannot be read as suppression or collateral separately.
- At 20% budget: random hard (C3) 39.85, soft suppression (C6) 43.31, strategic hard (C7) 43.55 dB (n=3 each). C6 vs C7 differ by 0.24 dB, within seed noise.
- Seed range (max-min) per condition, dB: control 0.37, C1 1.00, C2 1.37, C3 1.61, C4 1.42, C5 0.35, C6 1.71, C7 0.77. Differences under ~1 dB are not resolved.
