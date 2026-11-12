# Phase 8 closeout checks (2026-11-12)

Read-only on data. No metric was recomputed, nothing was trained or re-rendered, nothing was pushed. `per_run.csv`, the
metrics outputs, renders, checkpoints and frozen data were not modified. Findings are stated without conclusions about the
research results. Assumptions are listed at the end.

## 1. Figures 1, 4, 6a, 6b, 8a (PNG viewed)

| Figure | Overlapping labels | Clipped titles / elements | Unreadable text | Axes notes |
|---|---|---|---|---|
| 1 | none | none | none | Independent y-axis per panel; none starts at zero (PSNR ~17-41, SSIM 0.70-1.00, LPIPS from 0). x-axis is linear in budget %. Raw-mask sensitivity points sit almost on top of the M_dil points at several budgets, so some grey markers are partly hidden. |
| 4 | none | **C6 target-PSNR sample-std bar was cut off by the top of the axes** (y-limit was computed from the per-seed points only). Fixed, see below. | none (footnote and tick-label text are small, 7.5 pt) | Independent y-axis per panel, none from zero. The y-limits are shared with Fig 5 by design. |
| 6a | none | none; title ends inside the canvas | column titles (8 pt) and rotated row labels are small but legible | Image panels, no axes. Figure is 1329 x 4485 px. Seed 1 only. |
| 6b | none | title ends about 9 px short of the right edge of the canvas (not cut off) | as 6a | The crop window is the M_dil bounding box + 20 px, so magnification differs per column and there is no scale bar. 400 x 400 renders are enlarged with visible pixel blocks in some crops. |
| 8a | none; both legends sit in empty areas | none | none | Independent y-axis per panel (about 7-42 dB and 8-41 dB). Violin outlines are kernel-density shapes, not data. Dense overlapping points at control. Title carries the EXPLORATORY label. |

**Fix applied (layout only).** `scripts/analyze_phase8.py`: the shared y-limits for the pair figures now also include
mean +/- sample std of C3, C6 and C7 (commit 002f642). No plotted value changed. Fig 5 uses the same limits, so it was
regenerated too; I viewed Fig 4 after regeneration but did not re-open Fig 5.

**After regeneration** (`python scripts/analyze_phase8.py`, 28 s, rc 0):
- `git status` shows no change under `experiments/results/phase8/` (all CSV tables byte-identical).
- PNGs changed only for Fig 4 and Fig 5. The other PNGs are byte-identical. The PDFs show as modified; I did not inspect
  why (assumed embedded metadata).
- All 11 sidecar JSONs: `per_run_sha256` = `ba10c20623a59e9b97ef6da494d6b94146ed818101dbaf8d0f740aa8a242ac40`, equal to the
  current `sha256sum` of `experiments/results/metrics/per_run.csv`. The only sidecar field that changed is `git_commit`
  (d987c9a -> 002f642); `git_dirty_paths` is empty.

Commands: `Read` on the five PNGs; `sha256sum paper/figures/* experiments/results/phase8/*` before; `python scripts/analyze_phase8.py`;
`git status --porcelain`; `git diff -U0 -- 'paper/figures/*.json'`; `grep -h per_run_sha256 paper/figures/*.json | sort | uniq -c`.

## 2. Shared-seed check

**How the code seeds things** (`src/nerf/training.py`, `src/utils/seeding.py`, `scripts/train.py`):
- `set_seed(seed)` seeds python `random`, `numpy` and `torch` (CPU and all CUDA devices). It runs at the start of training,
  before the data are loaded and before `_build_model`. `load_blender_data` was not seen to use any RNG.
- Model initialisation: the PyTorch default initialiser after `torch.manual_seed(seed)`. Same architecture config and the
  same seed give the same initial weights across conditions.
- Training-image choice: `np.random.choice(i_train)`. Pixel-batch choice: `np.random.choice(H*W, size=n_rand, replace=False)`.
  Both draw from the numpy RNG and neither depends on image content. `i_train` has 100 views in every run (all `train.log`
  show `train/val/test counts: 100/10/75`). Stratified sampling noise uses `torch.rand` on the GPU with identical shapes.
- So, by code reading, init and the sequence of (image index, pixel indices) are the same across conditions at one seed,
  provided nothing else consumes RNG differently. What differs between conditions at the same seed is the pixel content of the
  training images (and the poisoned view set). This is a statement about the code, not a bitwise guarantee: GPU
  non-determinism exists (D-027) and the runs were not compared bit for bit.

**What the logs show.** TensorBoard `train/loss` is logged every 500 iterations (`log_every: 500`), and it is the loss of
the single batch at that iteration, not an average. The first logged value is therefore at iteration 500, after 500
optimiser steps. Initial-iteration losses are **not available**, so identity at the first iterations cannot be tested from these logs.

Loss at the first logged iterations (all 8 conditions per seed; values from `experiments/runs/<run>/tb`):

| seed | iter | min across conditions | max across conditions |
|---|---|---|---|
| 1 | 500 | 0.03022 (C7) | 0.03496 (C6) |
| 1 | 1000 | 0.00309 (C1) | 0.00504 (C2) |
| 1 | 1500 | 0.00117 (control) | 0.00165 (C3) |
| 2 | 500 | 0.00582 (control) | 0.00811 (C4) |
| 2 | 1000 | 0.00257 (control) | 0.00374 (C4) |
| 2 | 1500 | 0.00232 (control) | 0.00355 (C4) |
| 3 | 500 | 0.00424 (control) | 0.00610 (C3) |
| 3 | 1000 | 0.00343 (control) | 0.00603 (C5) |
| 3 | 1500 | 0.00146 (control) | 0.00229 (C5) |

- At iteration 500 the loss is not identical across conditions at a given seed (coefficient of variation across the 8 conditions: seed 1 0.05, seed 2 0.12, seed 3 0.10).
- At iteration 500 the mean across conditions is 0.0326 (seed 1), 0.0069 (seed 2) and 0.0053 (seed 3).
- Correlation of log10(loss) over the first 20 logged points (iterations 500-10000): same-seed, different-condition pairs
  (n = 84): mean r = 0.914 (range 0.682-0.992); different-seed pairs (n = 192): mean r = 0.749 (range 0.518-0.882). Both
  include the common downward trend of the loss, so r is high in both groups.
- `control_seed3` has two event files (resumed run); for each step the value from the first file was used.

Commands: `python -I scratchpad/tb_early.py` (EventAccumulator over `experiments/runs/*/tb`), then a short numpy script for the table and correlations; `grep` on `train.log`.

## 3. Selection nesting (`data/poisoned/MANIFEST.csv`, rows with `poisoned == True`, per `dataset_id`)

| seed | C1⊂C2 | C2⊂C3 | C3⊂C4 | C4⊂C5 | \|C1∩C2\| | \|C2∩C3\| | \|C3∩C4\| | \|C4∩C5\| | C3 == C6 (sets) | \|C3∩C7\| |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | no | no | no | no | 1/5 | 2/10 | 6/20 | 11/30 | yes | 4 |
| 2 | no | no | no | no | 1/5 | 2/10 | 6/20 | 17/30 | yes | 3 |
| 3 | no | no | no | no | 1/5 | 3/10 | 8/20 | 17/30 | yes | 6 |

- "A⊂B" = every view poisoned in A is also poisoned in B, at that seed. No C1..C5 pair is nested at any seed.
- C3 and C6 poison the identical set of 20 views at each seed (20/20 overlap).
- Cross-seed overlap of the same condition: C1: 0, 0, 0 of 5 (seeds 1-2, 1-3, 2-3). C3: 6, 4, 6 of 20. C5: 24, 26, 23 of 50.
- Mechanism: `src/poisoning/view_selection.py::sample_random` calls `np.random.default_rng(seed).choice(v_target, size=k, replace=False)`
  with a different `k` per budget, so each budget is an independent draw. The nesting result above comes from the manifest, not only from this code reading.
- C7 (strategic, one dataset for all seeds) poisons views 1, 4, 11, 15, 17, 18, 24, 25, 30, 32, 33, 37, 41, 42, 45, 51, 54, 55, 59, 62.

Command: a short Python script over the manifest (`csv.DictReader`, set operations).

## 4. Resume of control_seed3

- `experiments/runs/control_seed3/train.log`: first segment started at line 2 (seed 3, commit 76fbeba); line 353
  `Reloading from checkpoint: .../control_seed3/150000.tar`; line 355 `Restored RNG state; continuing at iteration 150001`.
- `harness_events.jsonl`: `interrupted` at last logged iteration 160500 (2026-10-14T09:02:42Z); `resume` at 2026-10-23T08:53:04Z
  from `150000.tar`, `from_iteration` 150000, same commit 76fbeba.
- `summary.json`: `resume = {from_checkpoint: 150000.tar, from_iteration: 150000, rng_restored: true}`.
- The checkpoint `150000.tar` itself (loaded read-only on CPU) contains `rng_state` (keys numpy, python, torch_cpu, torch_cuda),
  `iteration` = 150000 and `global_step` = 149999. The code path `_resume_state` uses the `rng_state` branch when that key is present.
- The log shows the RNG-restoring path, not the legacy path. The "bit-equivalent to an uninterrupted run" statement is taken
  from `summary.json` and D-039; I did not re-verify it.
- Iterations 150001-160500 were run in both segments, so the TensorBoard data has two event files with overlapping steps there.

Commands: `grep -n -i "resum|reload|restor" train.log`, `harness_events.jsonl`, `summary.json`, and `torch.load(..., map_location="cpu")` on the checkpoint, keys only.

## 5. Per-view spread (75 views, from `experiments/results/metrics/per_view/<run>.csv`)

Percentiles use `numpy.percentile` (linear interpolation). Values in dB.

| run | metric | min | p10 | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| C5_seed1 | plate_psnr | 8.342 | 10.777 | 12.403 | 15.938 | 20.676 | 24.525 | 27.551 |
| C5_seed1 | tgt_psnr | 9.576 | 11.683 | 13.506 | 17.560 | 21.624 | 24.981 | 27.777 |
| control_seed1 | plate_psnr | 9.755 | 10.733 | 11.216 | 12.078 | 13.217 | 13.496 | 13.610 |
| control_seed1 | tgt_psnr | 33.008 | 36.670 | 38.070 | 38.962 | 39.847 | 40.461 | 41.139 |

Views sorted ascending (view index : dB):

**C5_seed1, plate_psnr, ascending (eval view index: dB)**

`5:8.34  0:9.39  1:9.65  35:9.71  45:10.12  7:10.59  38:10.69  52:10.73  34:10.84  14:10.87  60:11.22  49:11.30  47:11.40  41:11.53  23:11.67  27:11.79  44:11.95  9:11.97  37:12.33  3:12.48  51:12.57  26:12.71  43:12.77  59:12.80  61:13.06  33:13.27  53:13.60  20:14.00  16:14.32  48:14.40  22:14.42  31:14.45  36:14.64  73:15.31  12:15.60  55:15.65  50:15.92  65:15.94  40:16.36  19:16.47  56:16.49  66:16.72  58:17.55  30:17.68  11:17.76  57:17.78  25:18.07  69:18.08  32:18.15  10:18.36  29:18.38  6:18.69  4:18.83  24:19.13  63:19.16  62:20.67  70:20.69  46:21.39  54:21.40  67:21.49  72:21.59  13:21.89  15:22.50  28:23.61  18:23.85  42:24.28  21:24.52  68:24.53  8:24.74  64:25.16  2:25.17  17:25.21  39:25.37  71:25.37  74:27.55`

**C5_seed1, tgt_psnr, ascending (eval view index: dB)**

`17:9.58  68:10.62  67:10.69  25:11.04  72:11.13  71:11.21  46:11.50  70:11.67  15:11.70  21:11.76  8:12.01  13:12.27  2:12.67  74:12.75  64:12.88  24:13.06  29:13.23  28:13.44  18:13.48  39:13.53  54:13.90  12:14.13  36:14.33  10:14.36  4:14.45  30:14.60  42:14.70  66:14.84  57:15.35  5:15.37  40:15.46  20:16.50  63:16.74  69:16.98  62:17.10  6:17.33  0:17.43  58:17.56  55:17.68  19:17.82  73:18.06  1:18.29  50:18.58  32:19.04  56:19.48  48:19.49  43:19.53  16:19.56  53:19.67  31:20.06  23:20.10  26:20.49  9:21.18  11:21.22  65:21.34  14:21.38  52:21.87  3:22.72  41:23.04  59:23.29  45:23.57  7:23.63  33:23.85  51:23.93  49:23.93  60:24.16  22:24.56  35:25.26  38:25.85  47:25.98  37:26.20  61:26.76  44:26.85  27:27.01  34:27.78`

**control_seed1, plate_psnr, ascending (eval view index: dB)**

`12:9.75  17:10.16  20:10.17  25:10.40  21:10.43  29:10.59  23:10.60  36:10.69  15:10.80  40:10.90  24:10.94  38:11.02  9:11.03  35:11.04  43:11.04  1:11.05  27:11.05  10:11.06  48:11.21  30:11.23  51:11.27  14:11.33  5:11.37  44:11.49  59:11.49  53:11.49  55:11.53  31:11.56  64:11.69  50:11.71  34:11.77  68:11.78  45:11.79  66:11.81  57:11.94  70:12.04  16:12.05  7:12.08  71:12.10  47:12.23  74:12.42  73:12.45  67:12.46  28:12.47  19:12.59  60:12.61  52:12.74  22:12.77  0:12.95  33:12.95  41:13.07  72:13.12  65:13.12  26:13.15  4:13.18  49:13.20  37:13.24  13:13.27  39:13.29  61:13.30  56:13.37  54:13.38  8:13.39  63:13.42  69:13.42  46:13.47  2:13.49  32:13.50  3:13.50  62:13.51  58:13.54  18:13.57  42:13.57  6:13.59  11:13.61`

**control_seed1, tgt_psnr, ascending (eval view index: dB)**

`1:33.01  5:34.13  0:35.76  6:36.10  7:36.27  74:36.52  73:36.57  12:36.60  3:36.78  14:36.84  71:37.21  9:37.37  64:37.72  48:37.82  68:37.89  17:37.93  52:37.96  4:37.99  70:37.99  55:38.15  16:38.17  25:38.23  65:38.23  20:38.24  36:38.28  2:38.52  66:38.55  34:38.55  11:38.63  56:38.63  67:38.64  19:38.70  40:38.77  39:38.81  59:38.87  72:38.88  30:38.89  44:38.96  10:39.01  63:39.02  32:39.02  43:39.04  21:39.04  29:39.05  24:39.06  50:39.11  8:39.18  13:39.29  69:39.35  62:39.38  22:39.41  53:39.48  58:39.52  38:39.61  23:39.66  57:39.77  31:39.92  46:39.95  51:40.01  26:40.08  27:40.11  35:40.19  33:40.21  42:40.27  18:40.30  54:40.37  61:40.42  60:40.49  15:40.51  37:40.53  28:40.57  45:40.68  47:40.94  41:41.05  49:41.14`

Command: a short Python script over the two per-view CSVs (columns `plate_psnr`, `tgt_psnr`).

## Assumptions

1. Figures were judged from the PNGs at the size the viewer displayed (the tall figures were downscaled), so small-text legibility is a judgement at that scale.
2. "Layout fix" was read as: change axis limits only so that already-plotted elements (the std bars) are inside the axes.
3. Fig 5 and the PDF versions were regenerated by the same script run; only Fig 4 was re-viewed.
4. Per-step loss values for `control_seed3` use the first event file; the two files overlap in steps 150500-160500 only, beyond the iterations compared.
5. "Near-identical" has no threshold here; the numbers are reported without one.
6. Nesting and equality were tested on the `view_index` column of the manifest for rows where `poisoned` is `True`.
7. Percentiles use numpy's default linear interpolation.
8. The `git_commit` in the sidecars is the commit at generation time (002f642), not the commit at which the figure code last changed content.
