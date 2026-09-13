# Experiment Log — phase2_lego_sanity

## Identification

- **Run ID:** phase2_lego_sanity
- **Condition:** Phase 2 pipeline-verification sanity check (not a paper condition — clean Blender Synthetic Lego, no poisoning). See `docs/ROADMAP.md` Phase 2.
- **Seed:** 0
- **Config file path:** `configs/scenes/lego_sanity.yaml`
- **Git commit hash:** `d1906094dd7437f5c0bc582d989a35991940df43` (training start); Phase 2 closeout work (this log, D-016, sample renders) done at commit `57d2f52367c7463e0246bc48af15f035fd08c86f` and later
- **Date/time started:** 2026-09-12 18:40 CEST
- **Date/time finished:** training loop finished 2026-09-13 06:34 CEST (11.91h); independent test-set evaluation (see Anomalies below) finished 2026-09-13 ~13:00 CEST

## Environment

- **Where run:** local (WSL2)
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, sm_120, compute capability (12, 0))
- **PyTorch / CUDA versions:** torch 2.14.0+cu130, CUDA 13.0 (driver 582.05)
- **Environment file used:** `pyproject.toml` / `uv.lock` (per `docs/DECISION_LOG.md` D-009/D-010), no modifications since last `uv sync`

## Data

- **Dataset:** `data/nerf_synthetic/lego/` — standard Blender Synthetic Lego dataset (100 train / 100 val / 200 test views), downloaded per `docs/DECISION_LOG.md` D-014. Not the main-study eval set — `data/blender_scenes/eval_holdout/` does not exist yet (Phase 4 has not started), so this field does not apply to this run.
- **Eval set used:** `data/nerf_synthetic/lego/transforms_test.json` split (200 views) — this is a Phase 2 pipeline-verification run, explicitly out of scope for the frozen main-study eval-holdout rule (`docs/METHODOLOGY.md` §4 applies to Phase 4+ only).

## Training

- **Iterations/epochs:** 200,000 / 200,000 (completed, no early stop)
- **Final training loss:** 0.00299 (final train-batch MSE loss at iteration 200,000)
- **Convergence check:** converged — loss decreased monotonically from ~0.3 (iter 10) to 0.003 (iter 200,000); train-batch PSNR rose from ~8 dB to 33.36 dB; val-subset PSNR at iteration 200,000 was 31.63 dB. No divergence, no NaN/Inf observed.
- **Anomalies observed:** yes — the vendored training script's automatic post-training full-test-set render (200 views) took 5.5+ hours on this run instead of the expected ~40-70 min, due to the process running near the 8GB VRAM cap (~7.3-7.6GB used), which was indistinguishable from a hang by log output alone (confirmed genuinely still computing via `/proc/<pid>/stat` CPU-tick deltas across repeated samples, not a guess). The process was killed after its final checkpoint (`200000.tar`, saved during the training loop itself, before the contended render) was confirmed on disk — no training progress was lost. The test-set PSNR reported below was computed independently afterward via a new standalone script (`scripts/evaluate.py`), loading that checkpoint with no retraining, on a fully-free GPU, completing in 38.2 minutes. Full detail in `docs/DECISION_LOG.md` D-016.

## Metrics (computed by `scripts/evaluate.py`)

| Metric | Value |
|---|---|
| Test-set PSNR (200 held-out views, unmasked whole-frame) | **31.550 dB** |
| D-013 acceptance range | 29–33 dB |
| Gate result | **PASS** |
| Final train-batch PSNR (iter 200,000) | 33.359 dB |
| Val-subset PSNR (5 views, iter 200,000, during training) | 31.632 dB |

This project's masked/unmasked target-region PSNR/SSIM/LPIPS metrics
(`docs/METHODOLOGY.md` §6) do not apply here — there is no target object
or mask in the Lego sanity-check scene. This table reports the simpler
whole-frame PSNR that Phase 2's gate (`docs/ROADMAP.md`) actually asks for.

## Notes

This is explicitly **not a paper result** — per `docs/ROADMAP.md` Phase 2
framing, this run exists solely to de-risk the training/render/metric
pipeline before Phase 3+ builds on it. Sample novel-view renders (4 test
views, ground-truth-vs-rendered side by side) are saved at
`experiments/results/phase2_lego_sanity/` for visual inspection. Camera
convention confirmed for later Phase 4 cross-check: standard NeRF-synthetic
Blender convention (right-handed world coords, 4x4 camera-to-world
`transform_matrix`, camera looks down local -Z with +Y up / +X right) —
see `src/nerf/datasets/blender.py`.
