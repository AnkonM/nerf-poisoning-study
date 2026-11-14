# Phase 9 (laptop) runbook — B100 selected (D-045 / D-046). NOTHING HAS BEEN LAUNCHED.

Queue (one process at a time, in this order): `lp9_C3_seed1` (calibration), `lp9_B100_seed1`, `lp9_B100_seed2`, `lp9_B100_seed3`.
Projected 59.0 h of the 60 h cap (D-046); C3 seeds 2-3 are NOT queued (+29.5 h would exceed the cap).

## Before launching
1. `git status` must be clean (the preflight refuses a dirty tree except under `experiments/logs/`, `experiments/results/`).
2. Plugged in. Nothing else on the GPU (the preflight needs <= 2200 MiB in use; Windows alone held 586-1845 MiB when measured).
3. `.venv/bin/python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 run --dry-run` -> every line `[PASS]`.

## Commands (repo root)
- Launch (survives closing the terminal; detach with Ctrl-b d):
  `tmux new -d -s p9 ".venv/bin/python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 run 2>&1 | tee -a experiments/logs/phase9_stdout.log"`
  (`nohup ... &` also works.) Re-running the same command resumes: completed runs are skipped, a partial run resumes from its latest checkpoint.
- Status: `.venv/bin/python scripts/run_phase9_queue.py --hardware-profile laptop_phase9 status` (its ETA lines use HPC constants; laptop figures are 5.36 it/s and ~14.7 h per run).
- Graceful stop (finishes the CURRENT run first): `... stop`.  Immediate stop (<= 25,000 iterations, ~1.3 h, redone on resume): `... stop --now`.
- After a STOPPED marker (exit 10) investigate, then `... clear-stopped`.
- Attach: `tmux attach -t p9`.

Outputs: `experiments/runs/lp9_*` (git-ignored), `experiments/logs/lp9_*.md`, `experiments/results/results_phase9.csv` (results.csv is never touched).

## Unattended behaviour on this machine (facts read 2026-11-14; consequences marked "untested")
- WSL: `systemd=true` in /etc/wsl.conf, user systemd running, `Linger=no`, tmux and nohup present; `.wslconfig`: memory=20GB, swap=32GB.
- Windows power: Balanced; on AC sleep after = never (0), on battery 300 s. Unplugged => the machine sleeps after 5 min and the run stalls/dies. Display-off is irrelevant to the run.
- Windows Update restarts, `wsl --shutdown`, or the WSL VM shutting down kill the harness and the child. Restart is safe (resume, <= 25k iterations lost, RNG restored, D-039). Whether a background process alone keeps the WSL VM alive after the last terminal closes was NOT tested; to be safe keep one Windows-side session open (e.g. `wsl.exe -- sleep infinity`).
- The preflight fails closed if the GPU, driver, torch build or hostname differ from `configs/phase9/laptop_phase9.yaml` (a Windows driver update therefore halts the queue until the profile is deliberately updated and recorded).

## Findings to know before launching (facts; no change made)
- Validation of 5 views took 464 s (93 s/view) at the config's `chunk_size` 32768; torch reserved 8.27 GiB against 8151 MiB of VRAM. D-044's depth renders of the same size at chunk 2048 took ~12 s/view. The 4.4 h per run of validation + final eval in the cost estimate is therefore probably dominated by that chunk size; a smaller eval chunk was NOT tried and would differ from the HPC config (the lead's decision).
- The speed test ran 2,500 iterations only; the GPU reported `SW Thermal Slowdown` at 85-86 C. Sustained 10-hour speed is unmeasured; the cap margin is 1.0 h.

## Post-run plan (not started)
1. `scripts/render_eval_views.py` and `scripts/compute_metrics.py` on the new runs, into a Phase 9 output location (never into `per_run.csv`, `experiments/renders/` or the Phase 8 outputs).
2. Add C3-laptop and B100 to the analysis under both aggregates (mean-of-PSNR, pooled-MSE PSNR).
3. Test the D-045 F predictions for B100: P1 mean plate PSNR > max of C5's three per-seed values; P2 target PSNR < min of C5's per-seed values; both aggregates; per-view distributions as exploratory.
4. Report the calibration pair (laptop C3 seed 1 vs HPC C3 seed 1) as ONE data point against C3's per-seed spread; no equivalence claim; label every Phase 6 vs Phase 9 comparison as crossing GPU architectures (Turing -> Blackwell); record driver 582.05 and torch 2.14.0+cu130.
5. Update the limitations text.
