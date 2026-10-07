# Environment Setup

Two environments, kept in sync via `pyproject.toml`/`uv.lock` (local) and
`requirements.txt` (cloud): **local (WSL2)** for development/debugging/small
runs, **cloud (Colab/Kaggle)** for the bulk of the Phase 6 sweep. See
`DECISION_LOG.md` D-005, D-006, and D-009 for why this split, why vanilla
NeRF is the default model, and why `uv` (not conda) is the local package
manager.

## 1. Local: WSL2 on Windows 11

### 1.1 Install WSL2 + Ubuntu

```powershell
# In an elevated PowerShell
wsl --install -d Ubuntu-24.04
```
Reboot if prompted, then launch Ubuntu from the Start menu and complete the
initial user setup.

### 1.2 Confirm the GPU is visible inside WSL2

```bash
nvidia-smi
```
This should list the RTX 5060 laptop GPU. If it doesn't, update the
**Windows-side** NVIDIA driver (WSL2 uses the Windows driver directly — do
not install a separate Linux NVIDIA driver inside WSL2).

### 1.3 Install PyTorch with explicit CUDA 12.8+ wheels

The RTX 5060 laptop GPU is Blackwell architecture (sm_120, compute
capability 12.0). Stable PyTorch releases only gained native sm_120 support
from version 2.7.0 with CUDA 12.8 wheels — installing PyTorch without
pinning at least this explicitly risks silently falling back to an
unsupported kernel path.

**Verified minimum: PyTorch 2.7+/cu128.** This project's `pyproject.toml`
currently pins `cu130` wheels (torch 2.14.0+cu130), which has been verified
working on this hardware (RTX 5060 laptop, compute capability (12, 0) — see
§1.4 and `DECISION_LOG.md` D-009). cu128 remains the documented floor for
anyone building the environment on older/pinned wheels; it is not a
required exact match — any cu128+ build that passes the §1.4 check is fine.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 1.4 Verify Blackwell is actually being used

Run this before writing any research code — an environment bug here is far
cheaper to catch now than after Phase 2:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("Matrix multiply: PASSED")
```
Expected: compute capability `(12, 0)`, no `CUDA error: no kernel image is
available for execution on the device`.

### 1.5 Clone repo, create environment

This project uses `uv` (not conda) to manage the local Python environment —
see `DECISION_LOG.md` D-009. `pyproject.toml` + `uv.lock` are the source of
truth for pinned versions.

```bash
git clone <repo_url> nerf-poisoning-study
cd nerf-poisoning-study
uv sync
source .venv/bin/activate
```

`uv sync` creates `.venv/` (if it doesn't already exist) and installs
exactly the locked dependency versions from `uv.lock`. Run
`python scripts/verify_env.py` from inside the activated `.venv` (or
`uv run python scripts/verify_env.py` without activating) to confirm the
GPU is actually being used — see §1.4.

### 1.6 Known risk: tiny-cuda-nn / Nerfacto

If Phase 9 pursues the Nerfacto cross-model check, `tiny-cuda-nn` (a
Nerfstudio dependency) has had reported build failures on sm_120 as of late
2025. Do not sink debugging time into this locally — run it on a
prebuilt/cloud environment instead if it doesn't build cleanly on the first
attempt (see D-005).

## 2. Cloud: Colab / Kaggle

Both are used as overflow capacity for the Phase 6 sweep so full-length
training runs don't tie up the laptop.

### 2.1 Setup (either platform)

```python
!pip install -r requirements.txt
```
`requirements.txt` must resolve to the same versions as `pyproject.toml`/
`uv.lock` — if you change one, change the other in the same commit, or the
"identical environment everywhere" guarantee this split depends on breaks
silently. Colab/Kaggle GPUs are not Blackwell/sm_120, so the cu128 floor
(not the cu130 pin verified in §1.3) is what actually gets exercised there;
this is untested and tracked as an open item until the first cloud run (see
`docs/ROADMAP.md` Phase 0 status tracker).

### 2.2 Repo access

Mount the repo via `git clone` at the top of the notebook (using a
read/deploy token, not personal credentials pasted into a shared notebook).
Do not maintain a second copy of `src/` pasted into notebook cells — that's
exactly the kind of untracked divergence `PROJECT_STRUCTURE.md` rules
against.

### 2.3 Which platform for what

- **Colab:** convenient default, GPU tier varies by session/subscription.
- **Kaggle:** larger free weekly GPU-hour allowance — prefer for the bulk of
  the Phase 6 sweep once the pipeline is confirmed working on Colab.

### 2.4 Getting results back

Every cloud run still produces an `experiments/logs/<run_id>.md` entry and a
`results.csv` row — commit these back to the repo (or sync via the repo's
remote) before the notebook session ends. A run whose log never makes it
back to the repo did not happen, as far as this project's audit trail is
concerned.

## 3. Blender: native Windows install, not WSL2

Blender is installed and configured manually — nothing here is scriptable
into place — and it lives **on Windows, not inside WSL2**. See
`DECISION_LOG.md` D-008 for the reasoning; summary: Blackwell/OptiX support
in Blender is mature and driver-based (no sm_120 wheel-pinning fights like
PyTorch), and the interactive scene-building GUI is a poor fit for WSL2/WSLg.

### 3.1 Install

- Download the latest stable Blender from blender.org (not an old LTS —
  current Blackwell/OptiX support matters here).
- Install/update the NVIDIA **Studio** driver (not just Game Ready) on the
  Windows side.

### 3.2 Configure GPU rendering

- Blender Preferences → System → set Cycles device to **OptiX** (preferred
  over CUDA on RT-core hardware).
- Confirm the RTX 5060 laptop GPU appears and is checked under that device
  list.

### 3.3 Interactive scene building

Build/edit `data/raw/*.blend` scene files in the normal Windows GUI.

### 3.4 Scripted dataset rendering (headless)

The `original` / `background_plate` / `mask` triples and camera-pose JSON
(`METHODOLOGY.md` §3–4) are produced by running Blender in background mode.
The Blender *process* is a Windows process either way (D-008), but it can be
**launched and orchestrated from a WSL2 shell** — verified 2026-09-14, see
`DECISION_LOG.md` D-021. This is the form Phase 4+ scripts use, so the whole
pipeline can be driven from one shell:

```bash
"/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" \
    --background --factory-startup \
    "$(wslpath -w data/raw/final_scene.blend)" \
    --python "$(wslpath -w scripts/render_scene.py)" -- <args>
```

Notes (all verified, D-021):

- Blender is **not** on the inherited Windows PATH inside WSL2 — the explicit
  `/mnt/c/...` path is required. Version in use: **Blender 5.2.1 LTS**.
- Pass WSL2-resident paths through `wslpath -w` (`\\wsl.localhost\...`);
  Blender reads and writes them fine.
- `--factory-startup` is deliberate: without it the render depends on
  whatever GUI preferences are saved on the local machine, which is
  untracked state and breaks reproducibility.
- When enabling OptiX in a render script, filter devices on
  `d.type == 'OPTIX'` specifically. This machine also lists an AMD Radeon
  780M iGPU (HIP) and the RTX 5060 again under CUDA; enabling everything
  can push work onto the iGPU.

### 3.5 Bridging Windows-rendered output into the WSL2 repo

Pick **one** of these and use it consistently (don't mix):

- **UNC path:** point the render script's output directly at
  `\\wsl.localhost\Ubuntu-24.04\home\<user>\nerf-poisoning-study\data\...`
  so no copy step is needed.
- **Explicit sync:** render to a local Windows folder, then
  `robocopy`/`rsync` into `data/blender_scenes/` and `data/masks/` inside
  WSL2 as a documented step before Phase 4/5 scripts run.

**DECIDED (2026-09-14, `DECISION_LOG.md` D-021): the UNC path.** Blender
writes directly into the WSL2 repo; there is no copy step, and no Phase
4/5 script should add one. Measured cost of this choice: a UNC write is
~9.6 ms/file vs ~1.6 ms/file Windows-native, i.e. **~8 ms extra per
file**, which across Phase 4's ~545 output files totals **~4 seconds** —
against a ~2 s/frame render, and below the ~20% run-to-run variance that
GPU thermal/boost state already introduces. The 4 seconds buys a render
that is fully reproducible from a script alone, with no hand-run step
outside version control (the failure mode logged in D-020).

Convert a repo path to its UNC form with `wslpath -w`:

```bash
wslpath -w /home/ankon/projects/nerf-poisoning-study/data
# -> \\wsl.localhost\Ubuntu-24.04\home\ankon\projects\nerf-poisoning-study\data
```

## 4. Sanity check before Phase 1 begins

- [ ] `nvidia-smi` shows the RTX 5060 in WSL2.
- [ ] The compute-capability verification script above prints `(12, 0)` and
  "PASSED".
- [ ] `uv sync` installs cleanly locally, and `requirements.txt` installs
  cleanly and produces a compatible PyTorch/CUDA build on Colab and Kaggle
  (unverified as of Phase 0 — see `docs/ROADMAP.md` status tracker).
- [ ] Blender (native Windows install) opens, shows the RTX 5060 under
  OptiX in Preferences, and `blender.exe --background scene.blend --python
  scripts/render_scene.py` runs successfully from a Windows terminal
  (needed for Phase 4's scripted scene rendering).

## 5. Frozen eval set: re-apply read-only permissions after a fresh clone

D-026 froze `data/blender_scenes/transforms_test.json` (plus the three
`eval_holdout/` directories) as read-only. Git stores only the executable bit,
not write permission, so a fresh clone leaves `transforms_test.json`
**writable** even though it is part of the frozen 226-file set. After every
fresh clone (and after restoring the ignored-files zip), run:

```bash
chmod a-w data/blender_scenes/transforms_test.json
python scripts/freeze_eval_set.py --verify   # must report PASS
```

`--verify` prints a non-failing `WARNING` while any frozen file is still
writable; the PASS/FAIL verdict depends on content only.

## 6. Native Ubuntu 24.04 HPC machine (Phase 6 sweep platform)

Added after the repo moved off the WSL2 laptop (D-036). Sections 1–3 above
describe the original laptop and remain accurate for it; this section
describes the machine that runs Phase 6.

- **OS / hardware:** native Ubuntu 24.04, kernel 7.0 (HWE), 24 threads
  (Xeon Silver 4116), ~78 GiB RAM, HDD-backed root (`/dev/sdb2`, 3.6 TB).
  No WSL2, no Windows.
- **GPU:** Quadro RTX 5000, 16 GB, Turing, compute capability (7, 5),
  NVIDIA driver 580.173.02 (nvidia-driver-580 from the Ubuntu archive).
  `scripts/verify_env.py` reports the capability for information only; its
  gate is CUDA available + GPU matmul. The GNOME desktop runs on this same
  GPU (a display is attached), so a little VRAM is used by Xorg/gnome-shell.
- **uv:** installed to `~/.local/bin` (`uv`, `uvx`) by the official
  installer. `~/.profile` already prepends `~/.local/bin` to `PATH` when that
  directory exists, and `~/.bashrc` exports it too, so no profile edit was
  needed. Environment: `uv sync --frozen` from the repo root. If the repo is
  ever moved, delete `.venv` and re-run `uv sync --frozen`: the venv's console
  scripts embed absolute paths and `uv sync` does not rewrite them.
- **Frozen eval set permissions:** see §5 — `chmod a-w
  data/blender_scenes/transforms_test.json` after every fresh clone, then
  `python scripts/freeze_eval_set.py --verify`.
- **Not needed for Phase 6:** WSL2 and Blender. All renders, masks, plates and
  poisoned datasets are restored from the ignored-files zip as literal files
  (D-027: re-rendering is not bit-exact), never regenerated.
- **Git:** this machine has no git identity, credential helper or SSH key
  configured; commits here use a per-command `-c user.name/-c user.email` and
  pushing requires credentials to be set up by the project lead.

## 7. Running the Phase 6 sweep (harness, service, tmux fallback)

The harness is `scripts/run_sweep.py` (logic in `src/utils/sweep_*.py`, D-040). It runs
the 24 runs seed-major, **one training process at a time**, and derives all state from
disk, so it is safe to stop, kill or reboot at any point and simply start it again.
Output goes to `experiments/runs/<run_id>/` (checkpoints, `train.log`, events),
`experiments/logs/<run_id>.md` + `<run_id>_loss.png`, and `experiments/results/results.csv`.

**Nothing below has been run for the real sweep. The service unit is installed but NOT
enabled and NOT started.**

### Machine facts checked on 2026-10-06
- `loginctl show-user`: **Linger=yes** — user services survive logout and start at boot,
  so no sudo is needed for the service route.
- `tmux` is installed (`/usr/bin/tmux`).
- GPU persistence mode on, CPU governor `performance`, NVIDIA driver/kernel packages
  held (D-038). The harness's preflight does not re-check these, but its hardware pin
  fails the sweep closed if the GPU, driver or torch build ever changes.

### Option A (recommended): systemd user service
```bash
# one-time install (already done; the unit is disabled and inactive)
mkdir -p ~/.config/systemd/user
cp environment/nerf-sweep.service ~/.config/systemd/user/
systemctl --user daemon-reload

# start the real sweep now (does not survive a reboot unless also enabled)
systemctl --user start nerf-sweep.service
# start now AND on every boot
systemctl --user enable --now nerf-sweep.service

systemctl --user status nerf-sweep.service      # is it up
python scripts/run_sweep.py status              # the 24-run table, ETAs, GPU, disk, STOPPED marker
journalctl --user -u nerf-sweep -f              # live harness log (also in experiments/logs/sweep_harness.log)
tail -f experiments/runs/<run_id>/train.log     # the current run's training log
```
Restart policy: exit 0 (done / graceful stop) and exit codes 10 (STOPPED by policy) and 11
(lock held) are **not** restarted; a harness crash (exit 1 / signal) is restarted after
120 s, and 5 failures in an hour stop the restarts. After that: `systemctl --user
reset-failed nerf-sweep.service`.

### Option B: tmux (no systemd)
```bash
tmux new -s sweep -d 'cd ~/nerf-poisoning/nerf-poisoning-study && .venv/bin/python -u scripts/run_sweep.py run 2>&1 | tee -a experiments/logs/sweep_tmux.log'
tmux attach -t sweep          # look; detach with Ctrl-b d
tmux kill-session -t sweep    # hard stop (see "stopping")
```
Without tmux or systemd, `setsid nohup .venv/bin/python -u scripts/run_sweep.py run >> experiments/logs/sweep_nohup.log 2>&1 &`
survives a closed shell or SSH disconnect (not a reboot). Launch it from your own terminal,
not from inside a Claude Code session.

### Stopping, and what it costs
- **Graceful (recommended):** `python scripts/run_sweep.py stop` — the harness finishes the
  *current* run (training, evaluation, log entry, results row), then exits before the next
  one. Costs nothing, but can take up to ~10 h.
- **Now:** `python scripts/run_sweep.py stop --now`, `systemctl --user stop nerf-sweep`, or
  killing tmux/the process. The training child is stopped immediately; the run stays PARTIAL
  and **resumes from its latest checkpoint** next start. Checkpoints are written every
  25,000 iterations, so this redoes **up to 25,000 iterations ≈ 71 min** (at the measured
  5.83 it/s). Resume restores RNG state, so the finished run is bit-identical to an
  uninterrupted one (D-039); every resume is recorded in the run's log entry anyway.
- **After a STOPPED marker** (`experiments/runs/STOPPED`, also shown by `status`): the queue
  halted on purpose — a diverged/NaN run, a failed preflight or validation. Investigate the
  reason, then `python scripts/run_sweep.py clear-stopped`. A FAILED run additionally stays
  FAILED until its run directory is moved aside (it is never silently retried or skipped).
- **Before starting for real:** `python scripts/run_sweep.py run --dry-run` prints the queue
  state and runs the preflight read-only. The working tree must be clean (the harness's own
  outputs under `experiments/logs/` and `experiments/results/` are exempt), so commit
  or stash anything else first.
