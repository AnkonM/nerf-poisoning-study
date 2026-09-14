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
