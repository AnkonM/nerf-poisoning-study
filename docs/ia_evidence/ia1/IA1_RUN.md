# IA-1 reproduction commands

Snapshot: annotated tag `ia1-snapshot` -> commit `efa8745` (Phase 3 closeout, 2026-09-13).
Commands below were checked against files at that tag; the poisoning build (step 5) was executed
in a separate worktree and reproduced the logged view selection.

```bash
# 1. clone and check out the snapshot
git clone <repo-url> nerf-poisoning-study && cd nerf-poisoning-study
git checkout ia1-snapshot            # detached HEAD, expected

# 2. environment (uv, Python 3.12, cu130 torch wheels per pyproject.toml)
uv sync

# 3. GPU/environment check (needs a CUDA GPU; expects compute capability (12, 0) on the dev laptop)
uv run python scripts/verify_env.py

# 4. compositor tests: NONE at this tag (no tests/ directory; pytest suite arrives in Phase 5, commit 9d283fb)

# 5. rebuild one poisoned set (20% budget, hard erasure, seed 0)
#    Requires the Blender Synthetic Lego data at data/nerf_synthetic/lego/ (gitignored, not in the clone).
uv run python scripts/build_poison_set.py configs/poisoning/phase3_poc_budget_20.yaml
#    Expected: |V_target| = 100, num_poisoned = 20, selected views
#    [1, 3, 6, 15, 22, 26, 42, 46, 52, 53, 55, 57, 59, 61, 68, 70, 73, 83, 92, 93],
#    100 images in data/poisoned/phase3_poc_budget_20/train, rows appended to data/poisoned/MANIFEST.csv
```

Not included: training/evaluation reruns (30k iterations per condition; Phase 2 run took ~12 h). Their
logs are in `experiments/logs/`.

Regenerate the figures from tracked summaries: `.venv/bin/python docs/ia_evidence/ia1/make_figures.py`
(run from repo root of the main tree; reads `experiments/results/phase3_poc/*_4view_summary.json`).
