# Project Structure

This document defines what belongs in every directory and the rules that keep
the repo auditable. If you're about to put a file somewhere not described
here, stop and either find the right place or add a new documented directory —
don't let structure drift silently.

```
nerf-poisoning-study/
├── README.md
├── docs/
│   ├── PROJECT_STRUCTURE.md        # this file
│   ├── ROADMAP.md                  # phased plan + live status tracker
│   ├── METHODOLOGY.md              # locked protocol / "pre-registration"
│   ├── DECISION_LOG.md             # append-only record of non-trivial decisions
│   └── templates/
│       └── EXPERIMENT_LOG_TEMPLATE.md
├── configs/                        # single source of truth for every run
│   ├── base.yaml                   # shared defaults (model, optimizer, renderer)
│   ├── scenes/
│   │   ├── lego_sanity.yaml        # Phase 1 pipeline-verification config
│   │   └── final_scene.yaml        # Phase 4+ main-study scene config
│   └── poisoning/
│       ├── budget_00.yaml          # clean control
│       ├── budget_05.yaml
│       ├── budget_10.yaml
│       ├── budget_20.yaml
│       ├── budget_30.yaml
│       ├── budget_50.yaml
│       ├── soft_suppression_20.yaml
│       └── ablation_random_vs_strategic.yaml
├── data/
│   ├── raw/                        # Blender scene files (.blend), never edited by scripts
│   ├── blender_scenes/             # rendered clean training/eval views + camera poses
│   │   ├── train/
│   │   ├── eval_holdout/           # FROZEN after Phase 4 — see METHODOLOGY.md §4
│   │   └── cameras.json
│   ├── masks/                      # object-ID / alpha masks, one per view, from Blender
│   ├── background_plates/          # target-object-toggled-off renders, one per view
│   └── poisoned/
│       ├── <condition_id>/         # one folder per poisoning condition, script-generated
│       └── MANIFEST.csv            # which views were poisoned, by what recipe, per condition
├── src/
│   ├── nerf/                       # vanilla NeRF model/training code (vendored + adapted)
│   ├── poisoning/                  # compositing scripts: hard erasure, soft suppression
│   │   ├── compositor.py           # implements the fixed formulas from METHODOLOGY.md §3
│   │   └── view_selection.py       # random vs strategic view-selection logic
│   ├── data_pipeline/              # Blender render orchestration, dataset/manifest builders
│   ├── metrics/                    # PSNR/SSIM/LPIPS, masked + unmasked variants
│   └── utils/                      # seeding, config loading, logging helpers
├── scripts/                        # thin CLI entry points, no logic lives here
│   ├── render_scene.py             # calls Blender in headless mode to produce train/eval sets
│   ├── build_poison_set.py         # applies src/poisoning per a condition config
│   ├── train.py                    # trains one NeRF given a resolved config
│   ├── evaluate.py                 # computes metrics for a trained model vs frozen eval set
│   └── run_sweep.sh                # loops train.py + evaluate.py over configs/poisoning/*
├── experiments/
│   ├── logs/                       # one filled EXPERIMENT_LOG_TEMPLATE.md per run
│   └── results/                    # machine-readable metrics, one row per run
│       └── results.csv
├── notebooks/
│   └── exploratory/                # scratch analysis only — nothing here is load-bearing
├── environment/
│   ├── SETUP.md                    # local (WSL2) + cloud (Colab/Kaggle) setup instructions
│   ├── environment.yml             # conda environment, pinned versions
│   └── requirements.txt            # pip fallback / Colab install list
├── tests/                          # unit tests for compositor, metrics, view selection
└── paper/
    ├── draft.md
    └── figures/
```

## Directory contracts

- **`configs/` is the only place hyperparameters and condition definitions
  live.** Scripts read a config path as their only required argument. If a
  number that affects a result is hardcoded in a `.py` file instead of a
  `.yaml` file, that's a bug to fix, not a shortcut to take.
- **`data/raw/` and `data/blender_scenes/eval_holdout/` are read-only after
  Phase 4.** Nothing in `src/` or `scripts/` should ever write to them. If a
  script needs to, that's a sign it's touching the wrong directory.
- **`data/poisoned/` is always script-generated, never hand-edited.** Each
  condition's folder is fully reproducible by rerunning
  `build_poison_set.py` against its config — if you can't regenerate a
  poisoned set from its config alone, the config is incomplete.
- **`experiments/results/results.csv` is append-only and machine-generated**
  by `evaluate.py`. Every row must be traceable to a config file path, a git
  commit hash, and an entry in `experiments/logs/`.
- **`notebooks/` never produces a number that goes in the paper.** Notebooks
  are for looking at things quickly; anything that becomes a reported result
  gets promoted into a script under `scripts/` or `src/metrics/`.
