# Targeted Training-Data Poisoning of Neural Radiance Fields

Empirical security study measuring how the **fraction of poisoned training views**
affects **selective target-object suppression** in a NeRF, while quantifying
**collateral damage** to the rest of the scene.

This repo is structured so the project can be picked up, audited, and reproduced
by someone who was not in the room when the decisions were made — including a
future version of you six weeks from now. Every experimental choice that
matters for the paper's validity is written down *before* it's needed, not
reconstructed from memory afterward.

## Start here

Read in this order:

1. **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — the locked protocol: research
   question, poisoning-budget definition, attack conditions, metrics, evaluation
   set, statistical plan. This is the closest thing this project has to a
   pre-registration. **Do not deviate from it once Phase 5 starts without
   logging the deviation in DECISION_LOG.md.**
2. **[docs/ROADMAP.md](docs/ROADMAP.md)** — the phased execution plan, with a
   data-driven go/no-go gate at the end of every phase.
3. **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** — what lives where
   in this repo and the rules for keeping it that way.
4. **[docs/DECISION_LOG.md](docs/DECISION_LOG.md)** — every non-trivial decision
   made so far (environment, dataset, budget definition, attack design), with
   the alternatives considered and why they were rejected. Append to this, don't
   edit past entries.
5. **[environment/SETUP.md](environment/SETUP.md)** — how to stand up the local
   (WSL2) and cloud (Colab/Kaggle) environments from zero.

## Current status

See the top of `docs/ROADMAP.md` for the live phase tracker (single
source of truth — this section is not).

**Resuming work in a new conversation?** Start with the most recent file
in [`docs/handoffs/`](docs/handoffs/) — it's the fastest way to get
oriented. A new handoff file is created at the end of every phase
closeout, following [`docs/handoffs/TEMPLATE.md`](docs/handoffs/TEMPLATE.md).
It is an index only, though: still read `README.md`, `docs/ROADMAP.md`,
`docs/METHODOLOGY.md`, and `docs/DECISION_LOG.md` in full before acting on
anything.

## Ground rules for this project

- **No result is real until it's reproducible from a config file + a git
  commit hash.** No manual hyperparameter tweaking that isn't captured in a
  YAML file under `configs/`.
- **No poisoned image is ever hand-edited.** Every poisoned image is produced
  by the scripted compositing pipeline in `src/poisoning/`, from a fixed
  formula, applied identically regardless of condition (see METHODOLOGY.md §3).
- **The held-out evaluation views are frozen before the first poisoning run**
  and never touched again. If this rule is ever broken, the experiment is
  invalid and must be rerun.
- **Every training run gets a log entry** using
  `docs/templates/EXPERIMENT_LOG_TEMPLATE.md` before its metrics are allowed
  into a results table.
- **Claims in the paper trace back to a specific experiment ID.** No number
  in the write-up without a corresponding row in `experiments/results/`.
