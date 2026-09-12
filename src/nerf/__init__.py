"""Vanilla NeRF model/training code — vendored + adapted.

Vendored from yenchenlin/nerf-pytorch (MIT license — see
THIRD_PARTY_LICENSE in this directory), commit 63a5a630c9abd62b0f21c08703
d0ac2ea7d4b9dd, chosen per docs/DECISION_LOG.md D-005 (pure PyTorch, no
custom CUDA extensions) and its own recommendation writeup. Adapted from
the original's configargparse/CLI-driven structure to this project's
config-driven one (docs/PROJECT_STRUCTURE.md): hyperparameters are
resolved from configs/*.yaml, not CLI flags, and the original's single
run_nerf.py script is split into model.py / rays.py / rendering.py /
training.py / datasets/blender.py. The core numerical routines (ray
generation, hierarchical sampling, volumetric rendering) are kept
faithful to the original — this is a restructuring for this project's
conventions, not a reimplementation of the underlying math.
"""


def main() -> None:
    """Placeholder entry point for the `nerf` console script.

    Not implemented — the real CLI surface for this project is the
    scripts/ entry points (scripts/train.py, scripts/evaluate.py, ...)
    per docs/PROJECT_STRUCTURE.md.
    """
    raise NotImplementedError(
        "nerf:main is a scaffold placeholder. Use scripts/train.py "
        "<config_path> instead (see docs/PROJECT_STRUCTURE.md)."
    )
