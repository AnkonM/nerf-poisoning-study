"""Seeding helper — every run's seed must come from its config, not be left
implicit, per configs/base.yaml's `reproducibility.seed` field and
docs/METHODOLOGY.md §8.
"""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
