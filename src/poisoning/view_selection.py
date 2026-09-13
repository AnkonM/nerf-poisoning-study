"""View-selection logic for poisoning, per docs/METHODOLOGY.md §2 and §7.

Only random selection is implemented so far — the strategic method (C7
ablation, `docs/ROADMAP.md` Phase 7) is out of scope for the Phase 3 PoC
this module currently supports and will be added when that phase starts.
"""

from typing import Sequence

import numpy as np


def num_poisoned_for_budget(budget_percent: float, v_target_size: int) -> int:
    """round(budget / 100 * |V_target|), per docs/METHODOLOGY.md §2."""
    return round(budget_percent / 100.0 * v_target_size)


def sample_random(v_target: Sequence[int], budget_percent: float, seed: int):
    """Uniform sample without replacement from `v_target`, seeded.

    Returns (sorted list of selected view indices, num_poisoned).
    """
    v_target = list(v_target)
    num_poisoned = num_poisoned_for_budget(budget_percent, len(v_target))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(v_target), size=num_poisoned, replace=False)
    return sorted(int(i) for i in chosen), num_poisoned
