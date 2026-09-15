"""View-selection logic for poisoning, per docs/METHODOLOGY.md §2 and §7.

Two methods, both operating on integer view indices into the training set:

* `sample_random` — uniform without replacement, seeded (§7). Used by C1-C6.
  **Not modified since Phase 3**: its exact behaviour determines which views
  Phase 3's PoC poisoned, so it is left byte-identical on purpose.
* `sample_strategic` — "largest target mask area first" (§7), the C7 ablation.
"""

from typing import Mapping, Sequence

import numpy as np


def num_poisoned_for_budget(budget_percent: float, v_target_size: int) -> int:
    """round(budget / 100 * |V_target|), per docs/METHODOLOGY.md §2.

    Uses Python's builtin `round`, which is half-to-even. At this study's
    |V_target| = 100 every budget in §2's ladder {0,5,10,20,30,50} yields an
    exactly-representable product (0.0/5.0/.../50.0), so no budget ever lands on
    a .5 boundary and the rounding mode is never exercised — verified, see
    DECISION_LOG.md D-029. The mode still matters for any other |V_target|
    (e.g. |V_target|=10 at 5% gives 0.5 -> 0, i.e. ZERO poisoned views), which is
    exactly what ROADMAP.md's Phase 4 gate ("5% must round to >= 1") guards
    against. The behaviour is pinned by tests rather than changed: altering the
    rounding rule would be a METHODOLOGY.md §2 deviation.
    """
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


def sample_strategic(v_target: Sequence[int], budget_percent: float,
                     areas: Mapping[int, float]):
    """Strategic selection (C7): largest target mask area first, per §7.

    Deterministic and unseeded — the rule is a total order over the views, so
    there is no randomness to seed.

    Tie-break is **`(-area, view_index)`**: descending mask area, then ascending
    view index. This matters and is not decorative. Measured on the real frozen
    scene (DECISION_LOG.md D-028): exact area ties DO occur — views 57 and 73
    share 12,554 px at ranks 35/36 — while C7's own rank-20 boundary is untied by
    only 3 pixels. Without an explicit tie-break the result would depend on the
    input sequence's order, which is latent nondeterminism in a project whose
    central claim is reproducibility from config + commit.

    `areas` maps view index -> target mask area fraction, and comes from
    `data/blender_scenes/cameras.json`'s recorded `v_target.per_view_area_fraction`.
    It is never recomputed here: METHODOLOGY.md §2 requires those values be
    recorded once and not recomputed mid-study.

    Returns (sorted list of selected view indices, num_poisoned).
    """
    v_target = list(v_target)
    missing = [v for v in v_target if v not in areas]
    if missing:
        raise KeyError(
            "no recorded mask area for view(s) %s — areas must come from "
            "cameras.json, not be recomputed (METHODOLOGY.md §2)" % missing[:5])

    num_poisoned = num_poisoned_for_budget(budget_percent, len(v_target))
    ranked = sorted(v_target, key=lambda v: (-areas[v], v))
    return sorted(int(v) for v in ranked[:num_poisoned]), num_poisoned
