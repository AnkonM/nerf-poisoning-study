"""Budget arithmetic and view-selection determinism (METHODOLOGY.md §2, §7)."""

import pytest

from poisoning.view_selection import (num_poisoned_for_budget, sample_random,
                                      sample_strategic)

V_TARGET = 100          # this study's frozen |V_target| (D-025)


# Values below were COMPUTED from the real function against the real
# |V_target| = 100, not assumed (D-029). All six products are exactly
# representable floats, so no budget lands on a .5 boundary here.
@pytest.mark.parametrize("budget,expected", [
    (0, 0), (5, 5), (10, 10), (20, 20), (30, 30), (50, 50),
])
def test_manifest_counts_exact_at_real_v_target(budget, expected):
    assert num_poisoned_for_budget(budget, V_TARGET) == expected


@pytest.mark.parametrize("budget", [0, 5, 10, 20, 30, 50])
def test_no_budget_lands_on_a_half_boundary_at_v_target_100(budget):
    """Documents WHY the rounding mode never matters for this study's own data."""
    assert (budget / 100.0 * V_TARGET) % 1 != 0.5


# Python's builtin round() is half-to-even. These are the REAL outputs at other
# |V_target| values, pinned so the behaviour cannot change unnoticed. Changing
# the rounding rule would be a METHODOLOGY.md §2 deviation, so this test records
# the behaviour rather than asserting a "nicer" one.
@pytest.mark.parametrize("v_target,budget,actual,naive_half_up", [
    (90, 5, 4, 5),
    (50, 5, 2, 3),
    (10, 5, 0, 1),     # <- would give ZERO poisoned views: ROADMAP's Phase 4
])                     #    gate ("5% must round to >= 1") exists for this case
def test_half_boundary_behaviour_is_half_to_even(v_target, budget, actual, naive_half_up):
    assert (budget / 100.0 * v_target) % 1 == 0.5, "not a .5 case; test is wrong"
    assert num_poisoned_for_budget(budget, v_target) == actual
    assert actual != naive_half_up


def test_strategic_tie_break_is_ascending_view_index():
    """Equal areas must resolve deterministically by index, not by input order."""
    areas = {0: 0.9, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.1}   # 1,2,3 all tied
    sel, n = sample_strategic([4, 2, 0, 3, 1], budget_percent=60, areas=areas)
    assert n == 3
    assert sel == [0, 1, 2]        # top area, then the two lowest tied indices


def test_strategic_is_order_independent():
    import random
    areas = {i: (i % 7) / 10.0 for i in range(40)}     # many ties by construction
    ref, _ = sample_strategic(list(range(40)), 25, areas)
    for _ in range(50):
        shuffled = random.sample(range(40), 40)
        assert sample_strategic(shuffled, 25, areas)[0] == ref


def test_strategic_rejects_views_with_no_recorded_area():
    """Areas must come from cameras.json, never be silently recomputed (§2)."""
    with pytest.raises(KeyError):
        sample_strategic([0, 1, 2], 50, {0: 0.1, 1: 0.2})


def test_random_selection_is_seeded_and_reproducible():
    a, n = sample_random(list(range(V_TARGET)), 20, seed=1)
    b, _ = sample_random(list(range(V_TARGET)), 20, seed=1)
    c, _ = sample_random(list(range(V_TARGET)), 20, seed=2)
    assert a == b and n == 20
    assert a != c, "different seeds must select different views"
