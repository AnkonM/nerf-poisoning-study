"""(condition, seed) -> dataset resolution and its content assertion (D-032).

These guard the failure this project has the least defence against elsewhere:
training a run on the wrong dataset. That produces a complete, non-diverging,
entirely plausible run whose label is false, and no downstream check catches it.

The fixtures below are synthetic and self-contained — they do not read the real
frozen scene or the real poisoned datasets, so the suite stays fast and cannot
be broken by data changes.
"""

import csv
import os

import imageio.v2 as imageio
import numpy as np
import pytest

from utils.dataset_id import dataset_id_for, is_seed_invariant, resolve_dataset

MANIFEST_FIELDS = [
    "condition_id", "dataset_id", "selection_seed", "view_index", "file_path",
    "poisoned", "attack_type", "budget_percent", "view_selection", "alpha",
    "mask_source", "mask_dilation_px",
]


# --------------------------------------------------------------------------
# the id rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("budget,method,expected", [
    (0, "random", True),        # nothing is poisoned, so the seed cannot matter
    (0, "strategic", True),
    (20, "strategic", True),    # deterministic total order over views (§7)
    (5, "random", False),
    (20, "random", False),
    (50, "random", False),
])
def test_seed_invariance_rule(budget, method, expected):
    assert is_seed_invariant(budget, method) is expected


def test_seed_invariant_conditions_share_one_dataset_across_seeds():
    """control and C7 have ONE dataset but THREE runs — only model init differs."""
    assert {dataset_id_for("control", 0, "random", s) for s in (1, 2, 3)} == {"control"}
    assert {dataset_id_for("C7", 20, "strategic", s) for s in (1, 2, 3)} == {"C7"}


def test_seeded_conditions_get_one_dataset_per_seed():
    assert [dataset_id_for("C3", 20, "random", s) for s in (1, 2, 3)] == \
        ["C3_seed1", "C3_seed2", "C3_seed3"]


# --------------------------------------------------------------------------
# fixtures: a miniature scene + poisoned datasets on disk
# --------------------------------------------------------------------------

N_VIEWS = 100


def _write(path, value):
    # imageio, not PIL: imageio is a declared direct dependency of this project
    # (pyproject.toml) and is what the real pipeline reads and writes with.
    imageio.imwrite(str(path), np.full((4, 4, 3), value, np.uint8))


@pytest.fixture
def repo(tmp_path):
    """A minimal repo root: a clean train split plus two poisoned datasets."""
    clean = tmp_path / "data" / "blender_scenes" / "train"
    clean.mkdir(parents=True)
    for i in range(N_VIEWS):
        _write(clean / f"r_{i:03d}.png", 10)

    rows = []

    def make(dataset_id, condition_id, sel_seed, budget, method, poisoned_idx):
        d = tmp_path / "data" / "poisoned" / dataset_id / "train"
        d.mkdir(parents=True)
        (d.parent / "transforms_train.json").write_text("{}")
        for i in range(N_VIEWS):
            _write(d / f"r_{i:03d}.png", 200 if i in poisoned_idx else 10)
            rows.append({
                "condition_id": condition_id, "dataset_id": dataset_id,
                "selection_seed": sel_seed, "view_index": i,
                "file_path": f"data/poisoned/{dataset_id}/train/r_{i:03d}.png",
                "poisoned": str(i in poisoned_idx),
                "budget_percent": budget, "view_selection": method,
            })

    make("control", "control", "", 0, "random", set())
    make("C3_seed1", "C3", "1", 20, "random", set(range(20)))
    # same budget, DIFFERENT views — the wrong-seed case
    make("C3_seed2", "C3", "2", 20, "random", set(range(20, 40)))

    mpath = tmp_path / "data" / "poisoned" / "MANIFEST.csv"
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    return tmp_path


def _cfg(condition_id, budget, method, seed, template="data/poisoned/{dataset_id}"):
    return {
        "condition_id": condition_id,
        "poisoning": {"budget_percent": budget, "view_selection": method},
        "dataset": {"path": None, "path_template": template},
        "reproducibility": {"seed": seed},
    }


# --------------------------------------------------------------------------
# the assertion
# --------------------------------------------------------------------------

def test_correct_dataset_resolves_and_verifies(repo):
    info = resolve_dataset(_cfg("C3", 20, "random", 1), 1, repo_root=str(repo))
    assert info["dataset_id"] == "C3_seed1"
    assert info["n_poisoned_verified"] == 20 == info["n_poisoned_expected"]
    assert info["seed_invariant"] is False
    assert len(info["train_set_sha256"]) == 64


def test_control_resolves_with_zero_poisoned(repo):
    info = resolve_dataset(_cfg("control", 0, "random", 2), 2, repo_root=str(repo))
    assert info["dataset_id"] == "control"
    assert info["n_poisoned_verified"] == 0
    assert info["seed_invariant"] is True


def test_poisoned_condition_pointed_at_the_control_directory_raises(repo):
    """The headline case: C3 resolving to clean data must NOT train silently."""
    cfg = _cfg("C3", 20, "random", 1, template="data/poisoned/control")
    with pytest.raises(RuntimeError, match="WRONG DATASET"):
        resolve_dataset(cfg, 1, repo_root=str(repo))


def test_wrong_seeds_directory_raises_even_though_the_count_matches(repo):
    """C3_seed2 has 20 poisoned views too — only the identities differ.

    A count-only or path-only check passes this. That is why the assertion
    compares which files differ, not just how many.
    """
    cfg = _cfg("C3", 20, "random", 1, template="data/poisoned/C3_seed2")
    with pytest.raises(RuntimeError, match="not the ones the manifest marks"):
        resolve_dataset(cfg, 1, repo_root=str(repo))


def test_config_seed_must_equal_run_seed(repo):
    with pytest.raises(RuntimeError, match="reproducibility.seed"):
        resolve_dataset(_cfg("C3", 20, "random", 9), 1, repo_root=str(repo))


def test_missing_path_template_raises(repo):
    cfg = _cfg("C3", 20, "random", 1)
    del cfg["dataset"]["path_template"]
    with pytest.raises(RuntimeError, match="path_template"):
        resolve_dataset(cfg, 1, repo_root=str(repo))


def test_unknown_dataset_raises(repo):
    with pytest.raises(RuntimeError, match="no train/ directory"):
        resolve_dataset(_cfg("C9", 20, "random", 1), 1, repo_root=str(repo))


def test_tampered_image_is_detected(repo):
    """An extra modified file breaks the count even if the manifest is intact."""
    _write(repo / "data" / "poisoned" / "C3_seed1" / "train" / "r_050.png", 99)
    with pytest.raises(RuntimeError, match="21 of 100"):
        resolve_dataset(_cfg("C3", 20, "random", 1), 1, repo_root=str(repo))
