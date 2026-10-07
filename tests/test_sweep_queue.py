"""Queue order/size and the (condition, seed) -> dataset mapping (D-040)."""

import csv
import os

import pytest

from utils.sweep_queue import EXPECTED_CONDITIONS, build_queue

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = os.path.join(REPO, "configs", "poisoning")
REAL_MANIFEST = os.path.join(REPO, "data", "poisoned", "MANIFEST.csv")


def test_queue_is_24_runs_seed_major():
    q = build_queue(CONFIGS)
    assert len(q) == 24
    assert [r.seed for r in q] == [1] * 8 + [2] * 8 + [3] * 8
    for s in range(3):
        assert [r.condition_id for r in q[s * 8:(s + 1) * 8]] == EXPECTED_CONDITIONS
    assert len({r.run_id for r in q}) == 24


def test_run_ids_follow_train_py_convention():
    q = build_queue(CONFIGS)
    assert q[0].run_id == "control_seed1" and q[8].run_id == "control_seed2" and q[23].run_id == "C7_seed3"


def test_dataset_mapping_including_seed_invariant_conditions():
    q = {r.run_id: r for r in build_queue(CONFIGS)}
    for s in (1, 2, 3):
        assert q["control_seed%d" % s].dataset_id == "control"      # one dataset, three training seeds
        assert q["C7_seed%d" % s].dataset_id == "C7"
        for c in ("C1", "C2", "C3", "C4", "C5", "C6"):
            assert q["%s_seed%d" % (c, s)].dataset_id == "%s_seed%d" % (c, s)
    assert {r.n_poisoned_expected for r in q.values() if r.condition_id == "C3"} == {20}
    assert [q["%s_seed1" % c].n_poisoned_expected for c in ("control", "C1", "C2", "C3", "C4", "C5", "C6", "C7")] \
        == [0, 5, 10, 20, 30, 50, 20, 20]


@pytest.mark.skipif(not os.path.exists(REAL_MANIFEST), reason="real MANIFEST.csv not restored")
def test_mapping_agrees_with_the_real_manifest():
    assert len(build_queue(CONFIGS, REAL_MANIFEST)) == 24


def test_manifest_disagreement_is_an_error(tmp_path):
    m = tmp_path / "MANIFEST.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition_id", "dataset_id", "selection_seed"])
        w.writeheader()
        for cid, did, sel in [("control", "control", ""), ("C7", "C7", "")] + \
                [(c, "%s_seed%d" % (c, s), str(s if not (c == "C3" and s == 2) else 9))
                 for c in ("C1", "C2", "C3", "C4", "C5", "C6") for s in (1, 2, 3)]:
            w.writerow({"condition_id": cid, "dataset_id": did, "selection_seed": sel})
    with pytest.raises(RuntimeError, match="disagrees"):
        build_queue(CONFIGS, str(m))


def test_missing_condition_is_an_error(tmp_path):
    import shutil
    shutil.copytree(os.path.join(REPO, "configs"), tmp_path / "configs")
    os.remove(tmp_path / "configs" / "poisoning" / "budget_50.yaml")      # C5
    with pytest.raises(RuntimeError, match="is missing"):
        build_queue(str(tmp_path / "configs" / "poisoning"))
