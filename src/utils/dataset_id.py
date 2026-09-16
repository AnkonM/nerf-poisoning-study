"""Resolve a (condition, seed) pair to its poisoned dataset, and PROVE it.

Phase 6 trains 24 runs from 20 datasets. Silently training the wrong data is the
single worst failure available in this phase: it produces a complete, plausible,
non-diverging run whose label is a lie, and nothing downstream can detect it.

That is not hypothetical here. Before D-032, every condition config resolved
`dataset.path` to `data/blender_scenes` -- the CLEAN scene -- inherited from
configs/scenes/final_scene.yaml, while the Phase 5 handoff recorded it as
"deliberately not set". Running `scripts/train.py configs/poisoning/budget_20.yaml`
would have trained the control, on seed 0, and labelled the result C3. No error,
no warning, no missing file.

So this module does two things:

1. `dataset_id_for` is the SINGLE definition of the (condition, seed) ->
   dataset_id rule. scripts/build_poison_set.py calls it when *writing* the
   datasets and the sweep harness calls it when *reading* them, so the two
   cannot drift apart into a mapping that is self-consistently wrong.

2. `resolve_dataset` verifies the resolved directory by its CONTENT, not its
   path. A path check only confirms a string; it cannot tell
   `data/poisoned/control` from `data/poisoned/C3_seed1` when both exist, both
   hold 100 valid PNGs, and both load cleanly. Counting how many training
   images actually differ from the clean originals can, and that count is
   pinned by METHODOLOGY.md §2's budget formula.
"""

import csv
import hashlib
import os
from typing import Any, Dict, Optional

from poisoning.view_selection import num_poisoned_for_budget

MANIFEST_FIELDS_REQUIRED = ("condition_id", "dataset_id", "selection_seed",
                            "view_index", "file_path", "poisoned")


def is_seed_invariant(budget_percent: float, view_selection: str) -> bool:
    """True when a condition's poisoned view set does not depend on the seed.

    Two cases, per DECISION_LOG.md D-028: a 0% budget poisons nothing, and
    strategic selection is a deterministic total order over the views (§7). Both
    therefore need ONE dataset shared by all three training runs, which differ
    only in model initialisation.
    """
    return (budget_percent == 0) or (view_selection == "strategic")


def dataset_id_for(condition_id: str, budget_percent: float,
                   view_selection: str, seed: int) -> str:
    """The (condition, seed) -> dataset_id rule. The only definition of it."""
    if is_seed_invariant(budget_percent, view_selection):
        return condition_id
    return "%s_seed%d" % (condition_id, seed)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_dataset(cfg: Dict[str, Any], seed: int, repo_root: str = ".",
                    manifest_path: Optional[str] = None,
                    clean_train_dir: Optional[str] = None) -> Dict[str, Any]:
    """Resolve and VERIFY this run's training dataset. Raises on any mismatch.

    Returns a dict recorded verbatim into the run's summary.json and its
    results.csv row, so the audit trail carries the evidence, not just the claim.
    """
    condition_id = cfg["condition_id"]
    pcfg = cfg["poisoning"]
    budget = pcfg["budget_percent"]
    method = pcfg["view_selection"]

    if manifest_path is None:
        manifest_path = os.path.join(repo_root, "data", "poisoned", "MANIFEST.csv")
    if clean_train_dir is None:
        clean_train_dir = os.path.join(repo_root, "data", "blender_scenes", "train")

    seed_invariant = is_seed_invariant(budget, method)
    dataset_id = dataset_id_for(condition_id, budget, method, seed)

    template = cfg["dataset"].get("path_template")
    if not template:
        raise RuntimeError(
            "%s: dataset.path_template is not set. Phase 6 resolves the training "
            "set per (condition, seed); a bare dataset.path would silently point "
            "every condition at one directory (see this module's docstring)."
            % condition_id)
    rel = template.format(dataset_id=dataset_id, condition_id=condition_id, seed=seed)
    path = os.path.join(repo_root, rel)

    def fail(msg):
        raise RuntimeError(
            "dataset assertion FAILED for condition=%s seed=%s -> dataset_id=%s\n"
            "  resolved path: %s\n  %s" % (condition_id, seed, dataset_id, path, msg))

    # --- 1. structure -----------------------------------------------------
    train_dir = os.path.join(path, "train")
    if not os.path.isdir(train_dir):
        fail("no train/ directory there")
    if not os.path.isfile(os.path.join(path, "transforms_train.json")):
        fail("no transforms_train.json there")

    # --- 2. manifest agreement -------------------------------------------
    if not os.path.isfile(manifest_path):
        fail("manifest not found at %s" % manifest_path)
    with open(manifest_path, newline="") as f:
        rdr = csv.DictReader(f)
        missing = [c for c in MANIFEST_FIELDS_REQUIRED if c not in (rdr.fieldnames or [])]
        if missing:
            fail("manifest %s lacks required columns %s" % (manifest_path, missing))
        rows = [r for r in rdr if r["dataset_id"] == dataset_id]
    if not rows:
        fail("manifest has no rows for dataset_id=%r" % dataset_id)
    if len(rows) != 100:
        fail("manifest has %d rows for %s, expected 100" % (len(rows), dataset_id))
    bad = {r["condition_id"] for r in rows} - {condition_id}
    if bad:
        fail("manifest rows for %s claim condition(s) %s" % (dataset_id, sorted(bad)))
    expected_sel = "" if seed_invariant else str(seed)
    got_sel = {r["selection_seed"] for r in rows}
    if got_sel != {expected_sel}:
        fail("manifest selection_seed is %s, expected %r (seed_invariant=%s)"
             % (sorted(got_sel), expected_sel, seed_invariant))

    # --- 3. files on disk match the manifest -----------------------------
    manifest_names = sorted(os.path.basename(r["file_path"]) for r in rows)
    on_disk = sorted(n for n in os.listdir(train_dir) if n.endswith(".png"))
    if manifest_names != on_disk:
        only_m = sorted(set(manifest_names) - set(on_disk))[:5]
        only_d = sorted(set(on_disk) - set(manifest_names))[:5]
        fail("train/ contents disagree with the manifest (in manifest only: %s; "
             "on disk only: %s)" % (only_m, only_d))

    # --- 4. the config's seed is this run's seed --------------------------
    cfg_seed = cfg["reproducibility"]["seed"]
    if cfg_seed != seed:
        fail("config reproducibility.seed=%r but this run is seed=%r -- the "
             "harness must set them together (METHODOLOGY.md §8)" % (cfg_seed, seed))

    # --- 5. CONTENT: how many images actually differ from the clean scene? -
    # This is the check a path comparison cannot make. It distinguishes the
    # control from C3 from a half-built directory, using only the data itself.
    expected_poisoned = num_poisoned_for_budget(budget, len(rows))
    manifest_poisoned = sum(1 for r in rows if r["poisoned"] == "True")
    if manifest_poisoned != expected_poisoned:
        fail("manifest marks %d views poisoned, but round(%s/100 * %d) = %d"
             % (manifest_poisoned, budget, len(rows), expected_poisoned))

    differing = []
    for name in on_disk:
        clean = os.path.join(clean_train_dir, name)
        if not os.path.isfile(clean):
            fail("clean original missing for %s (looked in %s)" % (name, clean_train_dir))
        if _sha256(os.path.join(train_dir, name)) != _sha256(clean):
            differing.append(name)
    if len(differing) != expected_poisoned:
        fail("%d of %d training images differ from the clean originals, but this "
             "condition (budget=%s%%) must have exactly %d poisoned views.\n"
             "  This is the signature of a WRONG DATASET: %s."
             % (len(differing), len(on_disk), budget, expected_poisoned,
                "a clean/control directory resolved for a poisoned condition"
                if len(differing) == 0 and expected_poisoned > 0
                else "resolved directory does not match this condition"))

    manifest_flagged = sorted(os.path.basename(r["file_path"])
                              for r in rows if r["poisoned"] == "True")
    if manifest_flagged != sorted(differing):
        fail("the images that differ from clean are not the ones the manifest "
             "marks poisoned (differ-but-not-flagged: %s; flagged-but-identical: %s)"
             % (sorted(set(differing) - set(manifest_flagged))[:5],
                sorted(set(manifest_flagged) - set(differing))[:5]))

    # --- 6. digest of the resolved train set, for the audit trail ---------
    agg = hashlib.sha256()
    for name in on_disk:
        agg.update(name.encode())
        agg.update(_sha256(os.path.join(train_dir, name)).encode())

    return {
        "dataset_id": dataset_id,
        "dataset_path": rel,
        "seed_invariant": seed_invariant,
        "n_train_views": len(on_disk),
        "n_poisoned_verified": len(differing),
        "n_poisoned_expected": expected_poisoned,
        "train_set_sha256": agg.hexdigest(),
    }
