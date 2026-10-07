"""The Phase 6 run queue: 8 conditions x 3 seeds = 24 runs, SEED-MAJOR (D-040).

Seed-major (all 8 conditions at seed 1, then seed 2, then seed 3) means that an
interrupted or truncated sweep still has every condition at the lower N, which
is the "What to Cut" fallback in ROADMAP.md (N=3 -> N=2) rather than a few
conditions at N=3 and the rest at N=0. It also surfaces a condition-specific
bug in the first 8 runs instead of at run 16+ (D-033).

Run IDs and the (condition, seed) -> dataset rule are NOT defined here:
`run_id` follows scripts/train.py (`<condition_id>_seed<N>`), and the dataset id
comes from utils.dataset_id.dataset_id_for -- the one definition shared with
scripts/build_poison_set.py (D-032).
"""

import csv
import os
from dataclasses import dataclass
from typing import List, Optional

from utils.config import load_config
from utils.dataset_id import dataset_id_for
from poisoning.view_selection import num_poisoned_for_budget

EXPECTED_CONDITIONS = ["control", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]


@dataclass(frozen=True)
class Run:
    condition_id: str
    seed: int
    config_path: str
    budget_percent: float
    view_selection: str
    attack_type: str
    dataset_id: str
    n_poisoned_expected: int

    @property
    def run_id(self) -> str:  # same convention as scripts/train.py
        return "%s_seed%d" % (self.condition_id, self.seed)

    @property
    def dataset_rel(self) -> str:
        return os.path.join("data", "poisoned", self.dataset_id)


def _cond_sort_key(cid: str):
    return (-1, 0) if cid == "control" else (0, int(cid[1:]))


# The eight condition configs documented in PROJECT_STRUCTURE.md. Listed explicitly
# rather than discovered: configs/poisoning/ also holds Phase 3 PoC configs and
# probe_control_convergence.yaml, which inherit budget_00.yaml and therefore ALSO
# declare condition_id "control" -- auto-discovery would see two controls.
CONDITION_CONFIG_FILES = {
    "control": "budget_00.yaml", "C1": "budget_05.yaml", "C2": "budget_10.yaml",
    "C3": "budget_20.yaml", "C4": "budget_30.yaml", "C5": "budget_50.yaml",
    "C6": "soft_suppression_20.yaml", "C7": "ablation_random_vs_strategic.yaml",
}


def discover_condition_configs(config_dir: str, expected_conditions=EXPECTED_CONDITIONS) -> List[str]:
    """Paths of the condition configs in `expected_conditions` order, each verified to
    declare the condition it is listed for."""
    paths = []
    for cid in sorted(expected_conditions, key=_cond_sort_key):
        p = os.path.join(config_dir, CONDITION_CONFIG_FILES[cid])
        if not os.path.isfile(p):
            raise RuntimeError("condition %s: expected config %s is missing; expected conditions are %s"
                               % (cid, p, list(expected_conditions)))
        got = load_config(p).get("condition_id")
        if got != cid:
            raise RuntimeError("%s declares condition_id %r, expected %r" % (p, got, cid))
        paths.append(p)
    return paths


def build_queue(config_dir: str, manifest_path: Optional[str] = None,
                expected_conditions=EXPECTED_CONDITIONS) -> List[Run]:
    """All runs, seed-major. Raises unless exactly the expected conditions exist.

    If `manifest_path` is given, each run's dataset_id is cross-checked against
    MANIFEST.csv (condition_id + selection_seed), not just computed.
    """
    paths = discover_condition_configs(config_dir, expected_conditions)
    cfgs = [(p, load_config(p)) for p in paths]
    seed_sets = {tuple(c["seeds"]) for _, c in cfgs}
    if len(seed_sets) != 1:
        raise RuntimeError("condition configs declare different seed lists: %s" % sorted(seed_sets))
    seeds = list(seed_sets.pop())

    manifest = None
    if manifest_path is not None:
        with open(manifest_path, newline="") as f:
            manifest = {}
            for r in csv.DictReader(f):
                manifest.setdefault(r["dataset_id"], set()).add((r["condition_id"], r["selection_seed"]))

    queue = []
    for seed in seeds:
        for p, c in cfgs:
            pc = c["poisoning"]
            did = dataset_id_for(c["condition_id"], pc["budget_percent"], pc["view_selection"], seed)
            if manifest is not None:
                got = manifest.get(did)
                invariant = did == c["condition_id"]
                want = {(c["condition_id"], "" if invariant else str(seed))}
                if got != want:
                    raise RuntimeError("MANIFEST.csv disagrees with the dataset rule for %s seed %d: "
                                       "dataset_id %r has (condition, selection_seed) = %s, expected %s"
                                       % (c["condition_id"], seed, did, sorted(got or []), sorted(want)))
            queue.append(Run(c["condition_id"], seed, p, pc["budget_percent"], pc["view_selection"],
                             pc["attack_type"], did,
                             num_poisoned_for_budget(pc["budget_percent"], 100)))
    return queue
