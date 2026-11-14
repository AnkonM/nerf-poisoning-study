"""Phase 9 laptop queue (DECISION_LOG.md D-045 / D-046): an opt-in, additive layer over the Phase 6 sweep.

Nothing here changes the Phase 6 harness: the queue, the hardware pin, results.csv and its schema are untouched. The
Phase 9 queue has its own run ids (prefix), its own lock/STOPPED/pin files, results_phase9.csv (same row schema plus
`hardware_profile`) and a hardware PROFILE that fails closed when the machine differs.
"""

import csv
import os
import socket
from dataclasses import dataclass
from typing import Dict, List, Tuple

from utils import sweep_checks as chk
from utils import sweep_state as st
from utils.config import load_config
from utils.dataset_id import dataset_id_for
from utils.sweep_queue import Run
from utils.sweep_runner import Sweep
from poisoning.view_selection import num_poisoned_for_budget

PHASE9_RESULTS_FIELDS = st.RESULTS_FIELDS + ["hardware_profile"]


@dataclass(frozen=True)
class P9Run(Run):
    prefix: str = "lp9_"
    role: str = ""

    @property
    def run_id(self) -> str:
        return "%s%s_seed%d" % (self.prefix, self.condition_id, self.seed)


@dataclass(frozen=True)
class Phase9Paths(st.Paths):
    """Same layout as the Phase 6 paths, but with its own results file and its own lock / STOPPED / pin files
    (so a Phase 9 queue and the Phase 6 sweep can never block, pin or overwrite each other)."""
    results_name: str = "results_phase9.csv"

    @property
    def results_csv(self): return os.path.join(self.root, "results", self.results_name)
    @property
    def lockfile(self): return os.path.join(self.runs, ".phase9.lock")
    @property
    def stopped(self): return os.path.join(self.runs, "PHASE9_STOPPED")
    @property
    def stop_after_current(self): return os.path.join(self.runs, "PHASE9_STOP_AFTER_CURRENT")
    @property
    def hw_pin(self): return os.path.join(self.runs, "phase9_hardware.json")
    @property
    def harness_log(self): return os.path.join(self.logs, "phase9_harness.log")


def load_profile(path: str) -> Dict:
    prof = load_config(path)
    for k in ("profile", "hardware", "preflight", "paths", "queue"):
        if k not in prof:
            raise RuntimeError("%s: missing section %r" % (path, k))
    return prof


def check_profile(profile: Dict, gpu: Dict, hostname: str, torch_version: str) -> Tuple[bool, str]:
    """Fail closed: every field of profile['hardware'] must equal what is running now."""
    want = profile["hardware"]
    have = {"gpu_name": gpu.get("name"), "driver": gpu.get("driver"), "torch": torch_version, "hostname": hostname}
    diff = {k: (want[k], have.get(k)) for k in want if str(want[k]) != str(have.get(k))}
    if diff:
        return False, "machine differs from hardware profile %s (expected, found): %s" % (profile["profile"], diff)
    return True, "matches hardware profile %s (%s, driver %s, torch %s, host %s)" % (
        profile["profile"], have["gpu_name"], have["driver"], have["torch"], hostname)


def build_phase9_queue(profile: Dict, repo_root: str, manifest_path: str) -> List[P9Run]:
    q, prefix, cdir = [], profile["paths"]["run_id_prefix"], os.path.join(repo_root, profile["queue"]["config_dir"])
    manifest = {}
    with open(manifest_path, newline="") as f:
        for r in csv.DictReader(f):
            manifest.setdefault(r["dataset_id"], set()).add((r["condition_id"], r["selection_seed"]))
    entries = [e for e in profile["queue"]["entries"]] + [e for e in profile["queue"].get("optional", []) if e.get("enabled")]
    for e in entries:
        path = os.path.join(cdir, e["config"])
        cfg = load_config(path)
        if cfg["condition_id"] != e["condition_id"]:
            raise RuntimeError("%s declares condition_id %r, expected %r" % (path, cfg["condition_id"], e["condition_id"]))
        pc = cfg["poisoning"]
        for s in e["seeds"]:
            did = dataset_id_for(cfg["condition_id"], pc["budget_percent"], pc["view_selection"], s)
            invariant = did == cfg["condition_id"]
            want = {(cfg["condition_id"], "" if invariant else str(s))}
            if manifest.get(did) != want:
                raise RuntimeError("MANIFEST.csv disagrees with the dataset rule for %s seed %d (dataset %r: %s, expected %s)"
                                   % (cfg["condition_id"], s, did, sorted(manifest.get(did, [])), sorted(want)))
            q.append(P9Run(cfg["condition_id"], s, path, pc["budget_percent"], pc["view_selection"], pc["attack_type"],
                           did, num_poisoned_for_budget(pc["budget_percent"], 100), prefix, e["role"]))
    ids = [r.run_id for r in q]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate run ids in the Phase 9 queue: %s" % ids)
    return q


def append_results_row_p9(csv_path: str, row: Dict) -> bool:
    """Append-only, idempotent, schema-guarded appender for results_phase9.csv."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new = True
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="") as f:
            if next(csv.reader(f)) != PHASE9_RESULTS_FIELDS:
                raise RuntimeError("%s has a different header; refusing to append" % csv_path)
        if row["run_id"] in st.read_results(csv_path):
            return False
        new = False
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PHASE9_RESULTS_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in PHASE9_RESULTS_FIELDS})
        f.flush()
        os.fsync(f.fileno())
    return True


class Phase9Sweep(Sweep):
    """Sweep with the Phase 9 queue, the hardware profile gate and results_phase9.csv. Everything else (retry / stop /
    resume policy, flock, preflight, post-run validation) is the Phase 6 code."""

    def __init__(self, cfg, profile: Dict, log=print):
        super().__init__(cfg, log)
        self.profile = profile

    def queue(self):
        q = build_phase9_queue(self.profile, self.cfg.repo_root, self.cfg.manifest_path)
        return [r for r in q if r.run_id in set(self.cfg.only)] if self.cfg.only else q

    def _hardware_checks(self, snap, create_pin, rec):
        try:
            import importlib.metadata as md
            rec("hardware profile", *check_profile(self.profile, snap, socket.gethostname(), md.version("torch")))
            ident = chk.hardware_identity(snap)
            if create_pin or os.path.exists(self.cfg.paths.hw_pin):
                rec("hardware pin", *chk.check_hardware_pin(self.cfg.paths.hw_pin, ident))   # also pins the GPU uuid
        except Exception as e:  # noqa: BLE001
            rec("hardware profile", False, "could not determine hardware identity: %s" % e)

    def _write_record(self, run, iterations, rd, ev, pf):
        from utils.sweep_record import write_run_record
        prof = self.profile["profile"]
        write_run_record(
            self.cfg, run, iterations, rd, ev, pf,
            where_run="local — WSL2 laptop, hardware profile `%s` (Phase 9 extension, D-045/D-046)" % prof,
            note=("**Phase 9 extension run (%s, role: %s).** Hardware profile `%s`. Comparisons with Phase 6 runs cross GPU "
                  "architectures (Quadro RTX 5000 Turing -> RTX 5060 Blackwell). Not part of the METHODOLOGY §5 matrix."
                  % (run.condition_id, getattr(run, "role", ""), prof)),
            row_extra={"hardware_profile": prof}, append_fn=append_results_row_p9)


def phase9_paths(experiments_root: str, profile: Dict) -> Phase9Paths:
    return Phase9Paths(os.path.abspath(experiments_root), results_name=os.path.basename(profile["paths"]["results_csv"]))
