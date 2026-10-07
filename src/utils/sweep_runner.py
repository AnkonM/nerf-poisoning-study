"""Sweep execution: preflight -> train -> evaluate -> record, one run at a time.

Design rules (D-040):
  * One training process at a time, enforced by flock (SweepLock) AND by the GPU
    preflight; the child dies with the harness (PR_SET_PDEATHSIG), so a kill -9
    of the harness cannot leave an orphan training process running.
  * train.py runs with --skip-final-eval; scripts/evaluate.py then runs as a
    SEPARATE process after training exits (D-016/D-019; eval peaks at ~8.4 GiB).
  * Completion is judged by real exit codes plus the on-disk artifacts.
  * Crash / CUDA error / OOM / SIGKILL -> retry with --resume, max `max_retries`.
    Non-finite loss, windowed divergence, or ANY validation failure -> FAILED,
    no retry, STOP the queue: a non-converged run is a bug, not a data point.
"""

import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from utils import sweep_checks as chk
from utils import sweep_state as st
from utils.config import load_config
from utils.sweep_queue import Run, build_queue

EXIT_OK, EXIT_CRASH, EXIT_STOPPED, EXIT_LOCKED = 0, 1, 10, 11   # systemd: RestartPreventExitStatus=10 11


class StopQueue(Exception):
    def __init__(self, reason: str, run_id: Optional[str] = None, fail_run: bool = False):
        super().__init__(reason)
        self.reason, self.run_id, self.fail_run = reason, run_id, fail_run


class Interrupted(Exception):
    pass


@dataclass
class SweepConfig:
    repo_root: str
    paths: st.Paths
    config_dir: str
    python: str = sys.executable
    manifest_path: Optional[str] = None
    only: Optional[List[str]] = None
    allow_dirty: bool = False
    dry_run: bool = False
    max_retries: int = 2
    poll_s: float = 15.0
    min_free_gb: float = 20.0
    gpu_mem_max_mib: float = 2048.0   # desktop baseline 588 MiB (D-038) + ~1.4 GiB margin
    gpu_wait_s: float = 30.0
    window: int = 20
    ratio: float = 2.0
    extra_train_args: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.manifest_path is None:
            self.manifest_path = os.path.join(self.repo_root, "data", "poisoned", "MANIFEST.csv")


def _pdeathsig():
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _fsync_dir(p):
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Sweep:
    popen: Callable = staticmethod(subprocess.Popen)     # tests replace these
    preflight_fn: Optional[Callable] = None

    def __init__(self, cfg: SweepConfig, log: Callable[[str], None] = print):
        self.cfg, self.say = cfg, log
        self.terminate = threading.Event()

    # ------------------------------------------------------------ preflight
    def preflight(self, run: Run, create_pin: bool = True) -> Dict:
        if self.preflight_fn is not None:
            return self.preflight_fn(run)
        c = self.cfg
        results, warnings = [], []

        def rec(name, ok, detail):
            results.append((name, ok, detail))

        ok, d, commit = chk.check_git_clean(c.repo_root)
        if c.allow_dirty and not ok:
            warnings.append("DIRTY TREE ALLOWED (--allow-dirty): " + d)
            ok = True
        rec("git tree clean", ok, d)
        ok, d, w = chk.check_freeze(c.repo_root, c.python)
        rec("frozen eval set", ok, d)
        warnings += w
        rec("dataset", *chk.check_dataset(c.repo_root, run, c.manifest_path))
        rec("disk space", *chk.check_disk(c.paths.runs, c.min_free_gb))
        gpu_ok, d, snap = chk.check_gpu(c.gpu_mem_max_mib, c.gpu_wait_s)
        rec("GPU free", gpu_ok, d)
        try:
            ident = chk.hardware_identity(snap)
            if create_pin:
                rec("hardware pin", *chk.check_hardware_pin(c.paths.hw_pin, ident))
            elif os.path.exists(c.paths.hw_pin):
                rec("hardware pin", *chk.check_hardware_pin(c.paths.hw_pin, ident))
        except Exception as e:  # noqa: BLE001
            rec("hardware pin", False, "could not determine hardware identity: %s" % e)
            ident = {}
        failed = [(n, d) for n, ok, d in results if not ok]
        return {"ok": not failed, "failed": failed, "checks": results, "warnings": warnings,
                "commit": commit, "gpu": snap, "env": chk.environment_record(snap) if snap else {}}

    def _require_preflight(self, run: Run) -> Dict:
        pf = self.preflight(run)
        for w in pf["warnings"]:
            self.say("  preflight WARNING: %s" % w)
        if not pf["ok"]:
            raise StopQueue("preflight failed for %s: %s" % (run.run_id, "; ".join("%s: %s" % x for x in pf["failed"])),
                            run.run_id, fail_run=False)
        return pf

    # ----------------------------------------------------------- checkpoints
    @staticmethod
    def quarantine_bad_checkpoints(run_dir: str) -> List[str]:
        """A kill -9 during torch.save leaves a truncated newest .tar that
        would crash every resume. torch.save writes a zip archive, so check its
        CRCs and set a corrupt one aside, falling back to the previous one."""
        moved = []
        for name in reversed(st.checkpoints(run_dir)):
            p = os.path.join(run_dir, name)
            ok = zipfile.is_zipfile(p)
            if ok:
                try:
                    with zipfile.ZipFile(p) as z:
                        ok = z.testzip() is None
                except Exception:  # noqa: BLE001
                    ok = False
            if ok:
                break
            os.replace(p, p + ".corrupt")
            moved.append(name)
        return moved

    # ------------------------------------------------------------- training
    def _train_cmd(self, run: Run, resume: bool) -> List[str]:
        c = self.cfg
        cmd = [c.python, "-u", os.path.join(c.repo_root, "scripts", "train.py"), run.config_path,
               "--seed", str(run.seed), "--run-id", run.run_id, "--skip-final-eval",
               "--output-root", c.paths.runs, "--allow-dirty"]   # see D-040: preflight owns the tree check
        if resume:
            cmd.append("--resume")
        return cmd + list(c.extra_train_args)

    def _child_env(self):
        e = dict(os.environ)
        e["TQDM_MININTERVAL"] = "60"
        e["PYTHONUNBUFFERED"] = "1"
        return e

    def _run_child(self, cmd: List[str], rd: str, log_name: str, watch_loss: bool, iterations: int, run: Run) -> int:
        """Run one child to exit, pumping stdout to <rd>/<log_name> with timestamps.

        While it runs, a non-finite loss or a windowed divergence kills it and
        raises StopQueue(fail_run=True). Returns the exit code.
        """
        log_path = os.path.join(rd, log_name)
        err = open(os.path.join(rd, log_name.replace(".log", ".stderr.log")), "a")
        lf = open(log_path, "a")
        proc = self.popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                          cwd=self.cfg.repo_root, env=self._child_env(), preexec_fn=_pdeathsig)

        def pump():
            for line in proc.stdout:
                lf.write("%.3f %s" % (time.time(), line if line.endswith("\n") else line + "\n"))
                lf.flush()
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            while True:
                rc = proc.poll()
                if self.terminate.is_set():
                    self._kill(proc)
                    raise Interrupted()
                if watch_loss:
                    with open(log_path) as f:
                        health = chk.check_loss_health(chk.parse_loss_points(f), self.cfg.window, self.cfg.ratio)
                    if health["status"] in ("nonfinite", "diverging"):
                        self._kill(proc)
                        raise StopQueue("training %s: %s" % (health["status"], health["detail"]), run.run_id, True)
                if rc is not None:
                    break
                self.terminate.wait(self.cfg.poll_s)
            t.join(timeout=10)
            return rc
        finally:
            lf.close()
            err.close()

    @staticmethod
    def _kill(proc):
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _last_logged_iter(self, rd: str) -> int:
        p = os.path.join(rd, "train.log")
        if not os.path.exists(p):
            return 0
        with open(p) as f:
            pts = chk.parse_loss_points(f)
        return pts[-1][0] if pts else 0

    def _train(self, run: Run, iterations: int, rd: str) -> Dict:
        c = self.cfg
        retries = 0
        pf = None
        while True:
            pf = self._require_preflight(run)
            bad = self.quarantine_bad_checkpoints(rd)
            for b in bad:
                st.append_event(rd, "checkpoint_quarantined", checkpoint=b,
                                note="truncated/corrupt (kill during torch.save); resuming from the previous one")
            cks = st.checkpoints(rd)
            if not cks and os.path.isdir(os.path.join(rd, "tb")):
                aborted = "tb.aborted-%d" % int(time.time())
                os.replace(os.path.join(rd, "tb"), os.path.join(rd, aborted))
                st.append_event(rd, "restart_clean", note="no checkpoint to resume; earlier tensorboard dir moved to " + aborted)
            resume = bool(cks)
            kind = "resume" if resume else "start"
            resume_iter = int(cks[-1].split(".")[0]) if cks else 0
            st.append_event(rd, kind, attempt=retries, commit=pf["commit"], checkpoint=cks[-1] if cks else None,
                            from_iteration=resume_iter, env=pf["env"])
            self.say("  [%s] %s (attempt %d) from iteration %d" % (run.run_id, kind, retries, resume_iter))
            t0 = time.time()
            try:
                rc = self._run_child(self._train_cmd(run, resume), rd, "train.log", True, iterations, run)
            except Interrupted:
                st.append_event(rd, "interrupted", last_logged_iteration=self._last_logged_iter(rd),
                                note="harness stopped; run is PARTIAL and will resume from its latest checkpoint")
                raise
            except StopQueue as e:
                st.append_event(rd, "failed", reason=e.reason, last_logged_iteration=self._last_logged_iter(rd))
                raise
            wall = time.time() - t0
            st.append_event(rd, "attempt_end", attempt=retries, returncode=rc, wall_s=wall,
                            last_logged_iteration=self._last_logged_iter(rd))
            with open(os.path.join(rd, "train.log")) as f:
                health = chk.check_loss_health(chk.parse_loss_points(f), c.window, c.ratio)
            if health["status"] in ("nonfinite", "diverging"):
                raise StopQueue("training %s: %s" % (health["status"], health["detail"]), run.run_id, True)
            if rc == 0:
                return {"health": health, "preflight": pf}
            if retries >= c.max_retries:
                raise StopQueue("training exited with code %s and %d retries are exhausted" % (rc, c.max_retries),
                                run.run_id, True)
            retries += 1
            st.append_event(rd, "retry", retry=retries, after_returncode=rc,
                            last_logged_iteration=self._last_logged_iter(rd))
            self.say("  [%s] exit code %s -> retry %d/%d with --resume" % (run.run_id, rc, retries, c.max_retries))

    # ----------------------------------------------------------- evaluation
    def _evaluate(self, run: Run, iterations: int, rd: str) -> Dict:
        c = self.cfg
        final = os.path.join(rd, st.final_ckpt_name(iterations))
        eval_json = os.path.join(rd, "eval_%s.json" % run.run_id)
        for attempt in range(c.max_retries + 1):
            self._require_preflight(run)
            if os.path.exists(eval_json):
                os.remove(eval_json)
            cmd = [c.python, os.path.join(c.repo_root, "scripts", "evaluate.py"), run.config_path, final,
                   "--seed", str(run.seed), "--run-id", run.run_id]
            st.append_event(rd, "eval_start", attempt=attempt)
            rc = self._run_child(cmd, rd, "eval.log", False, iterations, run)
            st.append_event(rd, "eval_end", attempt=attempt, returncode=rc)
            if rc == 0 and os.path.exists(eval_json):
                break
            if attempt >= c.max_retries:
                raise StopQueue("evaluate.py exited with code %s; retries exhausted" % rc, run.run_id, True)
        with open(eval_json) as f:
            ev = json.load(f)
        return ev

    # ----------------------------------------------------------- validation
    def _post_validate(self, run: Run, iterations: int, rd: str, start_commit: str, ev: Dict):
        problems = []
        final = os.path.join(rd, st.final_ckpt_name(iterations))
        if not os.path.exists(final):
            problems.append("final checkpoint %s missing" % final)
        summ = os.path.join(rd, "summary.json")
        if not os.path.exists(summ):
            problems.append("summary.json missing")
        else:
            s = json.load(open(summ))
            if s.get("iterations") != iterations:
                problems.append("summary.json iterations=%r != config %r" % (s.get("iterations"), iterations))
            if s.get("seed") != run.seed or s.get("condition_id") != run.condition_id:
                problems.append("summary.json identity (%s, seed %s) != (%s, %s)"
                                % (s.get("condition_id"), s.get("seed"), run.condition_id, run.seed))
            if s.get("dataset_id") != run.dataset_id:
                problems.append("summary.json dataset_id %r != expected %r" % (s.get("dataset_id"), run.dataset_id))
            if s.get("n_poisoned_verified") != run.n_poisoned_expected:
                problems.append("verified poisoned views %r != %d" % (s.get("n_poisoned_verified"), run.n_poisoned_expected))
        import math
        psnr = ev.get("test_psnr")
        if psnr is None or not math.isfinite(psnr) or not ev.get("n_test_views"):
            problems.append("eval output invalid: %s" % ev)
        if not str(ev.get("checkpoint", "")).endswith(st.final_ckpt_name(iterations)):
            problems.append("eval scored %r, not the final checkpoint" % ev.get("checkpoint"))
        ok, d, commit = chk.check_git_clean(self.cfg.repo_root)
        if commit != start_commit:
            problems.append("git HEAD changed during the run (%s -> %s)" % (start_commit[:10], commit[:10]))
        elif not ok and not self.cfg.allow_dirty:
            problems.append("working tree became dirty during the run: " + d)
        if problems:
            raise StopQueue("post-run validation failed: " + "; ".join(problems), run.run_id, True)

    # --------------------------------------------------------------- one run
    def run_one(self, run: Run, iterations: int) -> None:
        rd = self.cfg.paths.run_dir(run.run_id)
        info = st.classify_run(self.cfg.paths, run.run_id, iterations)
        if info["state"] == "COMPLETE":
            return
        if info["state"] == "FAILED":
            raise StopQueue("%s is FAILED (%s); move its run directory aside after investigating"
                            % (run.run_id, info.get("reason")), run.run_id, False)
        os.makedirs(rd, exist_ok=True)
        self.say("[%s] state %s, next action %s" % (run.run_id, info["state"], info["next"]))
        pf = None
        if info["next"] == "train":
            res = self._train(run, iterations, rd)
            pf = res["preflight"]
        else:
            pf = self._require_preflight(run)
        start_commit = pf["commit"]
        info = st.classify_run(self.cfg.paths, run.run_id, iterations)
        ev = None
        if not info["has_eval"]:
            ev = self._evaluate(run, iterations, rd)
        else:
            ev = json.load(open(os.path.join(rd, "eval_%s.json" % run.run_id)))
        try:
            self._post_validate(run, iterations, rd, start_commit, ev)
        except StopQueue as e:
            st.append_event(rd, "failed", reason=e.reason)
            raise
        from utils.sweep_record import write_run_record
        write_run_record(self.cfg, run, iterations, rd, ev, pf)
        self.say("[%s] COMPLETE (test PSNR %.3f dB over %d views)" % (run.run_id, ev["test_psnr"], ev["n_test_views"]))

    # ----------------------------------------------------------------- queue
    def queue(self) -> List[Run]:
        q = build_queue(self.cfg.config_dir, self.cfg.manifest_path)
        if self.cfg.only:
            q = [r for r in q if r.run_id in set(self.cfg.only)]
        return q

    def iterations_for(self, run: Run) -> int:
        return int(load_config(run.config_path)["training"]["iterations"])

    def run_queue(self) -> int:
        c = self.cfg
        try:
            lock = st.SweepLock(c.paths.lockfile)
            lock.acquire()
        except st.LockHeld as e:
            self.say("REFUSED: %s" % e)
            return EXIT_LOCKED
        try:
            stopped = st.read_stopped(c.paths)
            if stopped:
                self.say("STOPPED marker present (%s): %s\n  %s" % (stopped.get("run_id"), stopped.get("reason"),
                                                                    stopped.get("how_to_resume")))
                return EXIT_STOPPED
            prev = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            for s in prev:
                signal.signal(s, lambda *_: self.terminate.set())
            try:
                for run in self.queue():
                    if os.path.exists(c.paths.stop_after_current):
                        os.remove(c.paths.stop_after_current)
                        self.say("graceful stop requested: stopping before %s" % run.run_id)
                        return EXIT_OK
                    iters = self.iterations_for(run)
                    try:
                        self.run_one(run, iters)
                    except StopQueue as e:
                        rd = c.paths.run_dir(e.run_id or run.run_id)
                        if e.fail_run:
                            st.write_failed(rd, e.reason)
                        st.write_stopped(c.paths, e.run_id or run.run_id, e.reason)
                        st.append_event(rd, "stopped", reason=e.reason, run_failed=e.fail_run)
                        self.say("STOPPED: %s" % e.reason)
                        return EXIT_STOPPED
                    except Interrupted:
                        self.say("interrupted: the child was stopped; %s will resume from its latest checkpoint" % run.run_id)
                        return EXIT_OK
                self.say("queue complete")
                return EXIT_OK
            finally:
                for s, h in prev.items():
                    signal.signal(s, h)
        finally:
            lock.release()
