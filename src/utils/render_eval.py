"""Render the 75 frozen eval views from a run's FINAL checkpoint (Phase 6 closeout, D-041).

Uses exactly evaluate.py's path: same config, same dataset resolution, same
load_blender_data + white-background compositing, same _build_model/_render_kwargs
(perturb 0, raw_noise 0), same `render(...)` call with the same chunk, fp32, no AMP.
Only the per-view PSNR bookkeeping is local, because evaluate_psnr returns one mean
and discards the images. This module computes NO METHODOLOGY §6 metric.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

N_VIEWS = 75
PSNR_TOL_DB = 0.01
META = "render_meta.json"
NPZ = "renders_fp16.npz"


def view_png_name(i: int) -> str:
    return "eval_%03d.png" % i


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def psnr_from_arrays(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean of per-view PSNR (dB), the same quantity evaluate_psnr averages (float64 here)."""
    mse = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).reshape(pred.shape[0], -1).mean(axis=1)
    return float(np.mean(-10.0 * np.log10(mse)))


def validate_renders(final_or_tmp_dir: str, gt: np.ndarray, harness_psnr: float,
                     tol_db: float = PSNR_TOL_DB, n_views: int = N_VIEWS) -> Dict[str, Any]:
    """Gate checks for one run's output directory. `gt`: [75,H,W,3] float32 (white-composited)."""
    res: Dict[str, Any] = {"ok": False, "problems": []}
    npz_path = os.path.join(final_or_tmp_dir, NPZ)
    if not os.path.isfile(npz_path):
        res["problems"].append("missing %s" % NPZ)
        return res
    with np.load(npz_path) as z:
        r = z["renders"]
        idx = z["view_index"]
    res["n_views"] = int(r.shape[0])
    if r.dtype != np.float16:
        res["problems"].append("dtype %s != float16" % r.dtype)
    if r.shape != (n_views,) + tuple(gt.shape[1:]) or gt.shape[0] != n_views:
        res["problems"].append("shape %s vs gt %s (expected %d views)" % (r.shape, gt.shape, n_views))
        return res
    if list(idx) != list(range(n_views)):
        res["problems"].append("view_index is not 0..%d" % (n_views - 1))
    if not np.isfinite(r.astype(np.float32)).all():
        res["problems"].append("NaN/inf in renders")
        return res
    pngs = [f for f in os.listdir(final_or_tmp_dir) if f.startswith("eval_") and f.endswith(".png")]
    res["n_png"] = len(pngs)
    if sorted(pngs) != [view_png_name(i) for i in range(n_views)]:
        res["problems"].append("PNG set is not eval_000..eval_%03d (found %d)" % (n_views - 1, len(pngs)))
    recomputed = psnr_from_arrays(r, gt)
    res["psnr_recomputed_fp16"] = recomputed
    res["psnr_harness"] = harness_psnr
    res["deviation_db"] = abs(recomputed - harness_psnr)
    if not res["deviation_db"] <= tol_db:
        res["problems"].append("PSNR deviation %.5f dB > %.3f dB" % (res["deviation_db"], tol_db))
    res["ok"] = not res["problems"]
    return res


def meta_is_valid(render_dir: str, checkpoint_path: str, eval_manifest_sha: str) -> bool:
    """Cheap structural check used for resume/skip. True only if a COMPLETE, validated render exists."""
    mp = os.path.join(render_dir, META)
    if not os.path.isfile(mp):
        return False
    try:
        with open(mp) as f:
            m = json.load(f)
        if not m.get("validation", {}).get("ok"):
            return False
        if m.get("eval_manifest_sha256") != eval_manifest_sha:
            return False
        if m.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            return False
        if not os.path.isfile(os.path.join(render_dir, NPZ)):
            return False
        pngs = [f for f in os.listdir(render_dir) if f.startswith("eval_") and f.endswith(".png")]
        return len(pngs) == m.get("n_views", N_VIEWS)
    except (OSError, ValueError, KeyError):
        return False


def tmp_dir_for(final_dir: str) -> str:
    return final_dir.rstrip("/") + ".partial"


def eval_set_status(repo_root: str) -> Dict[str, str]:
    """Run freeze_eval_set.py --verify; raise unless it PASSes. Returns manifest file sha + aggregate."""
    p = subprocess.run([sys.executable, os.path.join(repo_root, "scripts", "freeze_eval_set.py"), "--verify"],
                       capture_output=True, text=True)
    if p.returncode != 0 or "VERIFY: PASS" not in p.stdout:
        raise RuntimeError("freeze_eval_set.py --verify FAILED:\n%s\n%s" % (p.stdout, p.stderr))
    agg = [l.split(":", 1)[1].strip() for l in p.stdout.splitlines() if l.startswith("recomputed aggregate")][0]
    manifest = os.path.join(repo_root, "data", "blender_scenes", "eval_holdout_SHA256SUMS.txt")
    return {"manifest_sha256": sha256_file(manifest), "aggregate_sha256": agg}


def _git(repo_root: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo_root, *args], capture_output=True, text=True).stdout.strip()


def _env_info() -> Dict[str, str]:
    import torch
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"torch": torch.__version__, "cuda_build": str(torch.version.cuda), "driver": drv,
            "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
            "hostname": platform.node()}


def load_gt_independent(repo_root: str, cfg: Dict[str, Any]) -> np.ndarray:
    """Frozen ground-truth eval images, loaded separately from the render path: white-composited RGB [75,H,W,3]."""
    from nerf.datasets.blender import load_blender_data
    d = cfg["dataset"]
    base = os.path.join(repo_root, "data", "blender_scenes")
    imgs, _, _, i_split = load_blender_data(base, half_res=d.get("half_res", False), testskip=d.get("testskip", 1),
                                            val_dir=base, test_dir=base)
    imgs = imgs[i_split[2]]
    return (imgs[..., :3] * imgs[..., -1:] + (1.0 - imgs[..., -1:])).astype(np.float32)


def render_run(repo_root: str, run_id: str, renders_root: str) -> Dict[str, Any]:
    """Render + validate one run in THIS process. Output dir appears only on validated completion."""
    import imageio
    import torch
    from nerf.datasets.blender import load_blender_data
    from nerf.metrics import img2mse, mse2psnr, to8b
    from nerf.rays import get_rays  # noqa: F401  (render() uses it)
    from nerf.rendering import render
    from nerf.training import _build_model, _render_kwargs, device
    from utils.config import load_config
    from utils.dataset_id import resolve_dataset

    run_dir = os.path.join(repo_root, "experiments", "runs", run_id)
    with open(os.path.join(run_dir, "summary.json")) as f:
        summ = json.load(f)
    ckpt_path = os.path.join(run_dir, "%06d.tar" % int(summ["iterations"]))
    if os.path.realpath(ckpt_path) != os.path.realpath(summ["final_checkpoint"]):
        raise RuntimeError("final checkpoint mismatch: %s vs %s" % (ckpt_path, summ["final_checkpoint"]))
    with open(os.path.join(run_dir, "eval_%s.json" % run_id)) as f:
        harness = json.load(f)
    if os.path.realpath(harness["checkpoint"]) != os.path.realpath(ckpt_path):
        raise RuntimeError("harness eval JSON scored a different checkpoint")

    es = eval_set_status(repo_root)
    cfg = load_config(summ["config_path"])
    seed = int(summ["seed"])
    cfg["reproducibility"]["seed"] = seed
    info = resolve_dataset(cfg, seed, repo_root=repo_root)
    dcfg = dict(cfg["dataset"], path=os.path.join(repo_root, info["dataset_path"]))
    images, poses, hwf, i_split = load_blender_data(
        dcfg["path"], half_res=dcfg.get("half_res", False), testskip=dcfg.get("testskip", 1),
        val_dir=dcfg.get("val_path"), test_dir=dcfg.get("test_path"))
    i_test = i_split[2]
    if len(i_test) != N_VIEWS:
        raise RuntimeError("expected %d eval views, got %d" % (N_VIEWS, len(i_test)))
    if cfg["render"].get("white_background", True):
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]
    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    ckpt = torch.load(ckpt_path, map_location=device)
    model, model_fine, _, qfn, use_viewdirs = _build_model(cfg)
    model.load_state_dict(ckpt["network_fn_state_dict"])
    if model_fine is not None and ckpt.get("network_fine_state_dict") is not None:
        model_fine.load_state_dict(ckpt["network_fine_state_dict"])
    model.eval()
    if model_fine is not None:
        model_fine.eval()
    kwargs = _render_kwargs(cfg, qfn, model, model_fine, use_viewdirs, False)
    chunk = cfg["training"]["chunk_size"]

    final_dir = os.path.join(renders_root, run_id)
    tmp = tmp_dir_for(final_dir)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)           # our own incomplete output from an interrupted render
    os.makedirs(tmp)

    t0 = time.time()
    out16 = np.zeros((N_VIEWS, H, W, 3), dtype=np.float16)
    psnr32: List[float] = []
    with torch.no_grad():
        for k, idx in enumerate(i_test):
            c2w = torch.Tensor(poses[idx]).to(device)
            target = torch.Tensor(images[idx]).to(device)
            rgb, _, _, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **kwargs)
            psnr32.append(mse2psnr(img2mse(rgb, target)).item())
            arr = rgb.float().cpu().numpy()
            out16[k] = arr.astype(np.float16)
            imageio.imwrite(os.path.join(tmp, view_png_name(k)), to8b(arr))
            print("[render] %s view %d/%d psnr %.3f (%.1fs elapsed)" % (run_id, k + 1, N_VIEWS, psnr32[-1],
                                                                       time.time() - t0), flush=True)
    peak_gib = torch.cuda.max_memory_reserved() / 2 ** 30
    np.savez(os.path.join(tmp, NPZ), renders=out16, view_index=np.arange(N_VIEWS),
             frame_files=np.array(["test_%d" % i for i in range(N_VIEWS)]))

    gt = load_gt_independent(repo_root, cfg)
    val = validate_renders(tmp, gt, float(harness["test_psnr"]))
    val["psnr_render_time_fp32"] = float(np.mean(psnr32))
    val["deviation_fp32_db"] = abs(val["psnr_render_time_fp32"] - float(harness["test_psnr"]))
    val["tolerance_db"] = PSNR_TOL_DB

    meta = {
        "run_id": run_id, "condition": summ["condition_id"], "seed": seed,
        "checkpoint_path": os.path.relpath(ckpt_path, repo_root), "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_iteration": int(ckpt.get("iteration", ckpt.get("global_step", -1))),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty_paths": _git(repo_root, "status", "--porcelain").splitlines()[:20],
        "environment": _env_info(),
        "render_settings": {
            "config_path": summ["config_path"], "half_res": dcfg.get("half_res", False), "H": H, "W": W,
            "focal": float(focal), "chunk_size": chunk, "netchunk": cfg["training"]["netchunk"],
            "near": cfg["dataset"]["near"], "far": cfg["dataset"]["far"],
            "num_coarse_samples": cfg["render"]["num_coarse_samples"],
            "num_fine_samples": cfg["render"]["num_fine_samples"], "perturb": 0.0, "raw_noise_std": 0.0,
            "white_background": cfg["render"].get("white_background", True), "use_viewdirs": use_viewdirs,
            "precision": "fp32 (no AMP/TF32 enabled by this code); stored as float16",
            "dataset_id": info["dataset_id"], "n_views": N_VIEWS,
            "peak_vram_reserved_gib": round(peak_gib, 2),
        },
        "n_views": N_VIEWS,
        "eval_manifest_sha256": es["manifest_sha256"], "eval_set_aggregate_sha256": es["aggregate_sha256"],
        "harness_eval_json": os.path.relpath(os.path.join(run_dir, "eval_%s.json" % run_id), repo_root),
        "render_minutes": (time.time() - t0) / 60, "validation": val,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(tmp, META), "w") as f:
        json.dump(meta, f, indent=2)
    if not val["ok"]:
        failed = final_dir + ".FAILED_VALIDATION"
        if os.path.isdir(failed):
            shutil.rmtree(failed)
        os.rename(tmp, failed)
        raise RuntimeError("validation FAILED for %s: %s (left at %s)" % (run_id, val["problems"], failed))
    os.rename(tmp, final_dir)
    return meta


def run_all(repo_root: str, run_ids: List[str], renders_root: str, log) -> Dict[str, str]:
    """Sequential orchestrator: one child process per run (VRAM freed between runs); resumable; one retry."""
    es = eval_set_status(repo_root)
    log("eval set verify PASS, aggregate %s, manifest sha %s" % (es["aggregate_sha256"], es["manifest_sha256"]))
    status: Dict[str, str] = {}
    for rid in run_ids:
        run_dir = os.path.join(repo_root, "experiments", "runs", rid)
        with open(os.path.join(run_dir, "summary.json")) as f:
            ckpt = os.path.join(run_dir, "%06d.tar" % int(json.load(f)["iterations"]))
        final_dir = os.path.join(renders_root, rid)
        if meta_is_valid(final_dir, ckpt, es["manifest_sha256"]):
            log("%s: valid render exists, skipping" % rid)
            status[rid] = "skipped-valid"
            continue
        if os.path.isdir(final_dir):
            log("%s: render dir exists but is not valid; re-rendering" % rid)
            shutil.move(final_dir, final_dir + ".invalid-%d" % int(time.time()))
        for attempt in (1, 2):
            log("%s: render attempt %d" % (rid, attempt))
            rc = subprocess.call([sys.executable, "-u", os.path.join(repo_root, "scripts", "render_eval_views.py"),
                                  "--run-id", rid, "--renders-root", renders_root])
            if rc == 0:
                status[rid] = "rendered"
                log("%s: DONE" % rid)
                break
            log("%s: attempt %d FAILED rc=%d" % (rid, attempt, rc))
            status[rid] = "FAILED"
            if os.path.isdir(final_dir + ".FAILED_VALIDATION"):
                shutil.move(final_dir + ".FAILED_VALIDATION", final_dir + ".FAILED_VALIDATION.attempt%d" % attempt)
    return status
