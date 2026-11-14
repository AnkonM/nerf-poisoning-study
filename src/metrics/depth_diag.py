"""Exploratory depth / opacity diagnostics (D-044 Part 3). Statistics are pure numpy; `render_depth_acc`
is the only GPU function and refuses to run on CPU. Reuses the existing render path; changes nothing in
src/nerf."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata


# ---- statistics (pure) ------------------------------------------------------------------------
def region_mean(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-view mean of arr[v] over mask[v]; arr, mask: [V,H,W]. Raises if a view's mask is empty."""
    if arr.shape != mask.shape:
        raise ValueError("shape mismatch %s vs %s" % (arr.shape, mask.shape))
    n = mask.reshape(mask.shape[0], -1).sum(1)
    if (n == 0).any():
        raise ValueError("empty mask in view(s) %s" % np.where(n == 0)[0].tolist())
    return np.array([float(arr[v][mask[v]].astype(np.float64).mean()) for v in range(arr.shape[0])])


def region_mean_abs_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return region_mean(np.abs(a.astype(np.float64) - b.astype(np.float64)), mask)


def depth_from_disp(disp: np.ndarray) -> np.ndarray:
    """Expected depth = 1 / disp_map = sum(w z) / sum(w) (acc-normalised expected termination distance)."""
    return 1.0 / np.asarray(disp, dtype=np.float64)


def summarise(x: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max())}


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties); descriptive, no p-value."""
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bin_means(x: Sequence[float], y: Sequence[float], edges: Sequence[float]) -> List[Dict]:
    """Mean of y within bins of x: [-inf,e0), [e0,e1), ..., [e_last, inf)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo = [-np.inf] + list(edges)
    hi = list(edges) + [np.inf]
    out = []
    for l, h in zip(lo, hi):
        sel = (x >= l) & (x < h)
        label = ("<%g" % h) if l == -np.inf else ((">=%g" % l) if h == np.inf else ("%g-%g" % (l, h)))
        out.append({"bin_deg": label, "n": int(sel.sum()),
                    "mean_y": float(y[sel].mean()) if sel.any() else float("nan")})
    return out


# ---- GPU render (existing path) ---------------------------------------------------------------
def render_depth_acc(render_fn, poses: np.ndarray, device, chunk: int, min_chunk: int, on_view=None,
                     log=print) -> Tuple[np.ndarray, np.ndarray, int]:
    """Render each pose with `render_fn(c2w, chunk) -> (rgb, disp, acc)` (torch tensors on `device`),
    halving `chunk` on CUDA OOM down to `min_chunk`. Returns (depth[V,H,W] f32, acc[V,H,W] f32, final chunk).
    `on_view(i, rgb_np)` is called per view. Refuses CPU."""
    import torch
    if getattr(device, "type", str(device)) != "cuda":
        raise RuntimeError("depth/opacity diagnostics run on CUDA only (no CPU fallback); device=%s" % (device,))
    depth, acc = [], []
    for i in range(len(poses)):
        while True:
            try:
                with torch.no_grad():
                    rgb, disp, a = render_fn(torch.Tensor(poses[i]).to(device)[:3, :4], chunk)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk // 2 < min_chunk:
                    raise RuntimeError("CUDA OOM at the minimum ray chunk %d (view %d); stopping, no CPU fallback"
                                       % (chunk, i))
                chunk //= 2
                log("WARNING: CUDA OOM; ray chunk halved to %d" % chunk)
        d = depth_from_disp(disp.float().cpu().numpy()).astype(np.float32)
        ac = a.float().cpu().numpy().astype(np.float32)
        if not (np.isfinite(d).all() and np.isfinite(ac).all()):
            raise RuntimeError("non-finite depth/acc in view %d" % i)
        depth.append(d); acc.append(ac)
        if on_view is not None:
            on_view(i, rgb.float().cpu().numpy())
        if (i + 1) % 5 == 0:
            log("  rendered %d/%d views (chunk %d)" % (i + 1, len(poses), chunk))
    return np.stack(depth), np.stack(acc), chunk
