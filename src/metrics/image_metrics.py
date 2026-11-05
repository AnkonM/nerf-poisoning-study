"""§6 metric primitives (D-042 definitions D-G). All images float32 [H,W,3] in [0,1],
data_range = 1.0. Regions are bool [H,W]. Pure functions except the LPIPS wrapper.

Per image pair, the dense per-pixel quantities (squared error, SSIM map, LPIPS map)
are computed ONCE and then pooled over any region, so every region sees the same maps.
"""

import hashlib
import importlib.metadata
import os
from typing import Dict, List, Tuple

import numpy as np
from skimage.metrics import structural_similarity

SSIM_KW = dict(data_range=1.0, channel_axis=-1, gaussian_weights=True, sigma=1.5,
               win_size=11, use_sample_covariance=False, full=True)

# torchvision alexnet-owt weights, the LPIPS(net='alex') backbone. Pinned full sha256.
ALEXNET_FILE = "alexnet-owt-7be5be79.pth"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("bad image shapes %s vs %s" % (a.shape, b.shape))
    for x in (a, b):
        if x.dtype != np.float32:
            raise ValueError("images must be float32, got %s" % x.dtype)
        if not np.isfinite(x).all():
            raise ValueError("non-finite pixel values")


def _check_region(region: np.ndarray, shape) -> None:
    if region.dtype != bool or region.shape != tuple(shape[:2]):
        raise ValueError("region must be bool %s, got %s %s" % (shape[:2], region.dtype, region.shape))
    if not region.any():
        raise ValueError("empty region")


def psnr_from_mse(mse: float) -> float:
    """-10 log10(mse), data_range 1. mse == 0 -> inf (recorded as inf, never hidden)."""
    return float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))


def sq_err_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel squared error averaged over channels, float64 [H,W]."""
    _check_pair(a, b)
    return ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(axis=-1)


def region_mse(sq_map: np.ndarray, region: np.ndarray) -> float:
    """MSE pooled over the region's pixels and channels (channels already averaged)."""
    _check_region(region, sq_map.shape)
    return float(sq_map[region].mean())


def psnr_region(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> Tuple[float, float]:
    mse = region_mse(sq_err_map(a, b), region)
    return mse, psnr_from_mse(mse)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full SSIM map [H,W] (channel-averaged), D-042 E parameters."""
    _check_pair(a, b)
    _, s = structural_similarity(a, b, **SSIM_KW)
    return s.mean(axis=-1).astype(np.float64)


def region_mean(dense: np.ndarray, region: np.ndarray) -> float:
    _check_region(region, dense.shape)
    return float(dense[region].mean())


def lpips_weight_paths() -> Dict[str, str]:
    import lpips
    import torch
    return {
        "backbone": os.path.join(torch.hub.get_dir(), "checkpoints", ALEXNET_FILE),
        "linear": os.path.join(os.path.dirname(lpips.__file__), "weights", "v0.1", "alex.pth"),
    }


class LpipsMeter:
    """lpips net='alex', spatial=True, inputs in [-1,1]. Loads the backbone from the
    local file ONLY (never downloads) and fails loudly if it is missing or its sha256
    differs from the pinned value. CUDA only; no CPU fallback."""

    def __init__(self, batch_size: int = 4):
        import torch
        import lpips
        if not torch.cuda.is_available():
            raise RuntimeError("LPIPS requires CUDA here; refusing to fall back to CPU silently")
        paths = lpips_weight_paths()
        for k, p in paths.items():
            if not os.path.isfile(p):
                raise FileNotFoundError("LPIPS %s weights missing: %s (no download is attempted)" % (k, p))
        self.weights_sha256 = {k: sha256_file(p) for k, p in paths.items()}
        if self.weights_sha256["backbone"] != ALEXNET_SHA256:
            raise ValueError("alexnet backbone sha256 %s != pinned %s"
                             % (self.weights_sha256["backbone"], ALEXNET_SHA256))
        self.paths = paths
        self.version = importlib.metadata.version("lpips")
        # pnet_rand=True: do NOT let torchvision fetch weights; real ones are loaded below.
        net = lpips.LPIPS(net="alex", spatial=True, pnet_rand=True, verbose=False)
        sd = torch.load(paths["backbone"], map_location="cpu", weights_only=True)
        slice_of = {}
        for k, rng in enumerate([(0, 2), (2, 5), (5, 8), (8, 10), (10, 12)], start=1):
            for i in range(*rng):
                slice_of[i] = k
        mapped = {}
        for key, v in sd.items():
            if not key.startswith("features."):
                continue
            _, idx, kind = key.split(".")
            mapped["net.slice%d.%s.%s" % (slice_of[int(idx)], idx, kind)] = v
        own = {k for k in net.state_dict() if k.startswith("net.")}
        if set(mapped) != own:
            raise ValueError("backbone key mismatch: %s" % sorted(set(mapped) ^ own))
        missing, unexpected = net.load_state_dict(mapped, strict=False)
        if unexpected or any(k.startswith("net.") for k in missing):
            raise ValueError("backbone load problem: missing=%s unexpected=%s" % (missing, unexpected))
        self.net = net.cuda().eval()
        self.batch_size = int(batch_size)
        self.oom_fallback = False

    def _maps(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import torch
        ta = torch.from_numpy(a).permute(0, 3, 1, 2).cuda() * 2.0 - 1.0
        tb = torch.from_numpy(b).permute(0, 3, 1, 2).cuda() * 2.0 - 1.0
        with torch.no_grad():
            d = self.net(ta, tb)             # [B,1,H,W]
        out = d[:, 0].double().cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError("non-finite LPIPS map")
        return out

    def maps(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a, b: float32 [N,H,W,3] in [0,1] -> [N,H,W] float64 spatial LPIPS maps."""
        import torch
        if a.shape != b.shape or a.ndim != 4 or a.shape[-1] != 3:
            raise ValueError("bad shapes %s %s" % (a.shape, b.shape))
        outs: List[np.ndarray] = []
        i = 0
        while i < a.shape[0]:
            n = self.batch_size
            try:
                outs.append(self._maps(a[i:i + n], b[i:i + n]))
                i += n
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size = 1
                self.oom_fallback = True
                print("WARNING: LPIPS CUDA OOM; dropping to batch size 1")
        return np.concatenate(outs, 0)
