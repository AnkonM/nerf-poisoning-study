"""Ray generation and hierarchical (importance) sampling.

Vendored/adapted from yenchenlin/nerf-pytorch's run_nerf_helpers.py (MIT
license, see src/nerf/THIRD_PARTY_LICENSE), commit
63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd. Math is unchanged from the
original.
"""

import numpy as np
import torch


def get_rays(H: int, W: int, K, c2w: torch.Tensor):
    """Camera rays for a full H x W image, in world space (torch).

    Note: the vendored original relies on a since-deprecated global
    `torch.set_default_tensor_type('torch.cuda.FloatTensor')` call to put
    every newly-created tensor on the GPU. We don't carry that over (it's
    deprecated and applies process-wide, which is a bad fit for a config-
    driven pipeline), so tensors created here are placed explicitly on
    `c2w`'s device instead.
    """
    device = c2w.device
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device)
    )
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
    )
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H: int, W: int, K, c2w: np.ndarray):
    """Numpy counterpart of get_rays, used for the ray-pool batching path."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def sample_pdf(bins, weights, n_samples: int, det: bool = False):
    """Inverse-CDF hierarchical sampling (NeRF paper, section 5.2)."""
    device = bins.device
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    if det:
        u = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
