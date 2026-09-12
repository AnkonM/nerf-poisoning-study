"""Volumetric rendering: querying the MLP along rays and compositing.

Vendored/adapted from yenchenlin/nerf-pytorch's run_nerf.py and
run_nerf_helpers.py (MIT license, see src/nerf/THIRD_PARTY_LICENSE),
commit 63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd. Math is unchanged from
the original; only reorganized into this project's module layout and
stripped of the original's LLFF/NDC and deepvoxels/LINEMOD branches,
which this project's Blender-synthetic-only scope (docs/METHODOLOGY.md
§3) never exercises.
"""

import torch
import torch.nn.functional as F

from nerf.rays import get_rays, sample_pdf


def batchify(fn, chunk):
    """Apply `fn` to `chunk`-sized slices, to bound memory use."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk):
    """Encode points (+ optionally view directions) and query the MLP."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False):
    """Turn raw (rgb, sigma) network output into per-ray composited color."""
    device = raw.device
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)  # noqa: E731

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    weights = (
        alpha
        * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.0 - alpha + 1e-10], -1), -1
        )[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
):
    """Sample points along each ray, query the model(s), and composite."""
    device = ray_batch.device
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd
    )

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd
        )

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)

    return ret


def batchify_rays(rays_flat, chunk, **kwargs):
    """Render rays in chunk-sized minibatches to bound memory use."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            all_ret.setdefault(k, []).append(ret[k])
    return {k: torch.cat(v, 0) for k, v in all_ret.items()}


def render(H, W, K, chunk, rays=None, c2w=None, near=0.0, far=1.0, use_viewdirs=False, **kwargs):
    """Render a batch of rays, or a full image if `c2w` is given."""
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near_t, far_t = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays_batch = torch.cat([rays_o, rays_d, near_t, far_t], -1)
    if use_viewdirs:
        rays_batch = torch.cat([rays_batch, viewdirs], -1)

    all_ret = batchify_rays(rays_batch, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]
