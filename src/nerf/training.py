"""Config-driven training loop for the vendored vanilla NeRF.

Adapted from yenchenlin/nerf-pytorch's run_nerf.py `train()` (MIT license,
see src/nerf/THIRD_PARTY_LICENSE), commit
63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd. The optimization math (ray
sampling, precrop, loss, hierarchical sampling, exponential LR decay) is
unchanged; the CLI/configargparse-driven structure is replaced with a
single `train_from_config(config, ...)` entry point reading resolved
config dicts, per docs/PROJECT_STRUCTURE.md. Only the Blender-synthetic
dataset path is ported (LLFF/deepvoxels/LINEMOD branches are dropped —
out of scope per docs/METHODOLOGY.md §3's synthetic-scene-only design),
and only the "one random image per step + optional center-crop warmup"
ray-sampling strategy is ported (the original's alternative "pool all
rays up front" strategy is unused by every config this project defines).
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from nerf.datasets.blender import load_blender_data
from nerf.metrics import img2mse, mse2psnr
from nerf.model import NeRF, get_embedder
from nerf.rays import get_rays
from nerf.rendering import render, run_network
from utils.seeding import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(cfg: Dict[str, Any]):
    model_cfg = cfg["model"]
    render_cfg = cfg["render"]

    embed_fn, input_ch = get_embedder(model_cfg["pos_encoding_freqs"])
    embeddirs_fn, input_ch_views = get_embedder(model_cfg["dir_encoding_freqs"])

    use_viewdirs = render_cfg.get("use_viewdirs", True)
    output_ch = 5 if render_cfg["num_fine_samples"] > 0 else 4
    skips = [4]

    model = NeRF(
        D=model_cfg["num_layers"],
        W=model_cfg["hidden_dim"],
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views if use_viewdirs else 0,
        use_viewdirs=use_viewdirs,
    ).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if render_cfg["num_fine_samples"] > 0:
        model_fine = NeRF(
            D=model_cfg["num_layers"],
            W=model_cfg["hidden_dim"],
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views if use_viewdirs else 0,
            use_viewdirs=use_viewdirs,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, fn: run_network(  # noqa: E731
        inputs,
        viewdirs,
        fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn if use_viewdirs else None,
        netchunk=cfg["training"]["netchunk"],
    )

    return model, model_fine, grad_vars, network_query_fn, use_viewdirs


def _render_kwargs(cfg, network_query_fn, model, model_fine, use_viewdirs, train_mode: bool):
    render_cfg = cfg["render"]
    kwargs = {
        "network_query_fn": network_query_fn,
        "perturb": render_cfg.get("perturb", 1.0) if train_mode else 0.0,
        "N_importance": render_cfg["num_fine_samples"],
        "network_fine": model_fine,
        "N_samples": render_cfg["num_coarse_samples"],
        "network_fn": model,
        "use_viewdirs": use_viewdirs,
        "white_bkgd": render_cfg.get("white_background", True),
        "raw_noise_std": render_cfg.get("raw_noise_std", 0.0) if train_mode else 0.0,
        "near": cfg["dataset"]["near"],
        "far": cfg["dataset"]["far"],
    }
    return kwargs


def evaluate_psnr(
    images: np.ndarray,
    poses: np.ndarray,
    indices: np.ndarray,
    hwf,
    K,
    chunk: int,
    render_kwargs_test: Dict[str, Any],
    show_progress: bool = False,
) -> float:
    """Render each of `indices` and return the mean per-image PSNR (dB).

    `show_progress`: report per-image timing/PSNR via tqdm as it goes,
    rather than staying silent until the whole set is done. This phase (a
    full-image render loop, as opposed to the cheap per-ray training
    steps) can legitimately take a long time — see docs/DECISION_LOG.md's
    Phase 2 incident notes — and having no progress signal made a slow
    but healthy run indistinguishable from a hang. Off by default so the
    periodic small-subset call during training doesn't spam the log.
    """
    H, W, _ = hwf
    psnrs = []
    iterator = tqdm(indices, desc="test-set eval") if show_progress else indices
    with torch.no_grad():
        for idx in iterator:
            t0 = time.time()
            c2w = torch.Tensor(poses[idx]).to(device)
            target = torch.Tensor(images[idx]).to(device)
            rgb, _, _, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs_test)
            mse = img2mse(rgb, target)
            image_psnr = mse2psnr(mse).item()
            psnrs.append(image_psnr)
            if show_progress:
                tqdm.write(
                    f"[eval] view {idx}: psnr {image_psnr:.3f} dB "
                    f"({time.time() - t0:.2f}s, running mean {np.mean(psnrs):.3f} dB)"
                )
    return float(np.mean(psnrs))


def train_from_config(cfg: Dict[str, Any], run_dir: str, run_id: str = "run") -> Dict[str, Any]:
    """Train one NeRF per `cfg`, checkpointing/logging under `run_dir`.

    Returns a summary dict including the final held-out test-set PSNR.
    """
    seed = cfg["reproducibility"]["seed"]
    if seed is None:
        raise ValueError(
            "reproducibility.seed must be set explicitly in the config before training "
            "(configs/base.yaml's own comment, and docs/METHODOLOGY.md §8)."
        )
    set_seed(seed, cfg["reproducibility"].get("deterministic_cudnn", True))

    dataset_cfg = cfg["dataset"]
    if dataset_cfg["type"] != "blender":
        raise NotImplementedError(
            f"dataset.type={dataset_cfg['type']!r} not supported — only 'blender' is ported "
            "(see this module's docstring)."
        )

    images, poses, hwf, i_split = load_blender_data(
        dataset_cfg["path"],
        half_res=dataset_cfg.get("half_res", False),
        testskip=dataset_cfg.get("testskip", 1),
    )
    i_train, i_val, i_test = i_split
    print(f"Loaded blender data: images={images.shape}, hwf={hwf}, datadir={dataset_cfg['path']}")
    print(f"train/val/test counts: {len(i_train)}/{len(i_val)}/{len(i_test)}")

    white_bkgd = cfg["render"].get("white_background", True)
    if white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    os.makedirs(run_dir, exist_ok=True)

    model, model_fine, grad_vars, network_query_fn, use_viewdirs = _build_model(cfg)
    optimizer_cfg = cfg["optimizer"]
    optimizer = torch.optim.Adam(params=grad_vars, lr=optimizer_cfg["lr"], betas=(0.9, 0.999))

    global_step = 0
    ckpts = sorted(f for f in os.listdir(run_dir) if f.endswith(".tar"))
    if ckpts:
        ckpt_path = os.path.join(run_dir, ckpts[-1])
        print(f"Reloading from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        global_step = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None and ckpt.get("network_fine_state_dict") is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    render_kwargs_train = _render_kwargs(cfg, network_query_fn, model, model_fine, use_viewdirs, True)
    render_kwargs_test = _render_kwargs(cfg, network_query_fn, model, model_fine, use_viewdirs, False)

    poses_t = torch.Tensor(poses).to(device)

    training_cfg = cfg["training"]
    logging_cfg = cfg["logging"]
    n_rand = training_cfg["batch_size"]
    n_iters = training_cfg["iterations"] + 1
    precrop_iters = training_cfg.get("precrop_iters", 0)
    precrop_frac = training_cfg.get("precrop_frac", 0.5)
    decay_steps = optimizer_cfg["lr_decay_steps"]
    decay_rate = optimizer_cfg["lr_decay_rate"]

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    print(f"Begin training: {n_iters - 1} iterations, seed={seed}, device={device}")
    start = global_step + 1
    last_loss = None
    last_psnr = None
    t0 = time.time()
    for i in trange(start, n_iters, desc=run_id):
        img_i = np.random.choice(i_train)
        target = torch.Tensor(images[img_i]).to(device)
        pose = poses_t[img_i, :3, :4]

        rays_o, rays_d = get_rays(H, W, K, pose)

        if i < precrop_iters:
            dH = int(H // 2 * precrop_frac)
            dW = int(W // 2 * precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH, device=device),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW, device=device),
                ),
                -1,
            )
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, H - 1, H, device=device), torch.linspace(0, W - 1, W, device=device)
                ),
                -1,
            )

        coords = torch.reshape(coords, [-1, 2])
        select_inds = np.random.choice(coords.shape[0], size=[n_rand], replace=False)
        select_coords = coords[select_inds].long()
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]

        rgb, disp, acc, extras = render(
            H, W, K, chunk=training_cfg["chunk_size"], rays=batch_rays, retraw=True, **render_kwargs_train
        )

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0

        loss.backward()
        optimizer.step()

        new_lrate = optimizer_cfg["lr"] * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        if i % logging_cfg["checkpoint_every"] == 0:
            path = os.path.join(run_dir, f"{i:06d}.tar")
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": model.state_dict(),
                    "network_fine_state_dict": model_fine.state_dict() if model_fine is not None else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )

        if i % logging_cfg["log_every"] == 0:
            last_loss, last_psnr = loss.item(), psnr.item()
            writer.add_scalar("train/loss", last_loss, i)
            writer.add_scalar("train/psnr", last_psnr, i)
            tqdm.write(f"[{run_id}] iter {i} loss {last_loss:.5f} psnr {last_psnr:.3f}")

        if i % logging_cfg["eval_every"] == 0 and i > 0:
            val_psnr = evaluate_psnr(images, poses, i_val[:5], hwf, K, training_cfg["chunk_size"], render_kwargs_test)
            writer.add_scalar("val/psnr_subset", val_psnr, i)
            print(f"[{run_id}] iter {i} val-subset PSNR {val_psnr:.3f} dB")

        global_step += 1

    elapsed = time.time() - t0
    print(f"Training loop finished in {elapsed / 3600:.2f} h")

    test_psnr = evaluate_psnr(images, poses, i_test, hwf, K, training_cfg["chunk_size"], render_kwargs_test)
    writer.add_scalar("test/psnr_final", test_psnr, n_iters - 1)
    writer.close()

    final_ckpt_path = os.path.join(run_dir, f"{n_iters - 1:06d}.tar")
    torch.save(
        {
            "global_step": global_step,
            "network_fn_state_dict": model.state_dict(),
            "network_fine_state_dict": model_fine.state_dict() if model_fine is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_ckpt_path,
    )

    return {
        "final_train_loss": last_loss,
        "final_train_psnr": last_psnr,
        "test_psnr": test_psnr,
        "n_test_views": len(i_test),
        "elapsed_hours": elapsed / 3600,
        "final_checkpoint": final_ckpt_path,
        "camera_convention": (
            "Standard NeRF-synthetic Blender convention: right-handed world "
            "coordinates, per-frame transform_matrix is a 4x4 camera-to-world "
            "matrix, camera looks down local -Z with +Y up / +X right "
            "(OpenGL/Blender camera convention). See src/nerf/datasets/blender.py."
        ),
    }
