"""Small training-time metric helpers.

Vendored from yenchenlin/nerf-pytorch's run_nerf_helpers.py (MIT license,
see src/nerf/THIRD_PARTY_LICENSE). These are the training-loop diagnostics
(per-batch loss/PSNR); the study's actual masked/unmasked PSNR/SSIM/LPIPS
metrics (docs/METHODOLOGY.md §6) live in src/metrics/ once Phase 5+ adds
them.
"""

import numpy as np
import torch


def img2mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - y) ** 2)


def mse2psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log(mse) / torch.log(torch.tensor([10.0], device=mse.device))


def to8b(x: np.ndarray) -> np.ndarray:
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)
