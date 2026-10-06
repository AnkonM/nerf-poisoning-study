"""GPU/environment verification check — ROADMAP.md Phase 0 gate.

Confirms the environment actually uses the GPU for compute rather than
silently falling back to CPU or an unsupported kernel path, per
environment/SETUP.md §1.4. Run this before writing any research code.

Gate: CUDA available and a GPU matrix multiply succeeds. Compute capability is
reported for information only (the original RTX 5060 laptop was (12, 0); the
Phase 6 HPC machine's Quadro RTX 5000 is (7, 5) — see D-036).
"""

import sys

import torch


def main() -> int:
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if not cuda_available:
        print("Matrix multiply: FAILED (no CUDA device available)")
        return 1

    device_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print("Device:", device_name)
    print("Compute capability:", compute_capability)

    try:
        x = torch.randn(4096, 4096, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"Matrix multiply: FAILED ({e})")
        return 1

    print("Matrix multiply: PASSED")

    # Informational only: the pass/fail gate is CUDA available + GPU matmul.
    # (12, 0) is the original RTX 5060 laptop; other GPUs are fine (D-036).
    if compute_capability != (12, 0):
        print(
            f"NOTE: compute capability {compute_capability} differs from the "
            "original RTX 5060 laptop's (12, 0). Informational only; the gate "
            "is CUDA availability plus the matrix multiply above."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
