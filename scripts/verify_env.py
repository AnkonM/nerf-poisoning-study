"""GPU/environment verification check — ROADMAP.md Phase 0 gate.

Confirms the environment actually uses the GPU for compute rather than
silently falling back to CPU or an unsupported kernel path, per
environment/SETUP.md §1.4. Run this before writing any research code.

Expected on the reference dev machine (RTX 5060 laptop, Blackwell): compute
capability (12, 0), matrix multiply PASSED. Do not treat a different result
as a rounding error — flag it as a discrepancy (see environment/SETUP.md
and docs/ROADMAP.md Phase 0 gate) rather than silently working around it.
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

    if compute_capability != (12, 0):
        print(
            f"WARNING: compute capability {compute_capability} != expected (12, 0) "
            "for the RTX 5060 laptop (Blackwell/sm_120) reference machine. "
            "This is a discrepancy from environment/SETUP.md — do not silently "
            "proceed, investigate before trusting subsequent GPU runs."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
