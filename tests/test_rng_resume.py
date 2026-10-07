"""D-039: RNG state in checkpoints, and resume semantics (cheap, no training)."""

import random

import numpy as np
import pytest
import torch

from nerf.training import _capture_rng_state, _restore_rng_state, _resume_state


def _draw():
    return (random.random(), float(np.random.rand()), int(np.random.choice(1000)), torch.rand(3).tolist())


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def test_rng_round_trip():
    _seed(7)
    for _ in range(5):
        _draw()
    state = _capture_rng_state()
    expected = [_draw() for _ in range(4)]
    _seed(99)                       # scramble everything
    _draw()
    _restore_rng_state(state)
    assert [_draw() for _ in range(4)] == expected


def test_capturing_consumes_no_random_numbers():
    _seed(3)
    plain = [_draw() for _ in range(6)]
    _seed(3)
    with_capture = []
    for _ in range(6):
        _capture_rng_state()
        with_capture.append(_draw())
    assert with_capture == plain


def test_state_survives_torch_save_and_weights_only_load(tmp_path):
    """torch.load defaults to weights_only=True; a raw NumPy array would break every checkpoint load."""
    _seed(1)
    _draw()
    p = tmp_path / "c.tar"
    torch.save({"rng_state": _capture_rng_state(), "iteration": 5}, p)
    loaded = torch.load(p, map_location="cpu")          # default weights_only
    expected = _draw()
    _seed(5)
    _restore_rng_state(loaded["rng_state"])
    assert _draw() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_rng_round_trip():
    torch.manual_seed(1)
    torch.rand(4, device="cuda")
    state = _capture_rng_state()
    expected = torch.rand(5, device="cuda").cpu().tolist()
    torch.manual_seed(2)
    _restore_rng_state(state)
    assert torch.rand(5, device="cuda").cpu().tolist() == expected


def test_new_format_resumes_after_the_checkpointed_iteration():
    state = {"python": random.getstate()}
    ckpt = {"global_step": 149, "iteration": 150, "rng_state": state}
    step, rng, info = _resume_state(ckpt, "000150.tar")
    assert step == 150 and rng is state                 # loop continues at 151, no iteration repeated
    assert info == {"from_checkpoint": "000150.tar", "from_iteration": 150, "rng_restored": True}


def test_old_format_checkpoint_still_loads_with_a_clear_note(capsys):
    ckpt = {"global_step": 149, "network_fn_state_dict": {}, "optimizer_state_dict": {}}   # Phase 2/3/probe layout
    step, rng, info = _resume_state(ckpt, "025000.tar")
    assert step == 149 and rng is None                  # legacy: re-runs iteration global_step+1
    assert info["rng_restored"] is False and info["from_iteration"] == 150
    out = capsys.readouterr().out
    assert "no RNG state" in out and "NOT bit-equivalent" in out
