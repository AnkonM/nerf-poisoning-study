"""Config loading guards (D-032).

`configs/` is this project's single source of truth (CLAUDE.md), so a config
that silently loses content is a correctness failure, not a style one.
"""

import pytest
import yaml

from utils.config import load_config


def test_duplicate_top_level_key_raises(tmp_path):
    """PyYAML silently keeps only the LAST duplicate, dropping the first.

    This is not hypothetical: while adding a training block to
    configs/scenes/final_scene.yaml, a second top-level `render:` key deleted
    the entire Blender render section from the resolved config — including
    `background_plate.mask_dilation_px`, the 3 px value locked by D-024 and
    referenced by METHODOLOGY.md §3. Nothing raised.
    """
    p = tmp_path / "dup.yaml"
    p.write_text("render:\n  samples: 128\n  mask_dilation_px: 3\nrender:\n  perturb: 1.0\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'render'"):
        load_config(str(p))


def test_duplicate_nested_key_raises(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("training:\n  batch_size: 1024\n  batch_size: 4096\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'batch_size'"):
        load_config(str(p))


def test_stock_safe_load_would_have_accepted_it_silently(tmp_path):
    """Pins WHY the custom loader exists, so it is not 'simplified' away later."""
    text = "render:\n  samples: 128\nrender:\n  perturb: 1.0\n"
    assert yaml.safe_load(text) == {"render": {"perturb": 1.0}}   # samples: GONE


def test_every_real_config_loads(request):
    """All committed configs must survive the duplicate-key guard."""
    import glob
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    files = (glob.glob(os.path.join(root, "configs", "*.yaml"))
             + glob.glob(os.path.join(root, "configs", "*", "*.yaml")))
    assert len(files) >= 14
    for f in files:
        assert load_config(f), f


def test_condition_configs_never_inherit_a_literal_dataset_path():
    """The D-032 regression: an inherited dataset.path pointed every condition
    at the CLEAN scene, so `train.py <condition>` trained the control."""
    import glob
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    for f in sorted(glob.glob(os.path.join(root, "configs", "poisoning", "*.yaml"))):
        if "phase3_poc" in os.path.basename(f):
            continue        # Phase 3 PoC configs predate this scheme (D-020)
        cfg = load_config(f)
        assert cfg["dataset"]["path"] is None, f
        assert cfg["dataset"]["path_template"], f


def test_final_scene_keeps_both_its_blender_and_nerf_render_settings():
    """Guards the exact key that the duplicate-`render:` collision deleted."""
    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = load_config(os.path.join(root, "configs", "poisoning", "budget_20.yaml"))
    r = cfg["render"]
    assert r["background_plate"]["mask_dilation_px"] == 3      # D-024
    assert r["samples"] == 128                                  # Blender
    assert r["num_coarse_samples"] == 64                        # NeRF
    assert cfg["training"]["netchunk"] == 65536                 # was absent
    assert cfg["training"]["batch_size"] == 1024                # was 4096
