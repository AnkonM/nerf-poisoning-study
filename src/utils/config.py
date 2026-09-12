"""Config loading: resolves a scene/condition YAML against its `defaults`.

Per docs/PROJECT_STRUCTURE.md, `configs/` is the only place hyperparameters
live, and scripts take a single config path as their only required
argument. Config files declare a `defaults:` list of other config paths
(relative to the file itself) to deep-merge underneath their own keys —
e.g. configs/poisoning/budget_20.yaml's `defaults: [../base.yaml,
../scenes/final_scene.yaml]`. Later entries in `defaults` override earlier
ones, and the file's own top-level keys override everything in `defaults`.
"""

import copy
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` onto `base`, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str) -> Dict[str, Any]:
    """Load a config file, resolving its `defaults:` chain (if any)."""
    path = os.path.abspath(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.pop("defaults", [])
    resolved: Dict[str, Any] = {}
    base_dir = os.path.dirname(path)
    for rel in defaults:
        resolved = _deep_merge(resolved, load_config(os.path.join(base_dir, rel)))

    resolved = _deep_merge(resolved, raw)
    return resolved
