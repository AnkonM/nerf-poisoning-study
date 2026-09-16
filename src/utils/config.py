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


class _NoDuplicatesLoader(yaml.SafeLoader):
    """A SafeLoader that REFUSES a mapping containing a duplicate key.

    Stock `yaml.safe_load` accepts duplicates silently and keeps the last one.
    That is a live hazard here, because these configs are deep and sectioned:
    while drafting D-032 a second top-level `render:` block was appended to
    configs/scenes/final_scene.yaml, and it silently deleted the entire Blender
    render section -- resolution, sample count, and
    `background_plate.mask_dilation_px`, the 3 px value locked by D-024 and
    referenced by METHODOLOGY.md §3. Nothing raised; the config simply resolved
    without those keys, and `build_poison_set.py` would have died on a KeyError
    far from the cause (or worse, a future reader would have re-added a
    different value).

    Configs are this project's single source of truth (CLAUDE.md), so a config
    that silently loses half its content is a correctness failure, not a style
    one. This makes that class of loss impossible rather than merely unlikely.
    """


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                "found duplicate key %r -- PyYAML would silently keep only the "
                "last one, dropping everything under the first" % (key,),
                key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicatesLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


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
        raw = yaml.load(f, Loader=_NoDuplicatesLoader) or {}

    defaults = raw.pop("defaults", [])
    resolved: Dict[str, Any] = {}
    base_dir = os.path.dirname(path)
    for rel in defaults:
        resolved = _deep_merge(resolved, load_config(os.path.join(base_dir, rel)))

    resolved = _deep_merge(resolved, raw)
    return resolved
