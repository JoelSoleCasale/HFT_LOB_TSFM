from typing import Any, Dict, List, Iterator, Union
import yaml
import itertools
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _is_cross_product_list(value: Any) -> bool:
    """
    Check if a value should be treated as a list of options for cross product.
    Per requirements, actual lists in YAML trigger cross product.
    String representations of lists (e.g. "64,32") are treated as single values.
    """
    return isinstance(value, list)


def _flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config to support dot notation for easier cross product handling."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(_flatten_config(value, prefix=f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _unflatten_config(flat_config: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct nested config from flattened dictionary."""
    config = {}
    for key, value in flat_config.items():
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return config


def generate_config_variants(base_config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Generate all combinations of configuration options.
    Lists in the YAML are treated as options for Grid Search.
    """
    flat_config = _flatten_config(base_config)

    # Identify keys that have lists of options
    keys_with_options = [k for k, v in flat_config.items() if _is_cross_product_list(v)]
    fixed_keys = [k for k in flat_config.items() if not _is_cross_product_list(k[1])]

    if not keys_with_options:
        yield base_config
        return

    # Create cartesian product of options
    options = [flat_config[k] for k in keys_with_options]
    for combination in itertools.product(*options):
        # Start with fixed keys
        variant_flat = dict(fixed_keys)
        # Add the varied keys
        for k, v in zip(keys_with_options, combination):
            variant_flat[k] = v

        yield _unflatten_config(variant_flat)


def parse_list_arg(arg: Union[str, int, List], cast: type = int) -> List:
    """Parse a string like "128,64" into a list; wraps single values; handles existing lists."""
    if isinstance(arg, list):
        return [cast(x) for x in arg]
    if isinstance(arg, (int, float)):
        return [cast(arg)]
    if isinstance(arg, str):
        if "," in arg:
            return [cast(x.strip()) for x in arg.split(",")]
        if arg.strip() == "":
            return []
        return [cast(arg)]
    return []


def update_recursive(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *update* into *base*, returning *base* (mutated in-place)."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            update_recursive(base[k], v)
        else:
            base[k] = v
    return base


@dataclass
class ExperimentConfig:
    global_config: Dict[str, Any]
    model_config: Dict[str, Any]

    @property
    def merged(self) -> Dict[str, Any]:
        """Deep merge global and model config, model config takes precedence."""
        return update_recursive(deepcopy(self.global_config), self.model_config)


def load_and_generate_experiments(
    global_config_path: str, model_config_path: str
) -> List[Dict[str, Any]]:
    global_conf = load_yaml(global_config_path)
    model_conf = load_yaml(model_config_path)

    global_variants = list(generate_config_variants(global_conf))
    model_variants = list(generate_config_variants(model_conf))

    return [
        update_recursive(deepcopy(g_var), m_var)
        for g_var in global_variants
        for m_var in model_variants
    ]
