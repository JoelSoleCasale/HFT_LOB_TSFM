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


def parse_list_arg(arg: Union[str, int, List], type: type = int) -> List[int]:
    """
    Parse a string argument like "128,64" into a list of integers [128, 64].
    If it's already a list, return it (though this case shouldn't happen for hidden_sizes based on rules).
    If it's a single int/str, wrap in list.
    """
    if isinstance(arg, list):
        return [type(x) for x in arg]
    if isinstance(arg, (int, float)):
        return [type(arg)]
    if isinstance(arg, str):
        if "," in arg:
            return [type(x.strip()) for x in arg.split(",")]
        else:
            if arg.strip() == "":
                return []
            return [type(arg)]
    return []


@dataclass
class ExperimentConfig:
    global_config: Dict[str, Any]
    model_config: Dict[str, Any]
    # merged config is useful, but we keep them separate mostly

    @property
    def merged(self) -> Dict[str, Any]:
        """Deep merge global and model config."""
        # Simple merge, model config overrides global if conflict
        merged = deepcopy(self.global_config)

        def update_recursive(d: dict, u: dict):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_recursive(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        return update_recursive(merged, self.model_config)


def load_and_generate_experiments(
    global_config_path: str, model_config_path: str
) -> List[Dict[str, Any]]:
    global_conf = load_yaml(global_config_path)
    model_conf = load_yaml(model_config_path)

    # Generate variants for each separately?
    # Or merge first then generate?
    # Merging first is better to allow model config to override global arrays if needed,
    # but more complex if keys overlap.
    # Let's generate variants for each and cross product them.

    global_variants = list(generate_config_variants(global_conf))
    model_variants = list(generate_config_variants(model_conf))

    all_experiments = []

    for g_var in global_variants:
        for m_var in model_variants:
            # Merge: Model config takes precedence over global
            merged = deepcopy(g_var)

            # Recursive merge helper
            def update_recursive(base, update):
                for k, v in update.items():
                    if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                        update_recursive(base[k], v)
                    else:
                        base[k] = v

            update_recursive(merged, m_var)
            all_experiments.append(merged)

    return all_experiments
