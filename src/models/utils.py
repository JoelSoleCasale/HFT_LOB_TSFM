"""
Utility functions for model training and inference.
"""

import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for PyTorch operations.

    Args:
        device: Device specification ("auto", "cpu", "cuda")

    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(device)


def save_model(
    model: nn.Module,
    save_path: str,
    config: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    """
    Save a trained model with optional configuration and metadata.

    Args:
        model: The PyTorch model to save
        save_path: Path where to save the model
        config: Model configuration dictionary
        metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare save data
    save_data = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }

    if config is not None:
        save_data["config"] = config

    if metadata is not None:
        save_data["metadata"] = metadata

    # Save the model
    torch.save(save_data, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(
    load_path: str, model_class: nn.Module, device: torch.device | None = None
) -> tuple[nn.Module, dict[str, object] | None, dict[str, object] | None]:
    """
    Load a trained model from file.

    Args:
        load_path: Path to the saved model
        model_class: The model class to instantiate
        device: Device to load the model on

    Returns:
        Tuple of (model, config, metadata)
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    # Load the saved data
    save_data = torch.load(load_path, map_location=device)

    # Create model instance
    model = model_class
    model.load_state_dict(save_data["model_state_dict"])

    if device is not None:
        model = model.to(device)

    config = save_data.get("config")
    metadata = save_data.get("metadata")

    logger.info(f"Model loaded from {load_path}")
    return model, config, metadata


def set_seed(seed: int = None) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    if seed is None:
        logger.info("No seed provided, skipping seed setting.")
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
