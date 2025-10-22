"""
Utility functions for model training and inference.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
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
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
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
    load_path: str, model_class: nn.Module, device: Optional[torch.device] = None
) -> Tuple[nn.Module, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
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


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy for classification tasks.

    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels

    Returns:
        Accuracy as a float between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)

    correct = (predictions == targets).float().sum()
    total = targets.size(0)
    return (correct / total).item()


def calculate_classification_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int, per_class: bool = False
) -> Dict[str, float]:
    """
    Calculate classification metrics including precision, recall, and F1-score.

    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import classification_report

    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)

    # Convert to numpy for sklearn
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    # Calculate metrics
    report = classification_report(target_np, pred_np, output_dict=True, zero_division=0)

    # Calculate overall accuracy
    accuracy = calculate_accuracy(predictions, targets)

    metrics = {
        "accuracy": accuracy,
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "weighted_avg_precision": report["weighted avg"]["precision"],
        "weighted_avg_recall": report["weighted avg"]["recall"],
        "weighted_avg_f1": report["weighted avg"]["f1-score"],
    }

    # Add per-class metrics
    if per_class:
        for i in range(num_classes):
            if str(i) in report:
                metrics[f"class_{i}_precision"] = report[str(i)]["precision"]
                metrics[f"class_{i}_recall"] = report[str(i)]["recall"]
                metrics[f"class_{i}_f1"] = report[str(i)]["f1-score"]

    # Compute trade accuracy, assuming classes are -1, 0, 1 (encoded as 0, 1, 2)
    # Trade accuracy considers only classes 0 and 2 for predicted and true values
    if num_classes == 3:
        trade_mask = ((targets == 0) | (targets == 2)) & ((predictions == 0) | (predictions == 2))
        if trade_mask.sum().item() > 0:
            trade_correct = ((predictions == targets) & trade_mask).float().sum()
            trade_total = trade_mask.float().sum()
            trade_accuracy = (trade_correct / trade_total).item()
            metrics["trade_accuracy"] = trade_accuracy
        else:
            metrics["trade_accuracy"] = float("nan")

    return metrics


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

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
