"""
Utility functions for metrics calculation.

This module provides lightweight utility functions for computing basic metrics
during training. For comprehensive metric evaluation, use the MetricSuite system.
"""

import torch


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


def calculate_trade_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Compute trade accuracy, considering only non-neutral classes with confidence above threshold.

    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        threshold: Minimum confidence threshold for predictions

    Returns:
        Trade accuracy as a float between 0 and 1
    """
    if predictions.dim() > 1:
        # Get probabilities and predicted classes
        probabilities = torch.softmax(predictions, dim=1)
        max_probs, pred_classes = torch.max(probabilities, dim=1)
    else:
        pred_classes = predictions
        max_probs = torch.ones_like(predictions)  # Assume full confidence if already class indices

    # Create mask for high-confidence predictions
    confidence_mask = max_probs >= threshold

    # Create mask for non-neutral classes (assuming classes are -1, 0, 1 encoded as 0, 1, 2)
    trade_mask = ((targets == 0) | (targets == 2)) & ((pred_classes == 0) | (pred_classes == 2))

    # Combine both masks
    combined_mask = trade_mask & confidence_mask

    if combined_mask.sum().item() == 0:
        return float("nan")  # No trade samples to evaluate

    trade_correct = ((pred_classes == targets) & combined_mask).float().sum()
    trade_total = combined_mask.float().sum()

    return (trade_correct / trade_total).item()
