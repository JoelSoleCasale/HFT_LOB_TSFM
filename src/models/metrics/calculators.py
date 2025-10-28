"""
Metric calculator implementations.
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from .core import MetricCalculator, MetricResult


class AccuracyCalculator(MetricCalculator):
    """Calculate classification accuracy."""

    def __init__(self):
        super().__init__("accuracy")

    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricResult:
        pred_labels = torch.argmax(predictions, dim=1) if predictions.dim() > 1 else predictions
        accuracy = accuracy_score(targets.cpu().numpy(), pred_labels.cpu().numpy())
        return MetricResult(name=self.name, value=accuracy)


class PrecisionRecallF1Calculator(MetricCalculator):
    """Calculate precision, recall, and F1 for all averaging strategies."""

    def __init__(self, average: str = "macro"):
        super().__init__(f"{average}_metrics")
        self.average = average

    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> list[MetricResult]:
        pred_labels = torch.argmax(predictions, dim=1) if predictions.dim() > 1 else predictions
        pred_np = pred_labels.cpu().numpy()
        target_np = targets.cpu().numpy()

        results = [
            MetricResult(
                name=f"{self.average}_precision",
                value=precision_score(target_np, pred_np, average=self.average, zero_division=0),
            ),
            MetricResult(
                name=f"{self.average}_recall",
                value=recall_score(target_np, pred_np, average=self.average, zero_division=0),
            ),
            MetricResult(
                name=f"{self.average}_f1",
                value=f1_score(target_np, pred_np, average=self.average, zero_division=0),
            ),
        ]
        return results


class TradeAccuracyCalculator(MetricCalculator):
    """Calculate trade accuracy for financial predictions."""

    def __init__(self, threshold: float = 0.5):
        super().__init__("trade_accuracy")
        self.threshold = threshold

    @property
    def requires_probabilities(self) -> bool:
        return True

    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricResult:
        from .utils import calculate_trade_accuracy

        accuracy = calculate_trade_accuracy(predictions, targets, self.threshold)
        return MetricResult(name=self.name, value=accuracy, metadata={"threshold": self.threshold})


class ConfusionMatrixCalculator(MetricCalculator):
    """Calculate confusion matrix."""

    def __init__(self):
        super().__init__("confusion_matrix")

    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricResult:
        pred_labels = torch.argmax(predictions, dim=1) if predictions.dim() > 1 else predictions
        cm = confusion_matrix(targets.cpu().numpy(), pred_labels.cpu().numpy())

        # Return as metadata since it's a matrix
        return MetricResult(name=self.name, value=0.0, metadata={"matrix": cm})


class PerClassMetricsCalculator(MetricCalculator):
    """Calculate per-class precision, recall, F1."""

    def __init__(self, num_classes: int, class_names: list[str] | None = None):
        super().__init__("per_class_metrics")
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> list[MetricResult]:
        pred_labels = torch.argmax(predictions, dim=1) if predictions.dim() > 1 else predictions
        pred_np = pred_labels.cpu().numpy()
        target_np = targets.cpu().numpy()

        results = []
        for i, class_name in enumerate(self.class_names):
            # Binary metrics for each class
            binary_target = (target_np == i).astype(int)
            binary_pred = (pred_np == i).astype(int)

            results.extend(
                [
                    MetricResult(
                        name=f"{class_name}_precision",
                        value=precision_score(binary_target, binary_pred, zero_division=0),
                    ),
                    MetricResult(
                        name=f"{class_name}_recall",
                        value=recall_score(binary_target, binary_pred, zero_division=0),
                    ),
                    MetricResult(
                        name=f"{class_name}_f1",
                        value=f1_score(binary_target, binary_pred, zero_division=0),
                    ),
                ]
            )

        return results


class ROCAUCCalculator(MetricCalculator):
    """Calculate ROC AUC score for multi-class classification."""

    def __init__(self, multi_class: str = "ovr"):
        super().__init__("roc_auc")
        self.multi_class = multi_class

    @property
    def requires_probabilities(self) -> bool:
        return True

    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricResult:
        # Get probabilities
        if predictions.dim() > 1:
            probs = torch.softmax(predictions, dim=1).cpu().numpy()
        else:
            probs = predictions.cpu().numpy()

        target_np = targets.cpu().numpy()

        try:
            roc_auc = roc_auc_score(target_np, probs, multi_class=self.multi_class)
        except (ValueError, IndexError):
            # Handle cases where ROC AUC cannot be computed
            roc_auc = 0.0

        return MetricResult(name=self.name, value=roc_auc)
