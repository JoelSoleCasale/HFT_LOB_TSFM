"""
Metric calculator implementations.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
)
from .core import MetricCalculator, MetricResult
from typing import Literal


# A₁ encodes directional returns: classes {0,1,2} represent {-1, 0, +1}
_A1 = np.array(
    [
        [1, 0, -1],  # True: -1, Pred: -1 -> +1, 0 -> 0, +1 -> -1
        [0, 0, 0],  # True:  0, always 0 return
        [-1, 0, 1],  # True: +1, Pred: -1 -> -1, 0 -> 0, +1 -> +1
    ]
)

# A₂ encodes commission: sum of non-neutral (class 0 and 2) predicted probabilities
_A2 = np.array(
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
)


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
    """Calculate trade accuracy for financial predictions (normal mode)."""

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

        accuracy = calculate_trade_accuracy(predictions, targets, self.threshold, strict=False)
        return MetricResult(name=self.name, value=accuracy, metadata={"threshold": self.threshold})


class StrictTradeAccuracyCalculator(MetricCalculator):
    """Calculate strict trade accuracy for financial predictions.

    Only considers samples where the model predicts a trade (non-neutral prediction),
    measuring accuracy on the model's actual trade signals.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__("strict_trade_accuracy")
        self.threshold = threshold

    @property
    def requires_probabilities(self) -> bool:
        return True

    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricResult:
        from .utils import calculate_trade_accuracy

        accuracy = calculate_trade_accuracy(predictions, targets, self.threshold, strict=True)
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


class CohenKappaCalculator(MetricCalculator):
    """Calculate Cohen's kappa coefficient.

    Cohen's kappa measures inter-rater agreement for categorical items,
    accounting for agreement occurring by chance. Useful for imbalanced datasets.
    """

    def __init__(self):
        super().__init__("cohen_kappa")

    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricResult:
        pred_labels = torch.argmax(predictions, dim=1) if predictions.dim() > 1 else predictions
        kappa = cohen_kappa_score(targets.cpu().numpy(), pred_labels.cpu().numpy())
        return MetricResult(name=self.name, value=kappa)


class ExpectedReturnCalculator(MetricCalculator):
    """Calculate expected return for financial predictions.

    Computes expected return based on:
    - Horizontal barrier distance lambda (profit/loss per correct/incorrect prediction)
    - Commission theta (transaction cost proportional to predicted probability of taking a position)

    Uses matrix formulation: y^T A y = λ y^T A₁ ŷ - θ y^T A₂ ŷ
    where:
    - y is the one-hot encoded true label
    - ŷ is the predicted probability distribution
    - A₁ encodes directional returns
    - A₂ encodes commission structure

    Supports vectorized computation over multiple lambda/theta values.
    """

    def __init__(
        self,
        lambda_values: float | list[float] | np.ndarray = 1.0,
        theta_values: float | list[float] | np.ndarray = 0.0,
        aggregate: Literal["mean", "sum"] = "mean",
    ):
        """
        Initialize expected return calculator.

        Args:
            lambda_values: Horizontal barrier distance(s). Can be scalar, list, or array.
            theta_values: Commission rate(s). Can be scalar, list, or array.
            aggregate: How to aggregate returns across samples. Either "mean" or "sum".
        """
        super().__init__("expected_return")
        self.lambda_values = np.atleast_1d(np.asarray(lambda_values))
        self.theta_values = np.atleast_1d(np.asarray(theta_values))
        self.aggregate = aggregate
        self.A1 = _A1
        self.A2 = _A2

    @property
    def requires_probabilities(self) -> bool:
        return True

    def calculate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        are_logits: bool = True,
        binarize: bool = False,
        **kwargs,
    ) -> list[MetricResult]:
        """
        Calculate expected return for all combinations of lambda and theta.

        Args:
            predictions: Model predictions as logits or probabilities (N x 3)
            targets: True labels (N,) with values {0, 1, 2} representing {-1, 0, 1}
            are_logits: Whether predictions are logits (True) or probabilities (False)
            binarize: Whether to binarize predictions (i.e take argmax) before calculation

        Returns:
            List of MetricResult objects, one for each (lambda, theta) combination
        """
        if are_logits and predictions.dim() > 1:
            probs = torch.softmax(predictions, dim=1).cpu().numpy()
        else:
            probs = predictions.cpu().numpy()

        if binarize:
            pred_labels = np.argmax(probs, axis=1)
            probs = np.zeros_like(probs)
            probs[np.arange(len(pred_labels)), pred_labels] = 1.0

        target_np = targets.cpu().numpy()
        N = len(target_np)

        y_onehot = np.zeros((N, 3))
        y_onehot[np.arange(N), target_np] = 1

        directional_returns = np.einsum("ni,ij,nj->n", y_onehot, self.A1, probs)
        commission_probs = np.einsum("ni,ij,nj->n", y_onehot, self.A2, probs)

        results = []
        for lambda_val in self.lambda_values:
            for theta_val in self.theta_values:
                expected_returns = lambda_val * directional_returns - theta_val * commission_probs

                if self.aggregate == "mean":
                    aggregated_return = np.mean(expected_returns)
                elif self.aggregate == "sum":
                    aggregated_return = np.sum(expected_returns)
                else:
                    raise ValueError(
                        f"Invalid aggregate method: {self.aggregate}. Use 'mean' or 'sum'."
                    )

                metric_name = f"expected_return_lambda-{lambda_val:.4f}_theta-{theta_val:.4f}"
                results.append(
                    MetricResult(
                        name=metric_name,
                        value=aggregated_return,
                        metadata={
                            "lambda": lambda_val,
                            "theta": theta_val,
                            "aggregate": self.aggregate,
                        },
                    )
                )

        return results


class MaxThetaCalculator(MetricCalculator):
    """Calculate maximum theta value where expected return remains non-negative.

    Computes the maximum commission rate (theta) at which expected return is zero:
        max_theta = λ * sum(directional_returns) / sum(commission_probs)

    This represents the breakeven commission rate - any theta below this value
    yields positive expected return given the model's predictions.
    """

    def __init__(self, lambda_value: float = 1.0):
        """
        Initialize max theta calculator.

        Args:
            lambda_value: Horizontal barrier distance for expected return calculation.
        """
        super().__init__("max_theta")
        self.lambda_value = lambda_value
        self.A1 = _A1
        self.A2 = _A2

    @property
    def requires_probabilities(self) -> bool:
        return True

    def calculate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        are_logits: bool = True,
        binarized: bool = False,
        **kwargs,
    ) -> MetricResult:
        """
        Calculate maximum theta for non-negative expected return.

        Args:
            predictions: Model predictions as logits or probabilities (N x 3)
            targets: True labels (N,) with values {0, 1, 2} representing {-1, 0, 1}
            are_logits: Whether predictions are logits (True) or probabilities (False)
            binarized: Whether to binarize predictions (i.e take argmax) before calculation

        Returns:
            MetricResult with max_theta value (NaN if total commission_probs is zero)
        """
        if are_logits and predictions.dim() > 1:
            probs = torch.softmax(predictions, dim=1).cpu().numpy()
        else:
            probs = predictions.cpu().numpy()

        if binarized:
            pred_labels = np.argmax(probs, axis=1)
            probs = np.zeros_like(probs)
            probs[np.arange(len(pred_labels)), pred_labels] = 1.0

        target_np = targets.cpu().numpy()
        N = len(target_np)

        y_onehot = np.zeros((N, 3))
        y_onehot[np.arange(N), target_np] = 1

        directional_returns = np.einsum("ni,ij,nj->n", y_onehot, self.A1, probs)
        commission_probs = np.einsum("ni,ij,nj->n", y_onehot, self.A2, probs)

        total_directional_returns = np.sum(directional_returns)
        total_commission_probs = np.sum(commission_probs)

        if total_commission_probs == 0:
            max_theta = np.nan
        else:
            max_theta = self.lambda_value * total_directional_returns / total_commission_probs

        return MetricResult(
            name=self.name,
            value=max_theta,
            metadata={
                "lambda": self.lambda_value,
                "total_directional_returns": total_directional_returns,
                "total_commission_probs": total_commission_probs,
            },
        )
