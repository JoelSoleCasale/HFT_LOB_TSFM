"""
Modular metrics logging system for model training using WandB.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from .utils import calculate_trade_accuracy
from models.config import LoggingConfig


class MetricsLogger:
    """WandB-based metrics logger."""

    def __init__(
        self,
        log_config: LoggingConfig,
        train_config: dict = {},
    ) -> None:
        self.log_config = log_config
        self.train_config = train_config
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure WandB is initialized."""
        if not self._initialized:
            wandb.init(
                project=self.log_config.project_name,
                entity=self.log_config.wandb_entity,
                tags=self.log_config.wandb_tags,
                name=self.log_config.experiment_name,
                config=self.train_config,
                reinit=True,
            )
            self._initialized = True

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numerical metrics to WandB."""
        self._ensure_initialized()
        if step is not None:
            metrics["step"] = step
        wandb.log(metrics)

    def log_image(self, name: str, image: object, step: int | None = None) -> None:
        """Log an image to WandB."""
        self._ensure_initialized()
        wandb.log({name: wandb.Image(image)}, step=step)

    def log_histogram(self, name: str, values: np.ndarray, step: int | None = None) -> None:
        """Log a histogram to WandB."""
        self._ensure_initialized()
        wandb.log({name: wandb.Histogram(values)}, step=step)

    def close(self) -> None:
        """Close WandB run."""
        if self._initialized:
            wandb.finish()
            self._initialized = False


class MetricsCalculator:
    """Calculator for various model metrics."""

    @staticmethod
    def calculate_confusion_matrix(
        predictions: torch.Tensor, true_labels: torch.Tensor
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        from sklearn.metrics import confusion_matrix

        # Convert predictions to class labels
        if predictions.dim() > 1:
            pred_labels = torch.argmax(predictions, dim=1)
        else:
            pred_labels = predictions

        # Convert to numpy
        pred_np = pred_labels.cpu().numpy()
        true_np = true_labels.cpu().numpy()

        return confusion_matrix(true_np, pred_np)

    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray, class_names: list[str] | None = None
    ) -> plt.Figure:
        """Create a confusion matrix plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names or range(confusion_matrix.shape[0]),
            yticklabels=class_names or range(confusion_matrix.shape[0]),
            cbar=True,
            ax=ax,
        )

        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_trade_accuracy_vs_threshold(
        predictions: torch.Tensor,
        true_labels: torch.Tensor,
        thresholds: np.ndarray | None = None,
        num_points: int = 50,
    ) -> plt.Figure:
        """
        Plot trade accuracy across different confidence thresholds.

        Args:
            predictions: Model predictions (logits or probabilities)
            true_labels: Ground truth labels
            thresholds: Array of threshold values to test. If None, uses linspace from 0 to 1
            num_points: Number of threshold points to test (only used if thresholds is None)

        Returns:
            matplotlib Figure object
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, num_points)

        accuracies = []
        valid_thresholds = []

        for threshold in thresholds:
            accuracy = calculate_trade_accuracy(
                predictions, true_labels, threshold=float(threshold)
            )
            # Only include non-NaN values
            if not np.isnan(accuracy):
                accuracies.append(accuracy)
                valid_thresholds.append(threshold)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(valid_thresholds, accuracies, linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Confidence Threshold", fontsize=12)
        ax.set_ylabel("Trade Accuracy", fontsize=12)
        ax.set_title("Trade Accuracy vs Confidence Threshold", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add horizontal line at maximum accuracy
        if accuracies:
            max_acc = max(accuracies)
            max_threshold = valid_thresholds[accuracies.index(max_acc)]
            ax.axhline(
                y=max_acc, color="r", linestyle="--", alpha=0.5, label=f"Max: {max_acc:.4f}"
            )
            ax.axvline(
                x=max_threshold,
                color="g",
                linestyle="--",
                alpha=0.5,
                label=f"Optimal threshold: {max_threshold:.3f}",
            )
            ax.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def calculate_classification_metrics(
        predictions: torch.Tensor, true_labels: torch.Tensor
    ) -> dict[str, float]:
        """Calculate comprehensive classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        # Convert predictions to class labels
        if predictions.dim() > 1:
            pred_labels = torch.argmax(predictions, dim=1)
            pred_probs = torch.softmax(predictions, dim=1)
        else:
            pred_labels = predictions
            pred_probs = predictions

        # Convert to numpy
        pred_np = pred_labels.cpu().numpy()
        true_np = true_labels.cpu().numpy()
        pred_probs_np = pred_probs.cpu().numpy()

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(true_np, pred_np),
            "precision_macro": precision_score(true_np, pred_np, average="macro", zero_division=0),
            "recall_macro": recall_score(true_np, pred_np, average="macro", zero_division=0),
            "f1_macro": f1_score(true_np, pred_np, average="macro", zero_division=0),
        }

        # Add per-class metrics if we have enough classes
        unique_classes = len(np.unique(true_np))
        if unique_classes > 2:
            metrics.update(
                {
                    "precision_weighted": precision_score(
                        true_np, pred_np, average="weighted", zero_division=0
                    ),
                    "recall_weighted": recall_score(
                        true_np, pred_np, average="weighted", zero_division=0
                    ),
                    "f1_weighted": f1_score(true_np, pred_np, average="weighted", zero_division=0),
                }
            )

            # ROC AUC for multi-class
            try:
                metrics["roc_auc_ovr"] = roc_auc_score(true_np, pred_probs_np, multi_class="ovr")
            except ValueError:
                metrics["roc_auc_ovr"] = 0.0

        return metrics
