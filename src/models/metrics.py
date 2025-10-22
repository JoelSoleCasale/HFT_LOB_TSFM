"""
Modular metrics logging system for model training using WandB.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import wandb


class MetricsLogger:
    """WandB-based metrics logger."""

    def __init__(
        self, project: str, entity: Optional[str] = None, tags: Optional[List[str]] = None
    ):
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure WandB is initialized."""
        if not self._initialized:
            wandb.init(project=self.project, entity=self.entity, tags=self.tags, reinit=True)
            self._initialized = True

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log numerical metrics to WandB."""
        self._ensure_initialized()
        if step is not None:
            metrics["step"] = step
        wandb.log(metrics)

    def log_image(self, name: str, image: Any, step: Optional[int] = None) -> None:
        """Log an image to WandB."""
        self._ensure_initialized()
        if isinstance(image, plt.Figure):
            wandb.log({name: wandb.Image(image)}, step=step)
        else:
            wandb.log({name: wandb.Image(image)}, step=step)

    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None) -> None:
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
        confusion_matrix: np.ndarray, class_names: Optional[List[str]] = None
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
    def plot_learning_curves(history: Dict[str, List[float]]) -> plt.Figure:
        """Create learning curves plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss curves
        if "train_loss" in history and "val_loss" in history:
            ax1.plot(history["train_loss"], label="Training Loss", marker="o")
            ax1.plot(history["val_loss"], label="Validation Loss", marker="s")
            ax1.set_title("Loss Curves")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

        # Plot accuracy curves
        if "train_accuracy" in history and "val_accuracy" in history:
            ax2.plot(history["train_accuracy"], label="Training Accuracy", marker="o")
            ax2.plot(history["val_accuracy"], label="Validation Accuracy", marker="s")
            ax2.set_title("Accuracy Curves")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        return fig

    @staticmethod
    def calculate_classification_metrics(
        predictions: torch.Tensor, true_labels: torch.Tensor
    ) -> Dict[str, float]:
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


def create_metrics_logger(logging_config) -> Optional[MetricsLogger]:
    """Create a metrics logger based on logging configuration."""
    if logging_config.wandb_enabled:
        return MetricsLogger(
            project=logging_config.wandb_project,
            entity=logging_config.wandb_entity,
            tags=logging_config.wandb_tags,
        )
    return None
