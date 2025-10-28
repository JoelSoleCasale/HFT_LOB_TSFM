"""
Plot generator implementations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from .core import MetricPlotter


class ConfusionMatrixPlotter(MetricPlotter):
    """Generate confusion matrix heatmap."""

    def __init__(self, class_names: list[str] | None = None, figsize: tuple = (10, 8)):
        super().__init__("confusion_matrix")
        self.class_names = class_names
        self.figsize = figsize

    def plot(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> plt.Figure:
        pred_labels = torch.argmax(predictions, dim=1) if predictions.dim() > 1 else predictions
        cm = confusion_matrix(targets.cpu().numpy(), pred_labels.cpu().numpy())

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names or range(cm.shape[0]),
            yticklabels=self.class_names or range(cm.shape[0]),
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()

        return fig


class TradeAccuracyVsThresholdPlotter(MetricPlotter):
    """Plot trade accuracy across confidence thresholds."""

    def __init__(self, num_points: int = 50, figsize: tuple = (10, 6)):
        super().__init__("trade_accuracy_vs_threshold")
        self.num_points = num_points
        self.figsize = figsize

    @property
    def requires_probabilities(self) -> bool:
        return True

    def plot(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> plt.Figure:
        from .utils import calculate_trade_accuracy

        thresholds = np.linspace(0, 1, self.num_points)
        accuracies = []
        valid_thresholds = []

        for threshold in thresholds:
            acc = calculate_trade_accuracy(predictions, targets, float(threshold))
            if not np.isnan(acc):
                accuracies.append(acc)
                valid_thresholds.append(threshold)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(valid_thresholds, accuracies, linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Confidence Threshold", fontsize=12)
        ax.set_ylabel("Trade Accuracy", fontsize=12)
        ax.set_title("Trade Accuracy vs Confidence Threshold", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if accuracies:
            max_acc = max(accuracies)
            max_threshold = valid_thresholds[accuracies.index(max_acc)]
            ax.axhline(
                y=max_acc,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Max: {max_acc:.4f}",
            )
            ax.axvline(
                x=max_threshold,
                color="g",
                linestyle="--",
                alpha=0.5,
                label=f"Optimal: {max_threshold:.3f}",
            )
            ax.legend()

        plt.tight_layout()
        return fig


class ROCCurvePlotter(MetricPlotter):
    """Plot ROC curves for multi-class classification."""

    def __init__(self, class_names: list[str] | None = None, figsize: tuple = (10, 8)):
        super().__init__("roc_curves")
        self.class_names = class_names
        self.figsize = figsize

    @property
    def requires_probabilities(self) -> bool:
        return True

    def plot(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> plt.Figure:
        # Get probabilities
        probs = torch.softmax(predictions, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Binarize targets for multi-class
        n_classes = probs.shape[1]
        targets_bin = label_binarize(targets_np, classes=range(n_classes))

        # Handle binary classification case
        if n_classes == 2:
            targets_bin = np.hstack([1 - targets_bin, targets_bin])

        fig, ax = plt.subplots(figsize=self.figsize)

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            class_name = self.class_names[i] if self.class_names else f"Class {i}"
            ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig


class PredictionDistributionPlotter(MetricPlotter):
    """Plot distribution of prediction probabilities per class."""

    def __init__(self, class_names: list[str] | None = None, figsize: tuple = (12, 6)):
        super().__init__("prediction_distribution")
        self.class_names = class_names
        self.figsize = figsize

    @property
    def requires_probabilities(self) -> bool:
        return True

    def plot(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> plt.Figure:
        # Get probabilities
        probs = torch.softmax(predictions, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        n_classes = probs.shape[1]

        fig, axes = plt.subplots(1, n_classes, figsize=self.figsize)
        if n_classes == 1:
            axes = [axes]

        for i in range(n_classes):
            class_name = self.class_names[i] if self.class_names else f"Class {i}"

            # Get probabilities for this class for correct and incorrect predictions
            correct_mask = targets_np == i
            correct_probs = probs[correct_mask, i]
            incorrect_probs = probs[~correct_mask, i]

            axes[i].hist(correct_probs, bins=50, alpha=0.5, label="True positives", color="green")
            axes[i].hist(incorrect_probs, bins=50, alpha=0.5, label="False positives", color="red")
            axes[i].set_xlabel("Predicted Probability")
            axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"{class_name}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class LossLandscapePlotter(MetricPlotter):
    """Plot training and validation loss over epochs."""

    def __init__(self, figsize: tuple = (10, 6)):
        super().__init__("loss_landscape")
        self.figsize = figsize

    def plot(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> plt.Figure:
        """
        Plot loss landscape from training history.

        Args:
            predictions: Not used for this plotter
            targets: Not used for this plotter
            **kwargs: Must contain 'history' dict with 'train_loss' and 'val_loss' keys
        """
        history = kwargs.get("history", {})
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        if not train_loss or not val_loss:
            # Return empty figure if no history available
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(
                0.5,
                0.5,
                "No training history available",
                ha="center",
                va="center",
                fontsize=14,
            )
            return fig

        epochs = range(1, len(train_loss) + 1)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
        ax.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig
