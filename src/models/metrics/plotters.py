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
    """Plot trade accuracy across confidence thresholds (both normal and strict modes)."""

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
        normal_accuracies = []
        strict_accuracies = []
        num_trades = []
        valid_thresholds = []

        for threshold in thresholds:
            normal_acc = calculate_trade_accuracy(
                predictions, targets, float(threshold), strict=False
            )
            strict_acc = calculate_trade_accuracy(
                predictions, targets, float(threshold), strict=True
            )

            # Calculate number of trades (predictions with class 0 or 2 and confidence >= threshold)
            if predictions.dim() > 1:
                probabilities = torch.softmax(predictions, dim=1)
                max_probs, pred_classes = torch.max(probabilities, dim=1)
            else:
                pred_classes = predictions
                max_probs = torch.ones_like(predictions)

            confidence_mask = max_probs >= threshold
            trade_mask = (pred_classes == 0) | (pred_classes == 2)
            n_trades = (trade_mask & confidence_mask).sum().item()

            # Only include threshold if at least one metric is valid
            if not np.isnan(normal_acc) or not np.isnan(strict_acc):
                normal_accuracies.append(normal_acc if not np.isnan(normal_acc) else None)
                strict_accuracies.append(strict_acc if not np.isnan(strict_acc) else None)
                num_trades.append(n_trades)
                valid_thresholds.append(threshold)

        fig, ax1 = plt.subplots(figsize=self.figsize)

        # Plot normal trade accuracy
        normal_valid = [
            (t, a) for t, a in zip(valid_thresholds, normal_accuracies) if a is not None
        ]
        if normal_valid:
            normal_t, normal_a = zip(*normal_valid)
            ax1.plot(
                normal_t,
                normal_a,
                linewidth=2,
                marker="o",
                markersize=4,
                label="Trade Accuracy (Normal)",
                color="blue",
            )

        # Plot strict trade accuracy
        strict_valid = [
            (t, a) for t, a in zip(valid_thresholds, strict_accuracies) if a is not None
        ]
        if strict_valid:
            strict_t, strict_a = zip(*strict_valid)
            ax1.plot(
                strict_t,
                strict_a,
                linewidth=2,
                marker="s",
                markersize=4,
                label="Trade Accuracy (Strict)",
                color="orange",
            )

        ax1.set_xlabel("Confidence Threshold", fontsize=12)
        ax1.set_ylabel("Trade Accuracy", fontsize=12, color="black")
        ax1.set_title("Trade Accuracy vs Confidence Threshold", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis="y", labelcolor="black")

        # Create secondary y-axis for number of trades
        ax2 = ax1.twinx()
        ax2.plot(
            valid_thresholds,
            num_trades,
            linewidth=2,
            marker="^",
            markersize=4,
            label="Number of Trades",
            color="green",
            linestyle="--",
        )
        ax2.set_ylabel("Number of Trades (log scale)", fontsize=12, color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_yscale("log")

        # Set y-axis limits for number of trades
        if num_trades:
            max_trades = max(num_trades)
            min_trades = min([t for t in num_trades if t > 0], default=1)
            if max_trades > 0:
                ax2.set_ylim(max(0.5, min_trades * 0.5), max_trades * 2)

        # Add markers for maximum values
        if normal_valid:
            max_normal_acc = max(a for a in normal_accuracies if a is not None)
            max_normal_threshold = valid_thresholds[
                [a for a in normal_accuracies].index(max_normal_acc)
            ]
            ax1.axhline(
                y=max_normal_acc,
                color="blue",
                linestyle="--",
                alpha=0.3,
                linewidth=1,
            )
            ax1.axvline(
                x=max_normal_threshold,
                color="blue",
                linestyle="--",
                alpha=0.3,
                linewidth=1,
            )

        if strict_valid:
            max_strict_acc = max(a for a in strict_accuracies if a is not None)
            max_strict_threshold = valid_thresholds[
                [a for a in strict_accuracies].index(max_strict_acc)
            ]
            ax1.axhline(
                y=max_strict_acc,
                color="orange",
                linestyle="--",
                alpha=0.3,
                linewidth=1,
            )
            ax1.axvline(
                x=max_strict_threshold,
                color="orange",
                linestyle="--",
                alpha=0.3,
                linewidth=1,
            )

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

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
