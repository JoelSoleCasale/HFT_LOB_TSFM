"""
Callback system for training loop.
"""

from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from typing import Any

import torch

from .metrics import MetricsCalculator, MetricsLogger
from .config import LoggingConfig
from .utils import save_model
import time


class Callback:
    """Base callback class with empty methods."""

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_train_end(
        self, trainer: Any, test_predictions: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Called at the end of training."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting."""

    def __init__(
        self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True
    ):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in validation loss to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the epoch with the best value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Check if training should stop based on validation loss."""
        val_loss = val_metrics.get("val_loss", float("inf"))

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            if self.restore_best_weights and self.best_weights is not None:
                trainer.model.load_state_dict(self.best_weights)
            trainer.should_stop = True


class ModelCheckpoint(Callback):
    """Callback to save model checkpoints when validation loss improves."""

    def __init__(self, checkpoint_dir: str | Path, save_best_only: bool = True):
        """
        Initialize model checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: If True, only save when validation loss improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_val_loss = float("inf")

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Save model checkpoint if validation loss has improved."""
        val_loss = val_metrics.get("val_loss", float("inf"))

        if not self.save_best_only or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"

            save_model(
                trainer.model,
                str(checkpoint_path),
                config=trainer.config.to_dict(),
                metadata={
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "model_type": trainer.config.get_architecture_config().model_type,
                },
            )

            logger.debug(f"Saved checkpoint to {checkpoint_path}")


class WandbMetricsLogger(Callback):
    """Callback for logging metrics and plots to WandB."""

    def __init__(
        self,
        log_config: LoggingConfig,
        train_config: dict[str, Any],
        log_frequency: int = 1,
    ):
        """
        Initialize WandB metrics logger callback.

        Args:
            log_config: Logging configuration
            train_config: Training configuration dictionary
            log_frequency: Log metrics every N epochs
        """
        self.log_config = log_config
        self.train_config = train_config
        self.log_frequency = log_frequency
        self.metrics_logger = None

        self.metrics_logger = MetricsLogger(log_config=log_config, train_config=train_config)

    def on_train_begin(self, trainer: Any) -> None:
        """Initialize WandB run at the beginning of training."""
        self.metrics_logger._ensure_initialized()

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Log metrics to WandB at the end of each epoch."""
        # Log metrics at specified frequency
        if epoch % self.log_frequency == 0:
            self.metrics_logger.log_metrics(train_metrics, step=epoch)
            self.metrics_logger.log_metrics(val_metrics, step=epoch)

    def on_train_end(
        self, trainer: Any, test_predictions: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Log final plots and close WandB run."""
        y_pred, y_true = test_predictions

        if self.log_config.log_confusion_matrix:
            confusion_matrix = MetricsCalculator.calculate_confusion_matrix(y_pred, y_true)
            cm_fig = MetricsCalculator.plot_confusion_matrix(confusion_matrix)
            self.metrics_logger.log_image("test_confusion_matrix", cm_fig)
            plt.close(cm_fig)

        if self.log_config.log_trade_accuracy_vs_threshold:
            trade_acc_fig = MetricsCalculator.plot_trade_accuracy_vs_threshold(y_pred, y_true)
            self.metrics_logger.log_image("trade_accuracy_vs_threshold", trade_acc_fig)
            plt.close(trade_acc_fig)

        self.metrics_logger.close()


class TrainingHistoryTracker(Callback):
    """Callback to track training history metrics."""

    def __init__(self):
        """Initialize training history tracker."""
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Record training metrics to history."""
        self.history["train_loss"].append(train_metrics.get("train_loss", 0.0))
        self.history["val_loss"].append(val_metrics.get("val_loss", 0.0))
        self.history["train_accuracy"].append(train_metrics.get("train_accuracy", 0.0))
        self.history["val_accuracy"].append(val_metrics.get("val_accuracy", 0.0))

    def on_train_end(
        self, trainer: Any, test_predictions: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Called at the end of training."""
        pass

    def get_history(self) -> dict[str, list[float]]:
        """Get the training history."""
        return self.history


class ConsoleLogger(Callback):
    """Callback to log training progress to console."""

    def __init__(self):
        """Initialize console logger."""
        self.start_time = None

    def on_train_begin(self, trainer: Any) -> None:
        """Log training start."""
        training_config = trainer.config.get_training_config()
        logger.info(f"Starting training for {training_config.num_epochs} epochs")
        self.start_time = time.time()

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Log epoch metrics to console."""
        training_config = trainer.config.get_training_config()
        epoch_time = train_metrics.get("epoch_time", 0.0)

        logger.info(
            f"Epoch {epoch+1}/{training_config.num_epochs} - "
            f"Train Loss: {train_metrics.get('train_loss', 0.0):.4f}, "
            f"Val Loss: {val_metrics.get('val_loss', 0.0):.4f}, "
            f"Train Acc: {train_metrics.get('train_accuracy', 0.0):.4f}, "
            f"Val Acc: {val_metrics.get('val_accuracy', 0.0):.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

    def on_train_end(
        self, trainer: Any, test_predictions: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Log training completion."""
        logger.info(f"Training completed in {time.time() - self.start_time:.2f} seconds")
