"""
Callback system for training loop.
"""

from pathlib import Path
from loguru import logger
import time
import torch
from typing import Any

from .metrics.logger import WandbLogger
from .metrics.presets import create_trading_suite
from .config import LoggingConfig
from .utils import save_model


class Callback:
    """Base callback class with empty methods."""

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_predictions: torch.Tensor,
        train_targets: torch.Tensor,
        train_loss: float,
        train_time: float,
        val_predictions: torch.Tensor,
        val_targets: torch.Tensor,
        val_loss: float,
    ) -> None:
        """
        Called at the end of each epoch.

        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            train_predictions: Training predictions (logits)
            train_targets: Training targets
            train_loss: Average training loss
            train_time: Training epoch time in seconds
            val_predictions: Validation predictions (logits)
            val_targets: Validation targets
            val_loss: Average validation loss
        """
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
        train_predictions: torch.Tensor,
        train_targets: torch.Tensor,
        train_loss: float,
        train_time: float,
        val_predictions: torch.Tensor,
        val_targets: torch.Tensor,
        val_loss: float,
    ) -> None:
        """Check if training should stop based on validation loss."""
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
        train_predictions: torch.Tensor,
        train_targets: torch.Tensor,
        train_loss: float,
        train_time: float,
        val_predictions: torch.Tensor,
        val_targets: torch.Tensor,
        val_loss: float,
    ) -> None:
        """Save model checkpoint if validation loss has improved."""
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
        num_classes: int = 3,
        class_names: list[str] | None = None,
    ):
        """
        Initialize WandB metrics logger callback.

        Args:
            log_config: Logging configuration
            train_config: Training configuration dictionary
            log_frequency: Log metrics every N epochs
            num_classes: Number of classes for classification
            class_names: Optional list of class names
        """
        self.log_config = log_config
        self.train_config = train_config
        self.log_frequency = log_frequency
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # Create WandB logger
        self.wandb_logger = WandbLogger(log_config=log_config, train_config=train_config)

        # Create metrics suite
        self.metrics_suite = create_trading_suite(
            num_classes=num_classes,
            class_names=self.class_names,
            trade_threshold=0.5,
        )

    def on_train_begin(self, trainer: Any) -> None:
        """Initialize WandB run at the beginning of training."""
        self.wandb_logger._ensure_initialized()

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_predictions: torch.Tensor,
        train_targets: torch.Tensor,
        train_loss: float,
        train_time: float,
        val_predictions: torch.Tensor,
        val_targets: torch.Tensor,
        val_loss: float,
    ) -> None:
        """Log metrics to WandB at the end of each epoch."""
        # Log metrics at specified frequency
        if epoch % self.log_frequency == 0:
            from models.metrics import calculate_accuracy

            # Compute training metrics
            train_accuracy = calculate_accuracy(train_predictions, train_targets)
            train_metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "epoch_time": train_time,
            }

            # Compute validation metrics using the metrics suite
            val_accuracy = calculate_accuracy(val_predictions, val_targets)
            val_metrics = {
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

            # Add detailed classification metrics for validation
            try:
                detailed_metrics = self.metrics_suite.compute_all(val_predictions, val_targets)
                for key, value in detailed_metrics.items():
                    val_metrics[f"val_{key}"] = value
            except Exception as e:
                logger.warning(f"Failed to compute detailed metrics: {e}")

            # Combine and log
            combined_metrics = {**train_metrics, **val_metrics}
            self.wandb_logger.log_metrics(combined_metrics, step=epoch)

    def on_train_end(
        self, trainer: Any, test_predictions: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Log final plots and close WandB run."""
        y_pred, y_true = test_predictions

        # Compute all metrics
        try:
            test_metrics = self.metrics_suite.compute_all(y_pred, y_true)
            self.wandb_logger.log_metrics(test_metrics)
            logger.info(f"Final test metrics: {test_metrics}")
        except Exception as e:
            logger.warning(f"Failed to compute final metrics: {e}")

        # Generate and log all plots
        try:
            # Get training history for loss landscape plot
            history = {}
            for callback in trainer.callbacks:
                if hasattr(callback, "get_history"):
                    history = callback.get_history()
                    break

            plots = self.metrics_suite.generate_all_plots(y_pred, y_true, history=history)
            self.wandb_logger.log_plots(plots)
            logger.info(f"Logged {len(plots)} plots to WandB")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

        self.wandb_logger.close()


class TrainingHistoryTracker(Callback):
    """Callback to track training history metrics."""

    def __init__(self):
        """Initialize training history tracker."""
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_trade_accuracy": [],
            "val_trade_accuracy": [],
        }

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_predictions: torch.Tensor,
        train_targets: torch.Tensor,
        train_loss: float,
        train_time: float,
        val_predictions: torch.Tensor,
        val_targets: torch.Tensor,
        val_loss: float,
    ) -> None:
        """Record training metrics to history."""
        from models.metrics import calculate_accuracy, calculate_trade_accuracy

        train_accuracy = calculate_accuracy(train_predictions, train_targets)
        val_accuracy = calculate_accuracy(val_predictions, val_targets)
        train_trade_acc = calculate_trade_accuracy(train_predictions, train_targets)
        val_trade_acc = calculate_trade_accuracy(val_predictions, val_targets)

        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_accuracy"].append(train_accuracy)
        self.history["val_accuracy"].append(val_accuracy)
        self.history["train_trade_accuracy"].append(train_trade_acc)
        self.history["val_trade_accuracy"].append(val_trade_acc)

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
        train_predictions: torch.Tensor,
        train_targets: torch.Tensor,
        train_loss: float,
        train_time: float,
        val_predictions: torch.Tensor,
        val_targets: torch.Tensor,
        val_loss: float,
    ) -> None:
        """Log epoch metrics to console."""
        from models.metrics import calculate_accuracy

        training_config = trainer.config.get_training_config()

        train_accuracy = calculate_accuracy(train_predictions, train_targets)
        val_accuracy = calculate_accuracy(val_predictions, val_targets)

        logger.info(
            f"Epoch {epoch+1}/{training_config.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, "
            f"Time: {train_time:.2f}s"
        )

    def on_train_end(
        self, trainer: Any, test_predictions: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Log training completion."""
        logger.info(f"Training completed in {time.time() - self.start_time:.2f} seconds")
