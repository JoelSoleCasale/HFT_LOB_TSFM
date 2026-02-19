"""
Callback system for training loop.
"""

from pathlib import Path
from loguru import logger
import numpy as np
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
        lambda_value: float = 1.0,
        theta_values: list[float] | None = None,
    ):
        """
        Initialize WandB metrics logger callback.

        Args:
            log_config: Logging configuration
            train_config: Training configuration dictionary
            log_frequency: Log metrics every N epochs
            num_classes: Number of classes for classification
            class_names: Optional list of class names
            lambda_value: Horizontal barrier distance for expected return calculation
            theta_values: List of commission rates for expected return calculation
        """
        self.log_config = log_config
        self.train_config = train_config
        self.log_frequency = log_frequency
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.lambda_value = lambda_value
        self.theta_values = theta_values or [0.0]

        # Create WandB logger
        self.wandb_logger = WandbLogger(log_config=log_config, train_config=train_config)

        # Create metrics suite
        self.metrics_suite = create_trading_suite(
            num_classes=num_classes,
            class_names=self.class_names,
            trade_threshold=0.5,
            include_per_class=False,
        )

        # Create expected return calculators for each theta
        from .metrics.calculators import ExpectedReturnCalculator, MaxThetaCalculator
        import numpy as np

        self.expected_return_calculators = [
            ExpectedReturnCalculator(
                lambda_values=np.array([lambda_value]),
                theta_values=np.array([theta]),
                aggregate="mean",
            )
            for theta in self.theta_values
        ]

        # Create max theta calculator
        self.max_theta_calculator = MaxThetaCalculator(lambda_value=lambda_value)

        # Create expected return vs threshold plotter
        from .metrics.plotters import ExpectedReturnVsThresholdPlotter

        self.expected_return_plotter = ExpectedReturnVsThresholdPlotter(
            lambda_value=lambda_value,
            theta_values=theta_values,
            num_points=50,
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

            # Compute expected returns for training and validation
            try:
                for calc in self.expected_return_calculators:
                    # Training expected return
                    train_er_results = calc.calculate(train_predictions, train_targets)
                    for result in train_er_results:
                        theta = result.metadata["theta"]
                        train_metrics[f"train_expected_return_theta_{theta:.4f}"] = result.value

                    # Validation expected return
                    val_er_results = calc.calculate(val_predictions, val_targets)
                    for result in val_er_results:
                        theta = result.metadata["theta"]
                        val_metrics[f"val_expected_return_theta_{theta:.4f}"] = result.value

                    # Binarize predictions for expected return calculation
                    train_er_results_bin = calc.calculate(
                        train_predictions, train_targets, binarize=True
                    )
                    for result in train_er_results_bin:
                        theta = result.metadata["theta"]
                        metric_name = f"train_expected_return_theta_{theta:.4f}_binarized"
                        train_metrics[metric_name] = result.value

                    val_er_results_bin = calc.calculate(
                        val_predictions, val_targets, binarize=True
                    )
                    for result in val_er_results_bin:
                        theta = result.metadata["theta"]
                        metric_name = f"val_expected_return_theta_{theta:.4f}_binarized"
                        val_metrics[metric_name] = result.value

            except Exception as e:
                logger.warning(f"Failed to compute expected returns: {e}")

            # Compute max theta for non-negative expected return
            try:
                for binarized in [False, True]:
                    suffix = "_binarized" if binarized else ""
                    train_max_theta = self.max_theta_calculator.calculate(
                        train_predictions, train_targets, binarized=binarized
                    )
                    train_metrics[f"train_max_theta{suffix}"] = train_max_theta.value

                    val_max_theta = self.max_theta_calculator.calculate(
                        val_predictions, val_targets, binarized=binarized
                    )
                    val_metrics[f"val_max_theta{suffix}"] = val_max_theta.value
            except Exception as e:
                logger.warning(f"Failed to compute max theta: {e}")

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
            new_test_metrics = {}
            for key, value in test_metrics.items():
                new_test_metrics[f"test_{key}"] = value
            self.wandb_logger.log_metrics(new_test_metrics)
            logger.info(f"Final test metrics: {new_test_metrics}")
        except Exception as e:
            logger.warning(f"Failed to compute final metrics: {e}")

        # Compute expected returns on test set for each theta
        try:
            test_er_metrics = {}
            for calc in self.expected_return_calculators:
                test_er_results = calc.calculate(y_pred, y_true)
                for result in test_er_results:
                    theta = result.metadata["theta"]
                    metric_name = f"test_expected_return_theta_{theta:.4f}"
                    test_er_metrics[metric_name] = result.value

                # Binarized expected return
                test_er_results_bin = calc.calculate(y_pred, y_true, binarize=True)
                for result in test_er_results_bin:
                    theta = result.metadata["theta"]
                    metric_name = f"test_expected_return_theta_{theta:.4f}_binarized"
                    test_er_metrics[metric_name] = result.value

            # Compute max theta on test set
            for binarized in [False, True]:
                suffix = "_binarized" if binarized else ""
                test_max_theta = self.max_theta_calculator.calculate(
                    y_pred, y_true, binarized=binarized
                )
                test_er_metrics[f"test_max_theta{suffix}"] = test_max_theta.value

            self.wandb_logger.log_metrics(test_er_metrics)
            logger.info(f"Test expected returns: {test_er_metrics}")
        except Exception as e:
            logger.warning(f"Failed to compute test expected returns: {e}")

        # Generate and log all plots
        try:
            # Get training history for loss landscape plot
            history = {}
            for callback in trainer.callbacks:
                if hasattr(callback, "get_history"):
                    history = callback.get_history()
                    break

            plots = self.metrics_suite.generate_all_plots(y_pred, y_true, history=history)

            # Add expected return vs threshold plot
            er_plot = self.expected_return_plotter.plot(y_pred, y_true)
            plots["expected_return_vs_threshold"] = er_plot

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
        self.history["test_predictions"] = test_predictions[0].cpu().numpy()
        self.history["test_targets"] = test_predictions[1].cpu().numpy()

    def get_history(self) -> dict[str, list[float]]:
        """Get the training history."""
        return self.history


class MLPSequenceWeightsLogger(Callback):
    """Callback to log MLP sequence weights after training."""

    def __init__(self, log_to_wandb: bool = True):
        """
        Initialize MLP sequence weights logger.

        Args:
            log_to_wandb: Whether to log to WandB (if enabled)
        """
        self.log_to_wandb = log_to_wandb

    def on_train_end(
        self, trainer: Any, test_predictions: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Log MLP sequence weights after training completes."""
        # Check if model is MLP
        from models.architectures.mlp import MLPTimeSeriesModel

        if not isinstance(trainer.model, MLPTimeSeriesModel):
            logger.debug("Model is not MLP, skipping sequence weights logging")
            return

        # Get the sequence weights
        sequence_weights = trainer.model.sequence_weights.weight.data.cpu().numpy()

        # Log weights to console
        logger.info("=" * 80)
        logger.info("MLP Sequence Weights Analysis")
        logger.info("=" * 80)
        logger.info(f"Weight shape: {sequence_weights.shape}")
        logger.info("Weight statistics:")
        logger.info(f"  Mean: {sequence_weights.mean():.6f}")
        logger.info(f"  Std: {sequence_weights.std():.6f}")
        logger.info(f"  Min: {sequence_weights.min():.6f}")
        logger.info(f"  Max: {sequence_weights.max():.6f}")
        logger.info(f"\nFull weight values (sequence_length={sequence_weights.shape[1]}):")

        # Log all weights
        weights_flat = sequence_weights.flatten()
        for i, weight in enumerate(weights_flat):
            logger.info(f"  Position {i:3d}: {weight:.6f}")

        # Identify most important timesteps
        abs_weights = np.abs(weights_flat)
        top_k = min(10, len(weights_flat))
        top_indices = np.argsort(abs_weights)[-top_k:][::-1]

        logger.info(f"\nTop {top_k} most important timesteps (by absolute weight):")
        for rank, idx in enumerate(top_indices, 1):
            logger.info(
                f"  {rank}. Position {idx:3d}: {weights_flat[idx]:.6f} (abs: {abs_weights[idx]:.6f})"
            )

        # Log to WandB if enabled
        if self.log_to_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    # Log summary statistics
                    wandb.log(
                        {
                            "mlp_sequence_weights/mean": float(sequence_weights.mean()),
                            "mlp_sequence_weights/std": float(sequence_weights.std()),
                            "mlp_sequence_weights/min": float(sequence_weights.min()),
                            "mlp_sequence_weights/max": float(sequence_weights.max()),
                        }
                    )

                    # Create visualization
                    import matplotlib.pyplot as plt

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                    # Plot all weights
                    ax1.plot(weights_flat, marker="o", linestyle="-", markersize=3)
                    ax1.set_xlabel("Sequence Position")
                    ax1.set_ylabel("Weight Value")
                    ax1.set_title("MLP Sequence Weights (Linear Combination Across Time)")
                    ax1.grid(True, alpha=0.3)
                    ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)

                    # Plot absolute weights
                    ax2.bar(range(len(abs_weights)), abs_weights, alpha=0.7)
                    ax2.set_xlabel("Sequence Position")
                    ax2.set_ylabel("Absolute Weight Value")
                    ax2.set_title("MLP Sequence Weights (Absolute Values)")
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()

                    # Log to WandB
                    wandb.log({"mlp_sequence_weights_plot": wandb.Image(fig)})
                    plt.close(fig)

                    # Log weights as table for detailed analysis
                    weights_table = wandb.Table(
                        columns=["position", "weight", "abs_weight"],
                        data=[[i, float(w), float(abs(w))] for i, w in enumerate(weights_flat)],
                    )
                    wandb.log({"mlp_sequence_weights_table": weights_table})

                    logger.info("Logged MLP sequence weights to WandB")
            except Exception as e:
                logger.warning(f"Failed to log MLP sequence weights to WandB: {e}")

        logger.info("=" * 80)


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
