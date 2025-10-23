"""
Training utilities for deep learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


from models.model import create_model, FinancialTimeSeriesModel
from models.utils import (
    get_device,
    save_model,
    calculate_accuracy,
    calculate_classification_metrics,
    set_seed,
    count_parameters,
)
from models.config import ModelConfig
from models.metrics import MetricsCalculator, create_metrics_logger
from utils import setup_logging


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(
        self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            model: The model being trained

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ModelTrainer:
    """
    Main training class for deep learning models.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the trainer.

        Args:
            config: Model configuration
        """
        self.config = config
        self.device = get_device(config.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        self.scaler = None
        self.metrics_logger = None

        # Set up logging
        setup_logging(config.logging.log_file)

        # Set random seed
        set_seed(config.seed)

        # Initialize metrics logger
        self.metrics_logger = create_metrics_logger(config.logging)

        logger.info(f"ModelTrainer initialized with device: {self.device}")
        logger.info(f"Configuration: {config}")

    def _create_model(self, input_size: int, sequence_length: int) -> FinancialTimeSeriesModel:
        """Create the model based on configuration."""
        arch_config = self.config.get_architecture_config()
        model = create_model(**arch_config.params())
        model = model.to(self.device)

        logger.info(
            f"Created {arch_config.model_type} model with {count_parameters(model)} parameters"
        )
        return model

    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer based on configuration."""
        training_config = self.config.get_training_config()

        if training_config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
        elif training_config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
        elif training_config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                momentum=0.9,
            )
        elif training_config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config.optimizer}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        training_config = self.config.get_training_config()

        if training_config.scheduler is None:
            return None

        if training_config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.num_epochs,
                **training_config.scheduler_params,
            )
        elif training_config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config.scheduler_params.get("step_size", 30),
                gamma=training_config.scheduler_params.get("gamma", 0.1),
            )
        elif training_config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=training_config.scheduler_params.get("patience", 10),
                factor=training_config.scheduler_params.get("factor", 0.5),
            )
        else:
            raise ValueError(f"Unknown scheduler: {training_config.scheduler}")

    def _create_criterion(self) -> nn.Module:
        """Create the loss function based on configuration."""
        training_config = self.config.get_training_config()

        losses = {
            "cross_entropy": nn.CrossEntropyLoss,
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "focal": FocalLoss,
        }

        if training_config.loss_function.lower() not in losses:
            raise ValueError(f"Unknown loss function: {training_config.loss_function}")

        return losses[training_config.loss_function.lower()](**training_config.loss_params)

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (sequences, labels) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            total_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{total_accuracy/(batch_idx+1):.4f}"}
            )

        return {
            "train_loss": total_loss / num_batches,
            "train_accuracy": total_accuracy / num_batches,
        }

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                # Calculate metrics
                total_loss += loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                total_accuracy += accuracy
                num_batches += 1

                # Store predictions and labels for detailed metrics
                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Calculate detailed metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_accuracy / num_batches,
        }

        # Add classification metrics
        if self.config.get_architecture_config().output_size > 1:
            classification_metrics = calculate_classification_metrics(
                all_predictions, all_labels, self.config.get_architecture_config().output_size
            )
            metrics.update(classification_metrics)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        feature_names: Optional[List[str]] = None,
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            feature_names: Names of feature columns
            label_names: Names of label columns

        Returns:
            Dictionary containing training results
        """
        training_config = self.config.get_training_config()
        logging_config = self.config.get_logging_config()

        # Get input dimensions from first batch
        sample_batch = next(iter(train_loader))
        sample_sequence, sample_label = sample_batch
        input_size = sample_sequence.shape[2]
        sequence_length = sample_sequence.shape[1]

        # Create model, optimizer, and criterion
        self.model = self._create_model(input_size, sequence_length)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.early_stopping = EarlyStopping(patience=training_config.early_stopping_patience)

        # Initialize WandB if enabled
        if self.metrics_logger is not None:
            import wandb

            wandb.init(
                project=logging_config.wandb_project,
                entity=logging_config.wandb_entity,
                name=logging_config.experiment_name,
                tags=logging_config.wandb_tags,
                config=self.config.to_dict(),
            )
            wandb.watch(self.model, log="all", log_freq=100)

        # Training loop
        best_val_loss = float("inf")
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        logger.info(f"Starting training for {training_config.num_epochs} epochs")
        logger.info(f"Input size: {input_size}, Sequence length: {sequence_length}")

        start_time = time.time()

        for epoch in range(training_config.num_epochs):
            epoch_start_time = time.time()

            # Train
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._validate_epoch(val_loader)

            # Update history
            training_history["train_loss"].append(train_metrics["train_loss"])
            training_history["val_loss"].append(val_metrics["val_loss"])
            training_history["train_accuracy"].append(train_metrics["train_accuracy"])
            training_history["val_accuracy"].append(val_metrics["val_accuracy"])

            # Log metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1}/{training_config.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # Log to metrics logger
            if (
                self.metrics_logger is not None
                and epoch % logging_config.log_metrics_frequency == 0
            ):
                log_dict = {
                    "epoch": epoch + 1,
                    **train_metrics,
                    **val_metrics,
                    "epoch_time": epoch_time,
                }
                self.metrics_logger.log_metrics(log_dict, step=epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Early stopping check
            if self.early_stopping(val_metrics["val_loss"], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self._save_checkpoint(epoch, val_metrics["val_loss"])

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Test evaluation
        test_metrics = {}
        if test_loader is not None:
            test_metrics = self._validate_epoch(test_loader)
            logger.info(f"Test metrics: {test_metrics}")

            # Compute and log confusion matrix on test data
            predictions, true_labels = self.predict(test_loader)
            confusion_matrix = MetricsCalculator.calculate_confusion_matrix(
                predictions, true_labels
            )
            logger.info(f"Test confusion matrix:\n{confusion_matrix}")

            if self.metrics_logger is not None:
                self.metrics_logger.log_metrics({"test": test_metrics})

                # Log confusion matrix as an image
                if logging_config.log_confusion_matrix:
                    cm_fig = MetricsCalculator.plot_confusion_matrix(confusion_matrix)
                    self.metrics_logger.log_image("test_confusion_matrix", cm_fig)
                    plt.close(cm_fig)

                # Log trade accuracy vs threshold plot
                if logging_config.log_trade_accuracy_vs_threshold:
                    trade_acc_fig = MetricsCalculator.plot_trade_accuracy_vs_threshold(
                        predictions, true_labels
                    )
                    self.metrics_logger.log_image("trade_accuracy_vs_threshold", trade_acc_fig)
                    plt.close(trade_acc_fig)

        # Save final model
        self._save_final_model(feature_names, label_names)

        # Close metrics logger
        if self.metrics_logger is not None:
            self.metrics_logger.close()

        return {
            "training_history": training_history,
            "test_metrics": test_metrics,
            "best_val_loss": best_val_loss,
            "total_training_time": total_time,
        }

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        logging_config = self.config.get_logging_config()
        checkpoint_dir = Path(logging_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"

        save_model(
            self.model,
            str(checkpoint_path),
            config=self.config.to_dict(),
            metadata={
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "model_type": self.config.get_architecture_config().model_type,
            },
        )

    def _save_final_model(
        self, feature_names: Optional[List[str]], label_names: Optional[List[str]]
    ):
        """Save the final trained model."""
        logging_config = self.config.get_logging_config()
        model_save_path = Path(logging_config.model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_type": self.config.get_architecture_config().model_type,
            "feature_names": feature_names,
            "label_names": label_names,
            "config": self.config.to_dict(),
        }

        save_model(
            self.model, str(model_save_path), config=self.config.to_dict(), metadata=metadata
        )

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        self.model.eval()
        return self._validate_epoch(test_loader)

    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on data.

        Args:
            data_loader: Data loader for prediction

        Returns:
            Tuple of (predictions, true_labels)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())

        return torch.cat(all_predictions, dim=0), torch.cat(all_labels, dim=0)


# Convenience function for training
def train_model(
    features,
    labels,
    config: ModelConfig,
    feature_columns: Optional[List[str]] = None,
    label_columns: Optional[List[str]] = None,
) -> Tuple[ModelTrainer, Dict[str, Any]]:
    """
    Convenience function to train a model.

    Args:
        features: Polars LazyFrame containing features
        labels: Polars LazyFrame containing labels
        config: Model configuration
        feature_columns: List of feature column names to use
        label_columns: List of label column names to use

    Returns:
        Tuple of (trainer, training_results)
    """
    from .data import prepare_data_for_training

    # Prepare data using the data configuration
    data_config = config.get_data_config()
    train_loader, val_loader, test_loader, scaler, feature_names, label_names = (
        prepare_data_for_training(features, labels, data_config, feature_columns, label_columns)
    )

    # Create trainer
    trainer = ModelTrainer(config)

    # Train model
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_names=feature_names,
        label_names=label_names,
    )

    return trainer, results
