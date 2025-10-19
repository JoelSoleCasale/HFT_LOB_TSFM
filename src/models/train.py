"""
Training utilities for deep learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from loguru import logger
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import time
from tqdm import tqdm

from .model import create_model, FinancialTimeSeriesModel
from .utils import (
    get_device,
    setup_logging,
    save_model,
    calculate_accuracy,
    calculate_classification_metrics,
    set_seed,
    count_parameters,
)
from .config import ModelConfig


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
        self.criterion = None
        self.early_stopping = None
        self.scaler = None

        # Set up logging
        setup_logging(config.log_file)

        # Set random seed
        set_seed(42)

        logger.info(f"ModelTrainer initialized with device: {self.device}")
        logger.info(f"Configuration: {config}")

    def _create_model(self, input_size: int, sequence_length: int) -> FinancialTimeSeriesModel:
        """Create the model based on configuration."""
        model_kwargs = {
            "input_size": input_size,
            "output_size": self.config.output_size,
            "dropout": self.config.dropout,
        }

        if self.config.model_type == "mlp":
            model_kwargs.update(
                {
                    "sequence_length": sequence_length,
                    "hidden_sizes": [self.config.hidden_size, self.config.hidden_size // 2],
                }
            )
        elif self.config.model_type == "lstm":
            model_kwargs.update(
                {
                    "hidden_size": self.config.hidden_size,
                    "num_layers": self.config.num_layers,
                }
            )
        elif self.config.model_type == "transformer":
            model_kwargs.update(
                {
                    "d_model": self.config.hidden_size,
                    "nhead": 8,
                    "num_layers": self.config.num_layers,
                }
            )

        model = create_model(self.config.model_type, **model_kwargs)
        model = model.to(self.device)

        logger.info(
            f"Created {self.config.model_type} model with {count_parameters(model)} parameters"
        )
        return model

    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_criterion(self) -> nn.Module:
        """Create the loss function."""
        return nn.CrossEntropyLoss()

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
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.4f}"})

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
        if self.config.output_size > 1:
            classification_metrics = calculate_classification_metrics(
                all_predictions, all_labels, self.config.output_size
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
        # Initialize wandb
        if self.config.wandb_enabled:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                tags=self.config.wandb_tags,
                config=self.config.to_dict(),
            )

        # Get input dimensions from first batch
        sample_batch = next(iter(train_loader))
        sample_sequence, sample_label = sample_batch
        input_size = sample_sequence.shape[2]
        sequence_length = sample_sequence.shape[1]

        # Create model, optimizer, and criterion
        self.model = self._create_model(input_size, sequence_length)
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)

        # Log model architecture
        if self.config.wandb_enabled:
            wandb.watch(self.model, log="all", log_freq=100)

        # Training loop
        best_val_loss = float("inf")
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Input size: {input_size}, Sequence length: {sequence_length}")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
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
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # Log to wandb
            if self.config.wandb_enabled:
                log_dict = {
                    "epoch": epoch + 1,
                    **train_metrics,
                    **val_metrics,
                    "epoch_time": epoch_time,
                }
                wandb.log(log_dict)

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

            if self.config.wandb_enabled:
                wandb.log({"test": test_metrics})

        # Save final model
        self._save_final_model(feature_names, label_names)

        # Finish wandb run
        if self.config.wandb_enabled:
            wandb.finish()

        return {
            "training_history": training_history,
            "test_metrics": test_metrics,
            "best_val_loss": best_val_loss,
            "total_training_time": total_time,
        }

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"

        save_model(
            self.model,
            str(checkpoint_path),
            config=self.config.to_dict(),
            metadata={
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "model_type": self.config.model_type,
            },
        )

    def _save_final_model(
        self, feature_names: Optional[List[str]], label_names: Optional[List[str]]
    ):
        """Save the final trained model."""
        model_save_path = Path(self.config.model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_type": self.config.model_type,
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

    # Prepare data
    train_loader, val_loader, test_loader, scaler, feature_names, label_names = (
        prepare_data_for_training(features, labels, config, feature_columns, label_columns)
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
