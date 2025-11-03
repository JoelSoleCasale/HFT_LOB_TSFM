"""
Training utilities for deep learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path
import time
from tqdm import tqdm


from models.model import create_model, FinancialTimeSeriesModel
from models.utils import (
    get_device,
    save_model,
    set_seed,
    count_parameters,
)
from models.config import ModelConfig
from models.factories import create_optimizer, create_scheduler, create_criterion
from models.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    WandbMetricsLogger,
    TrainingHistoryTracker,
    ConsoleLogger,
)
from utils import setup_logging


class ModelTrainer:
    """
    Main training class for deep learning models.
    """

    def __init__(self, config: ModelConfig, callbacks: list[Callback] | None = None):
        """
        Initialize the trainer.

        Args:
            config: Model configuration
            callbacks: List of callbacks to use during training
        """
        self.config = config
        self.device = get_device(config.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.callbacks = callbacks or []
        self.should_stop = False
        self.test_loader = None  # For use by callbacks

        # Set up logging
        setup_logging(config.logging.log_file)

        # Set random seed
        set_seed(config.seed)

        logger.info(f"ModelTrainer initialized with device: {self.device}")
        logger.info(f"Configuration: {config}")

    def _create_model(self) -> FinancialTimeSeriesModel:
        """Create the model based on configuration."""
        arch_config = self.config.get_architecture_config()
        model = create_model(**arch_config.params())
        model = model.to(self.device)

        logger.info(
            f"Created {arch_config.model_type} model with {count_parameters(model):,} parameters"
        )
        return model

    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer based on configuration."""
        training_config = self.config.get_training_config()

        return create_optimizer(
            optimizer_name=training_config.optimizer,
            model_parameters=self.model.parameters(),
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler based on configuration."""
        training_config = self.config.get_training_config()

        return create_scheduler(
            scheduler_name=training_config.scheduler,
            optimizer=self.optimizer,
            scheduler_params=training_config.scheduler_params,
            num_epochs=training_config.num_epochs,
        )

    def _create_criterion(self) -> nn.Module:
        """Create the loss function based on configuration."""
        training_config = self.config.get_training_config()

        return create_criterion(
            loss_name=training_config.loss_function, loss_params=training_config.loss_params
        )

    def _train_epoch(
        self, train_loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (all_predictions, all_targets, avg_loss, epoch_time)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        training_config = self.config.get_training_config()

        # Use tqdm only if enabled in config
        if training_config.use_tqdm:
            progress_bar = tqdm(train_loader, desc="Training", leave=False)
        else:
            progress_bar = train_loader

        epoch_start_time = time.time()

        for batch_idx, (sequences, labels) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if training_config.mixed_precision:
                with torch.autocast(device_type=self.device.type):
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping with unscaling
                if training_config.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), training_config.gradient_clip_norm
                    )

                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if training_config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), training_config.gradient_clip_norm
                    )

                # Optimizer step
                self.optimizer.step()

            # Store loss and predictions
            total_loss += loss.item()
            num_batches += 1
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(labels.cpu())

            # Update progress bar (only if tqdm is enabled)
            if training_config.use_tqdm:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return all_predictions, all_targets, avg_loss, epoch_time

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Validate for one epoch.

        Returns:
            Tuple of (all_predictions, all_targets, avg_loss)
        """
        self.model.eval()
        total_loss = 0.0
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

                # Store loss and predictions
                total_loss += loss.item()
                num_batches += 1

                # Store predictions and labels
                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / num_batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return all_predictions, all_labels, avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        feature_names: list[str] | None = None,
        label_names: list[str] | None = None,
    ) -> None:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            feature_names: Names of feature columns
            label_names: Names of label columns

        Returns:
            None
        """
        training_config = self.config.get_training_config()

        # Create model, optimizer, and criterion
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()

        # Initialize gradient scaler for mixed precision training
        if training_config.mixed_precision:
            self.scaler = torch.amp.GradScaler(self.device)

        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(training_config.num_epochs):

            train_predictions, train_targets, train_loss, train_time = self._train_epoch(
                train_loader
            )
            val_predictions, val_targets, val_loss = self._validate_epoch(val_loader)

            for callback in self.callbacks:
                callback.on_epoch_end(
                    self,
                    epoch,
                    train_predictions,
                    train_targets,
                    train_loss,
                    train_time,
                    val_predictions,
                    val_targets,
                    val_loss,
                )

            if self.should_stop:
                break

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        self._save_final_model(feature_names, label_names)

        for callback in self.callbacks:
            y_pred, y_true = self.predict(test_loader)
            callback.on_train_end(self, (y_pred, y_true))

        return None

    def _save_final_model(self, feature_names: list[str] | None, label_names: list[str] | None):
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

    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
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
        predictions, targets, avg_loss = self._validate_epoch(test_loader)

        # Compute metrics using the metrics system
        from models.metrics.calculators import AccuracyCalculator, PrecisionRecallF1Calculator

        metrics = {"test_loss": avg_loss}

        # Calculate metrics
        acc_calc = AccuracyCalculator()
        acc_result = acc_calc.calculate(predictions, targets)
        metrics["test_accuracy"] = acc_result.value

        # Add precision, recall, F1
        macro_calc = PrecisionRecallF1Calculator("macro")
        macro_results = macro_calc.calculate(predictions, targets)
        for result in macro_results:
            metrics[f"test_{result.name}"] = result.value

        weighted_calc = PrecisionRecallF1Calculator("weighted")
        weighted_results = weighted_calc.calculate(predictions, targets)
        for result in weighted_results:
            metrics[f"test_{result.name}"] = result.value

        return metrics

    def predict(self, data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
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
    feature_columns: list[str] | None = None,
    label_columns: list[str] | None = None,
) -> tuple[ModelTrainer, dict[str, object]]:
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

    # Prepare data using the data configuration (with temporal splits)
    data_config = config.get_data_config()
    train_loader, val_loader, test_loader, scaler, feature_names, label_names = (
        prepare_data_for_training(features, labels, data_config, feature_columns, label_columns)
    )

    # Create callbacks
    logging_config = config.get_logging_config()
    training_config = config.get_training_config()

    # Create history tracker
    history_tracker = TrainingHistoryTracker()

    callbacks = [
        ConsoleLogger(),
        history_tracker,
        EarlyStopping(
            patience=training_config.early_stopping_patience,
            min_delta=0.0,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            checkpoint_dir=logging_config.checkpoint_dir,
            save_best_only=True,
        ),
    ]

    # Add WandB logger if enabled
    if logging_config.wandb_enabled:
        callbacks.append(
            WandbMetricsLogger(
                log_config=logging_config,
                train_config=config.to_dict(),
                log_frequency=logging_config.log_metrics_frequency,
                num_classes=config.get_data_config().num_classes,
                class_names=config.get_data_config().class_names,
            )
        )

    # Create trainer with callbacks
    trainer = ModelTrainer(config, callbacks=callbacks)

    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_names=feature_names,
        label_names=label_names,
    )

    return trainer, history_tracker.get_history()
