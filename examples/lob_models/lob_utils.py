"""
Utility functions for LOB model training examples.

This module contains common code used across all LOB model training examples
to reduce code duplication.
"""

from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import polars as pl
import numpy as np

from core.orderbook import OrderBook
from features import InputSpace, TripleBarrierLabel
from models.config import DataConfig, TrainingConfig, LoggingConfig
from utils import date_range
from definitions import ROOT_DIR
from sklearn.utils.class_weight import compute_class_weight

# Default configuration constants
DEFAULT_FIRST_DATE = date(2025, 7, 1)
DEFAULT_N_DAYS = 10
DEFAULT_LEVELS = 10
DEFAULT_SAMPLE_TIME_DELTA = 100_000_000  # nanoseconds
DEFAULT_HORIZON = 200
DEFAULT_SEQUENCE_LENGTH = 200
DEFAULT_THRESHOLD = 2e-4


def get_orderbook_path(
    target_date: date,
    symbol: str = "BTCUSDT",
    exchange: str = "binance_futures",
    levels: int = 20,
    root_dir: Path = ROOT_DIR,
) -> Path:
    """
    Get the path to an orderbook snapshot file.

    Args:
        target_date: Date of the snapshot
        symbol: Trading symbol (default: BTCUSDT)
        exchange: Exchange name (default: binance_futures)
        levels: Number of orderbook levels (default: 20)
        root_dir: Root directory of the project

    Returns:
        Path to the orderbook snapshot file
    """
    return (
        root_dir
        / f"data/orderbook_snapshots/{exchange}/{symbol}/{target_date.strftime('%Y-%m-%d')}_L{levels}.parquet"
    )


def load_orderbook_data(
    first_date: date = DEFAULT_FIRST_DATE,
    n_days: int = DEFAULT_N_DAYS,
    levels: int = DEFAULT_LEVELS,
    symbol: str = "BTCUSDT",
    exchange: str = "binance_futures",
    sample_time_delta: Optional[int] = DEFAULT_SAMPLE_TIME_DELTA,
    interpolate: bool = True,
) -> OrderBook:
    """
    Load orderbook data for a date range.

    Args:
        first_date: Start date
        n_days: Number of days to load
        levels: Number of orderbook levels to select
        symbol: Trading symbol (default: BTCUSDT)
        exchange: Exchange name (default: binance_futures)
        sample_time_delta: Time delta for sampling in nanoseconds (optional)
        interpolate: Whether to interpolate when sampling (default: True)

    Returns:
        OrderBook with loaded data
    """
    ob_paths = [
        get_orderbook_path(d, symbol, exchange, 20)  # Source files are L20
        for d in date_range(first_date, first_date + timedelta(days=n_days - 1))
    ]

    orderbook_data = OrderBook.from_parquet(ob_paths, lazy=True).select_levels(levels)

    if sample_time_delta is not None:
        orderbook_data = orderbook_data.sample_by_time(
            time_delta=sample_time_delta, interpolate=interpolate
        )

    return orderbook_data


def create_input_space(orderbook_data: OrderBook) -> InputSpace:
    """
    Create an InputSpace from orderbook data.

    Args:
        orderbook_data: Loaded orderbook data

    Returns:
        InputSpace containing the orderbook snapshots
    """
    return InputSpace(orderbook_snapshots=orderbook_data)


def extract_labels(
    input_space: InputSpace,
    horizon: int = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[pl.LazyFrame, TripleBarrierLabel]:
    """
    Extract labels from input space using TripleBarrierLabel.

    Args:
        input_space: InputSpace containing orderbook data
        horizon: Number of timesteps for label calculation (default: 200)
        threshold: Price movement threshold for directional labels (default: 1e-4)

    Returns:
        LazyFrame with extracted labels and the label extractor instance
    """
    label_extractor = TripleBarrierLabel(config={"horizon": horizon, "threshold": threshold})
    labels = label_extractor.extract(input_space)
    return labels, label_extractor


def print_label_distribution(labels_collected: pl.DataFrame, label_column_name: str) -> None:
    """
    Print the distribution of labels in the dataset.

    Args:
        labels_collected: Collected labels DataFrame
        label_column_name: Name of the label column to analyze
    """
    label_counts = labels_collected[label_column_name].value_counts()
    total_samples = len(labels_collected)

    print("\nLabel distribution:")
    for label_value, count in label_counts.iter_rows():
        percentage = (count / total_samples) * 100
        label_name = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}.get(label_value, label_value)
        print(f"  {label_name:8s} (label={label_value:2d}): {count:,} samples ({percentage:.2f}%)")


def calculate_class_weights(labels_collected: pl.DataFrame, label_column_name: str) -> list[float]:
    """
    Calculate class weights for imbalanced data (inverse frequency weighting).

    Args:
        labels_collected: Collected labels DataFrame
        label_column_name: Name of the label column to analyze

    Returns:
        List of class weights [down, neutral, up]
    """

    labels = labels_collected[label_column_name].to_numpy()
    class_weights = compute_class_weight("balanced", classes=np.array([-1, 0, 1]), y=labels)

    print(f"\nClass weights for focal loss: {[f'{w:.2f}' for w in class_weights]}")
    return list(class_weights)


def print_training_header(model_name: str, first_date: date, n_days: int) -> None:
    """
    Print the training header with model information.

    Args:
        model_name: Name of the model being trained
        first_date: Start date of training data
        n_days: Number of days of training data
    """
    print("=" * 80)
    print(f"{model_name} Model Training Example")
    print("=" * 80)
    print(f"\nLoading orderbook data for {n_days} days starting from {first_date}...")


def print_data_loaded() -> None:
    """Print confirmation that data was loaded successfully."""
    print("✓ Orderbook data loaded successfully")
    print("\nCreating input space and feature pipeline...")


def print_pipeline_created(extractor_name: str) -> None:
    """
    Print confirmation that the feature pipeline was created.

    Args:
        extractor_name: Name of the feature extractor
    """
    print(f"✓ Feature pipeline created with {extractor_name} extractor")
    print("\nExtracting features and labels...")


def print_features_extracted(
    features_shape: tuple, labels_shape: tuple, expected_features: int, levels: int
) -> None:
    """
    Print information about extracted features and labels.

    Args:
        features_shape: Shape of features DataFrame
        labels_shape: Shape of labels DataFrame
        expected_features: Expected number of features
        levels: Number of orderbook levels
    """
    print(f"✓ Features extracted: {features_shape}")
    print(f"  Expected: {expected_features} features ({levels} levels × 4 features per level)")
    print(f"✓ Labels extracted: {labels_shape}")


def print_architecture_config(model_name: str, config_dict: dict) -> None:
    """
    Print model architecture configuration.

    Args:
        model_name: Name of the model architecture
        config_dict: Dictionary of configuration parameters
    """
    print("\n" + "=" * 80)
    print(f"Configuring {model_name} Model")
    print("=" * 80)
    print(f"\nArchitecture: {model_name}")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")


def print_training_start(config) -> None:
    """
    Print training configuration before starting training.

    Args:
        config: ModelConfig instance
    """
    print("\n" + "=" * 80)
    print("Starting Model Training")
    print("=" * 80)
    print("\nData configuration:")
    print(f"  Sequence length: {config.data.sequence_length}")
    print(f"  Batch size: {config.data.batch_size}")
    print(
        f"  Train/Val/Test split: {config.data.train_split:.0%}/"
        f"{config.data.val_split:.0%}/{config.data.test_split:.0%}"
    )
    print(f"  Device: {config.data.device}")

    print("\nTraining configuration:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Loss function: {config.training.loss_function}")
    if hasattr(config.training, "mixed_precision"):
        print(f"  Mixed precision: {config.training.mixed_precision}")


def create_data_config(
    sequence_length: int,
    batch_size: int = 128,
    stride: int = 5,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    device: str = "cuda",
    **kwargs,
) -> DataConfig:
    """
    Create a DataConfig with sensible defaults for LOB models.

    Args:
        sequence_length: Number of timesteps in each sequence
        batch_size: Batch size for training (default: 1024)
        stride: Stride for creating sequences (default: 5)
        train_split: Fraction of data for training (default: 0.8)
        val_split: Fraction of data for validation (default: 0.1)
        test_split: Fraction of data for testing (default: 0.1)
        device: Device to train on (default: "cuda")
        **kwargs: Additional arguments to pass to DataConfig

    Returns:
        Configured DataConfig instance
    """
    return DataConfig(
        sequence_length=sequence_length,
        batch_size=batch_size,
        stride=stride,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        device=device,
        **kwargs,
    )


def create_training_config(
    learning_rate: float = 0.0001,
    num_epochs: int = 100,
    optimizer: str = "adam",
    loss_function: str = "cross_entropy",
    loss_params: Optional[dict] = None,
    early_stopping_patience: int = 10,
    scheduler: str = "cosine",
    mixed_precision: bool = True,
    gradient_clip_norm: Optional[float] = None,
    **kwargs,
) -> TrainingConfig:
    """
    Create a TrainingConfig with sensible defaults for LOB models.

    Args:
        learning_rate: Learning rate for optimizer (default: 0.0001)
        num_epochs: Maximum number of training epochs (default: 50)
        optimizer: Optimizer name (default: "adam")
        loss_function: Loss function name (default: "focal")
        loss_params: Parameters for loss function (default: None)
        early_stopping_patience: Patience for early stopping (default: 5)
        scheduler: Learning rate scheduler (default: "cosine")
        mixed_precision: Whether to use mixed precision training (default: True)
        gradient_clip_norm: Gradient clipping norm (default: None)
        **kwargs: Additional arguments to pass to TrainingConfig

    Returns:
        Configured TrainingConfig instance
    """
    config_dict = {
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "early_stopping_patience": early_stopping_patience,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_function": loss_function,
        "mixed_precision": mixed_precision,
        **kwargs,
    }

    if loss_params is not None:
        config_dict["loss_params"] = loss_params

    if gradient_clip_norm is not None:
        config_dict["gradient_clip_norm"] = gradient_clip_norm

    return TrainingConfig(**config_dict)


def create_logging_config(
    experiment_name: str,
    lambda_value: float,
    project_name: str = None,
    wandb_enabled: bool = True,
    log_confusion_matrix: bool = True,
    log_trade_accuracy_vs_threshold: bool = True,
    theta_values: Optional[list[float]] = None,
    **kwargs,
) -> LoggingConfig:
    """
    Create a LoggingConfig with sensible defaults for LOB models.

    Args:
        experiment_name: Name of the experiment
        lambda_value: Lambda threshold value for label creation
        project_name: W&B project name (default: "financial-models-lob")
        wandb_enabled: Whether to enable W&B logging (default: True)
        log_confusion_matrix: Whether to log confusion matrix (default: True)
        log_trade_accuracy_vs_threshold: Whether to log trade accuracy vs threshold (default: True)
        theta_values: Threshold values for trade accuracy analysis (default: [0.0, 1e-4, 4e-4])
        **kwargs: Additional arguments to pass to LoggingConfig

    Returns:
        Configured LoggingConfig instance
    """
    if project_name is None:
        project_name = f"lob-models-{DEFAULT_THRESHOLD*1e4:.2}bps-{DEFAULT_HORIZON}H"
    if theta_values is None:
        theta_values = [0.0, 1e-4, 4e-4]

    return LoggingConfig(
        project_name=project_name,
        log_file=ROOT_DIR / f"logs/{experiment_name}.log",
        model_save_path=ROOT_DIR / f"model_checkpoint/{experiment_name}.pt",
        experiment_name=experiment_name,
        wandb_enabled=wandb_enabled,
        log_confusion_matrix=log_confusion_matrix,
        log_trade_accuracy_vs_threshold=log_trade_accuracy_vs_threshold,
        lambda_value=lambda_value,
        theta_values=theta_values,
        **kwargs,
    )
