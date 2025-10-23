"""
Models module for deep learning model training and inference.

This module provides:
- Neural network architectures for financial time series prediction
- Training utilities with wandb integration
- Data preparation and preprocessing
- Model configuration management
"""

from .model import create_model
from .architectures import (
    FinancialTimeSeriesModel,
    MLPTimeSeriesModel,
    LSTMTimeSeriesModel,
    TransformerTimeSeriesModel,
)
from .architectures.transformer import PositionalEncoding
from .train import ModelTrainer, train_model
from .data import FinancialDataset, create_dataloaders
from .utils import save_model, load_model, get_device
from .config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
)
from .architectures import (
    ModelArchitectureConfig,
    LSTMConfig,
    TransformerConfig,
    MLPConfig,
)
from .metrics import MetricsCalculator, MetricsLogger, create_metrics_logger

__all__ = [
    # Model architectures
    "FinancialTimeSeriesModel",
    "MLPTimeSeriesModel",
    "LSTMTimeSeriesModel",
    "TransformerTimeSeriesModel",
    "PositionalEncoding",
    "create_model",
    # Training
    "ModelTrainer",
    "train_model",
    # Data
    "FinancialDataset",
    "create_dataloaders",
    # Utilities
    "save_model",
    "load_model",
    "get_device",
    # Configuration classes
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "LoggingConfig",
    "ModelArchitectureConfig",
    "LSTMConfig",
    "TransformerConfig",
    "MLPConfig",
    # Metrics
    "MetricsCalculator",
    "MetricsLogger",
    "create_metrics_logger",
]
