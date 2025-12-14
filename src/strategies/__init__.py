"""
Strategies module for converting models into executable trading strategies
compatible with hftbacktest.
"""

from .base import BaseStrategy, StrategyConfig
from .deep_learning import (
    DeepLearningStrategy,
    ClassificationStrategy,
    ClassificationStrategyConfig,
)
from .utils import backtest_model, print_backtest_summary

__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "DeepLearningStrategy",
    "ClassificationStrategy",
    "ClassificationStrategyConfig",
    "backtest_model",
    "print_backtest_summary",
]
