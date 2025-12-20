"""
Modular metrics system for model training and evaluation.

This module provides a structured, scalable system for computing and visualizing
metrics, with easy integration with WandB logging.

Example usage:
    from models.metrics import create_trading_suite

    # Create a pre-configured suite
    suite = create_trading_suite(num_classes=3, class_names=["sell", "hold", "buy"])

    # Compute all metrics
    metrics = suite.compute_all(predictions, targets)

    # Generate all plots
    plots = suite.generate_all_plots(predictions, targets)
"""

from .core import MetricCalculator, MetricPlotter, MetricSuite, MetricResult
from .calculators import (
    AccuracyCalculator,
    PrecisionRecallF1Calculator,
    TradeAccuracyCalculator,
    StrictTradeAccuracyCalculator,
    ConfusionMatrixCalculator,
    PerClassMetricsCalculator,
    ROCAUCCalculator,
    ExpectedReturnCalculator,
)
from .plotters import (
    ConfusionMatrixPlotter,
    TradeAccuracyVsThresholdPlotter,
    ROCCurvePlotter,
    PredictionDistributionPlotter,
    ExpectedReturnVsThresholdPlotter,
)
from .presets import (
    create_classification_suite,
    create_trading_suite,
    create_minimal_suite,
)
from .utils import calculate_accuracy, calculate_trade_accuracy

__all__ = [
    # Core classes
    "MetricCalculator",
    "MetricPlotter",
    "MetricSuite",
    "MetricResult",
    # Calculators
    "AccuracyCalculator",
    "PrecisionRecallF1Calculator",
    "TradeAccuracyCalculator",
    "StrictTradeAccuracyCalculator",
    "ConfusionMatrixCalculator",
    "PerClassMetricsCalculator",
    "ROCAUCCalculator",
    "ExpectedReturnCalculator",
    # Plotters
    "ConfusionMatrixPlotter",
    "TradeAccuracyVsThresholdPlotter",
    "ROCCurvePlotter",
    "PredictionDistributionPlotter",
    "ExpectedReturnVsThresholdPlotter",
    # Presets
    "create_classification_suite",
    "create_trading_suite",
    "create_minimal_suite",
    # Utilities
    "calculate_accuracy",
    "calculate_trade_accuracy",
]
