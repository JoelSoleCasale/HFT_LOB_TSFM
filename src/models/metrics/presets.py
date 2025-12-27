"""
Pre-configured metric suites for common use cases.
"""

from .core import MetricSuite
from .calculators import (
    AccuracyCalculator,
    PrecisionRecallF1Calculator,
    TradeAccuracyCalculator,
    StrictTradeAccuracyCalculator,
    PerClassMetricsCalculator,
    ROCAUCCalculator,
    CohenKappaCalculator,
)
from .plotters import (
    ConfusionMatrixPlotter,
    TradeAccuracyVsThresholdPlotter,
    ROCCurvePlotter,
    PredictionDistributionPlotter,
)


def create_classification_suite(
    num_classes: int,
    class_names: list[str] | None = None,
    include_per_class: bool = True,
    include_roc: bool = True,
) -> MetricSuite:
    """
    Create a standard classification metrics suite.

    Args:
        num_classes: Number of classes in the classification task
        class_names: Optional list of class names for plotting
        include_per_class: Whether to include per-class metrics
        include_roc: Whether to include ROC curves

    Returns:
        MetricSuite configured for classification tasks
    """
    suite = MetricSuite("classification")

    # Add calculators
    suite.add_calculator(AccuracyCalculator())
    suite.add_calculator(PrecisionRecallF1Calculator("macro"))
    suite.add_calculator(PrecisionRecallF1Calculator("weighted"))
    suite.add_calculator(CohenKappaCalculator())

    if include_per_class:
        suite.add_calculator(PerClassMetricsCalculator(num_classes, class_names))

    if include_roc and num_classes > 1:
        suite.add_calculator(ROCAUCCalculator(multi_class="ovr"))

    # Add plotters
    suite.add_plotter(ConfusionMatrixPlotter(class_names))

    if include_roc and num_classes > 1:
        suite.add_plotter(ROCCurvePlotter(class_names))

    suite.add_plotter(PredictionDistributionPlotter(class_names))

    return suite


def create_trading_suite(
    num_classes: int,
    class_names: list[str] | None = None,
    trade_threshold: float = 0.5,
    include_per_class: bool = True,
) -> MetricSuite:
    """
    Create a trading-specific metrics suite.

    Args:
        num_classes: Number of classes in the classification task
        class_names: Optional list of class names for plotting
        trade_threshold: Confidence threshold for trade accuracy calculation
        include_per_class: Whether to include per-class metrics

    Returns:
        MetricSuite configured for trading tasks
    """
    # Start with classification suite
    suite = create_classification_suite(
        num_classes=num_classes, class_names=class_names, include_per_class=include_per_class
    )

    # Add trading-specific metrics (both normal and strict)
    suite.add_calculator(TradeAccuracyCalculator(trade_threshold))
    suite.add_calculator(StrictTradeAccuracyCalculator(trade_threshold))
    suite.add_plotter(TradeAccuracyVsThresholdPlotter())

    return suite


def create_minimal_suite() -> MetricSuite:
    """
    Create a minimal metrics suite with only essential metrics.

    Returns:
        MetricSuite with basic metrics only
    """
    suite = MetricSuite("minimal")

    # Add only basic metrics
    suite.add_calculator(AccuracyCalculator())
    suite.add_calculator(PrecisionRecallF1Calculator("macro"))

    return suite
