"""
Core interfaces for metrics system.
"""

from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
from typing import Any
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric computation results."""

    name: str
    value: float
    metadata: dict[str, Any] | None = None


class MetricCalculator(ABC):
    """Base class for metric calculators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricResult | list[MetricResult]:
        """Calculate metric(s) from predictions and targets."""
        pass

    @property
    def requires_probabilities(self) -> bool:
        """Whether this metric needs probability scores."""
        return False


class MetricPlotter(ABC):
    """Base class for metric plotters."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def plot(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> plt.Figure:
        """Generate plot from predictions and targets."""
        pass

    @property
    def requires_probabilities(self) -> bool:
        """Whether this plotter needs probability scores."""
        return False


class MetricSuite:
    """Container for multiple metrics and plots."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.calculators: list[MetricCalculator] = []
        self.plotters: list[MetricPlotter] = []

    def add_calculator(self, calculator: MetricCalculator) -> "MetricSuite":
        """Add a metric calculator."""
        self.calculators.append(calculator)
        return self

    def add_plotter(self, plotter: MetricPlotter) -> "MetricSuite":
        """Add a plot generator."""
        self.plotters.append(plotter)
        return self

    def compute_all(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> dict[str, float]:
        """Compute all metrics."""
        results = {}
        for calc in self.calculators:
            metric_results = calc.calculate(predictions, targets, **kwargs)
            if isinstance(metric_results, list):
                for result in metric_results:
                    results[result.name] = result.value
            else:
                results[metric_results.name] = metric_results.value
        return results

    def generate_all_plots(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> dict[str, plt.Figure]:
        """Generate all plots."""
        plots = {}
        for plotter in self.plotters:
            fig = plotter.plot(predictions, targets, **kwargs)
            plots[plotter.name] = fig
        return plots
