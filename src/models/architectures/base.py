"""
Base class for financial time series models.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelArchitectureConfig:
    """Base configuration for model architecture."""

    input_size: int
    model_type: str  # "lstm", "transformer", "mlp"
    output_size: int = 3  # For classification: -1, 0, 1
    dropout: float = 0.2

    def __post_init__(self):
        """Validate model architecture configuration."""
        valid_models = ["lstm", "transformer", "mlp"]
        if self.model_type not in valid_models:
            raise ValueError(f"Model type must be one of {valid_models}, got {self.model_type}")

    def params(self) -> dict[str, object]:
        """Return model-specific parameters as a dictionary."""
        return self.__dict__


class FinancialTimeSeriesModel(nn.Module):
    """
    Base class for financial time series models.
    """

    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
