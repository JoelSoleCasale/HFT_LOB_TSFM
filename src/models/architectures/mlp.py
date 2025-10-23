"""
Multi-layer Perceptron (MLP) architecture for time series prediction.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

from .base import ModelArchitectureConfig, FinancialTimeSeriesModel


@dataclass
class MLPConfig(ModelArchitectureConfig):
    """Configuration for MLP models."""

    model_type: str = "mlp"
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"


class MLPTimeSeriesModel(FinancialTimeSeriesModel):
    """
    Multi-layer perceptron for time series prediction.

    This model flattens the input sequence and processes it through
    fully connected layers.
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        hidden_sizes: list = [64, 32],
        output_size: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__(input_size, output_size)

        self.sequence_length = sequence_length
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        # Flatten input
        flattened_size = input_size * sequence_length

        # Build layers
        layers = []
        prev_size = flattened_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(activation, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Flatten the sequence
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        return self.network(x)
