"""
Multi-layer Perceptron (MLP) architecture for time series prediction.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field

from .base import ModelArchitectureConfig, FinancialTimeSeriesModel


@dataclass
class MLPConfig(ModelArchitectureConfig):
    """Configuration for MLP models."""

    model_type: str = "mlp"
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    sequence_length: int = 128


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

        # Learnable weights for linear combination across sequence length
        # This will produce a weighted combination of time steps
        self.sequence_weights = nn.Linear(sequence_length, 1, bias=False)

        # Build MLP layers on the reduced representation
        layers = []
        prev_size = input_size

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
        # Compute learnable linear combination across sequence length
        # x shape: (batch_size, sequence_length, input_size)
        # Transpose to (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)

        # Apply learnable weights to combine sequence dimension
        # Result shape: (batch_size, input_size, 1)
        x = self.sequence_weights(x)

        # Squeeze to (batch_size, input_size)
        x = x.squeeze(-1)

        # Pass through MLP
        return self.network(x)
