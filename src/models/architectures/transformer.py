"""
Transformer architecture for time series prediction.
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass

from .base import ModelArchitectureConfig, FinancialTimeSeriesModel


@dataclass
class TransformerConfig(ModelArchitectureConfig):
    """Configuration for Transformer models."""

    model_type: str = "transformer"
    d_model: int = 64
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 256
    activation: str = "relu"
    sequence_length: int = 5000
    learned_positional_encoding: bool = False  # If True, use learned PE; if False, use sinusoidal


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Supports both fixed sinusoidal and learned positional encodings.
    """

    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1, learned: bool = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned = learned

        if learned:
            # Learnable positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pe, mean=0, std=0.02)
        else:
            # Fixed sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerTimeSeriesModel(FinancialTimeSeriesModel):
    """
    Transformer-based model for time series prediction.

    This model uses transformer encoder layers to process sequential data
    and makes predictions based on the final output.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
        dim_feedforward: int = 256,
        activation: str = "relu",
        sequence_length: int = 5000,
        learned_positional_encoding: bool = False,
    ):
        super().__init__(input_size, output_size)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=sequence_length, dropout=dropout, learned=learned_positional_encoding
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer forward pass
        transformer_out = self.transformer(x)

        # Use the last time step for prediction
        output = transformer_out[:, -1, :]

        # Project to output size
        return self.output_projection(output)
