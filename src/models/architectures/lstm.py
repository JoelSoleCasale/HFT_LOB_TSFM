"""
LSTM architecture for time series prediction.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from .base import ModelArchitectureConfig, FinancialTimeSeriesModel


@dataclass
class LSTMConfig(ModelArchitectureConfig):
    """Configuration for LSTM models."""

    model_type: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    attention: bool = False


class LSTMTimeSeriesModel(FinancialTimeSeriesModel):
    """
    LSTM-based model for time series prediction.

    This model uses LSTM layers to process sequential data and
    makes predictions based on the final hidden state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        attention: bool = False,
    ):
        super().__init__(input_size, output_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Attention mechanism (optional)
        if attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_size * (2 if bidirectional else 1),
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )

        # Output layers
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        if self.attention:
            # Apply attention mechanism
            attn_out, _ = self.attention_layer(lstm_out, lstm_out, lstm_out)
            # Use the last time step
            output = attn_out[:, -1, :]
        else:
            # Use the last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                output = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                output = hidden[-1]

        # Apply fully connected layers
        return self.fc_layers(output)
