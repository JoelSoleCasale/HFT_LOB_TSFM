"""
Neural network architectures for financial time series prediction.
"""

import torch
import torch.nn as nn
import math


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
    ):
        super().__init__(input_size, output_size)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

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


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


# Factory function for creating models
def create_model(model_type: str, **kwargs) -> FinancialTimeSeriesModel:
    """
    Factory function to create models by type.

    Args:
        model_type: Type of model to create ("mlp", "lstm", "transformer")
        **kwargs: Model-specific parameters

    Returns:
        Instantiated model
    """
    model_classes = {
        "mlp": MLPTimeSeriesModel,
        "lstm": LSTMTimeSeriesModel,
        "transformer": TransformerTimeSeriesModel,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}"
        )

    return model_classes[model_type](**kwargs)
