"""
DeepLOB-Attention: Conv-Inception front-end with Luong-style attention

Reference:
    Zhang, Z., Zohren, S. (2021).
    Multi-Horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches
    and Hardware Acceleration using Intelligent Processing Units.
    arXiv: https://arxiv.org/abs/2105.10430

This PyTorch implementation follows the DeepLOB convolutional/inception front-end
and replaces the final temporal modeling with an LSTM encoder and Luong dot-product
attention pooling over encoder outputs to produce a single-horizon classification.

Input shape: (batch_size, 1, time_steps, features)
Output shape: (batch_size, output_size)

Default hyperparameters mirror the paper/notebooks:
    time_steps ~ 50, features = 40, conv_filters = 32, inception_filters = 64,
    lstm_hidden_size = 64, LeakyReLU slope = 0.01.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from ..base import ModelArchitectureConfig, FinancialTimeSeriesModel


@dataclass
class DeepLOBAttentionConfig(ModelArchitectureConfig):
    """Configuration for DeepLOB-Attention model.

    The model expects input shape: (batch, 1, time_steps, features)
    where features is typically 40 (10 levels of bid/ask prices and volumes).
    """

    model_type: str = "deeplob_attention"

    # Convolutional front-end parameters
    conv_filters: int = 32

    # Inception module parameters
    inception_filters: int = 64

    # Encoder LSTM parameters
    lstm_hidden_size: int = 64

    # Regularization
    dropout: float = 0.0

    # Activation function for conv layers
    activation: Literal["leaky_relu", "tanh"] = "leaky_relu"
    leaky_relu_slope: float = 0.01

    # Batch normalization in attention context (disabled by default for simplicity)
    use_batch_norm: bool = True


class DeepLOBAttentionModel(FinancialTimeSeriesModel):
    """
    DeepLOB-Attention model for limit order book prediction (single-horizon).

    This model combines convolutional layers for spatial feature extraction,
    inception modules for multi-scale feature processing, and an LSTM encoder
    with Luong dot-product attention pooling for temporal modeling.

    Input shape: (batch_size, 1, time_steps, features)
    Output shape: (batch_size, output_size)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 3,
        conv_filters: int = 32,
        inception_filters: int = 64,
        lstm_hidden_size: int = 64,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = 0.01,
        use_batch_norm: bool = True,
    ):
        super().__init__(input_size, output_size)

        self.conv_filters = conv_filters
        self.inception_filters = inception_filters
        self.lstm_hidden_size = lstm_hidden_size
        self.leaky_relu_slope = leaky_relu_slope
        self.use_batch_norm = use_batch_norm

        act_leaky = nn.LeakyReLU(negative_slope=leaky_relu_slope)

        # Convolutional front-end closely mirroring the attention notebooks
        # Block 1: (1x2 stride) -> (4x1) -> (4x1)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_filters, kernel_size=(1, 2), stride=(1, 2)),
            act_leaky,
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(4, 1),
                padding=(0, 0),
            ),
            act_leaky,
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(4, 1),
                padding=(0, 0),
            ),
            act_leaky,
        )

        # Block 2: (1x2 stride) -> (4x1) -> (4x1)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            act_leaky,
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(4, 1),
                padding=(0, 0),
            ),
            act_leaky,
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(4, 1),
                padding=(0, 0),
            ),
            act_leaky,
        )

        # Block 3: (1x10) -> (4x1)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=(1, 10)),
            act_leaky,
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(4, 1),
                padding=(0, 0),
            ),
            act_leaky,
        )

        # Inception module paths
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(1, 1),
                padding=0,
            ),
            act_leaky,
            nn.Conv2d(
                in_channels=inception_filters,
                out_channels=inception_filters,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
            act_leaky,
        )

        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(1, 1),
                padding=0,
            ),
            act_leaky,
            nn.Conv2d(
                in_channels=inception_filters,
                out_channels=inception_filters,
                kernel_size=(5, 1),
                padding=(2, 0),
            ),
            act_leaky,
        )

        self.inp3_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.inp3_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(1, 1),
                padding=0,
            ),
            act_leaky,
        )

        inception_output_size = inception_filters * 3

        # Encoder LSTM over time dimension
        self.encoder_lstm = nn.LSTM(
            input_size=inception_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Final classifier taking [context, h_n]
        self.fc = nn.Linear(lstm_hidden_size * 2, output_size)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for DeepLOB-Attention.

        Accepts either 3D (batch, time, features) or 4D (batch, channels, time, features).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(
                f"Expected 3D (batch, time, features) or 4D (batch, channels, time, features) input, got {x.dim()}D tensor with shape {x.shape}"
            )

        # Conv front-end
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Inception module
        inp3 = self.inp3_pool(x)
        inp3 = self.inp3_conv(inp3)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x = torch.cat((x_inp1, x_inp2, inp3), dim=1)

        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1, 3)  # (B, T, C, W)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))  # (B, T, C)

        # Encoder LSTM
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(x)
        # h_n: (num_layers, B, H) -> take last layer
        h_n = h_n[-1]  # (B, H)

        # Luong dot-product attention: scores = encoder_outputs · h_n
        # encoder_outputs: (B, T, H), h_n: (B, H)
        scores = torch.bmm(encoder_outputs, h_n.unsqueeze(2)).squeeze(2)  # (B, T)
        attn_weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)

        # Concatenate context and last hidden state, apply dropout, then classify
        combined = torch.cat([context, h_n], dim=1)
        combined = self.dropout(combined)
        out = self.fc(combined)

        return out
