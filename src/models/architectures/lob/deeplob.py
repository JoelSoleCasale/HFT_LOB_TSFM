"""
DeepLOB: Deep Convolutional Neural Networks for Limit Order Books

Reference:
    Zhang, Z., Zohren, S., & Roberts, S. (2019).
    DeepLOB: Deep convolutional neural networks for limit order books.
    IEEE Transactions on Signal Processing, 67(11), 3001-3012.
    arXiv: https://arxiv.org/abs/1808.03668

Architecture:
    1. Three convolutional blocks that extract spatial features from LOB data
    2. Inception module with three parallel paths (1x1->3x1, 1x1->5x1, maxpool->1x1)
    3. LSTM layer to capture temporal dependencies
    4. Fully connected output layer for classification
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from ..base import ModelArchitectureConfig, FinancialTimeSeriesModel


@dataclass
class DeepLOBConfig(ModelArchitectureConfig):
    """Configuration for DeepLOB model.

    The DeepLOB model expects input shape: (batch, 1, time_steps, features)
    where features is typically 40 (10 levels of bid/ask prices and volumes).
    """

    model_type: str = "deeplob"

    # Convolutional block parameters
    conv_filters: int = 32  # Number of filters in conv blocks

    # Inception module parameters
    inception_filters: int = 64  # Number of filters in inception module

    # LSTM parameters
    lstm_hidden_size: int = 64  # Hidden size of LSTM layer

    # Regularization
    dropout: float = 0.2

    # Activation function for conv layers
    leaky_relu_slope: float = 0.01


class DeepLOBModel(FinancialTimeSeriesModel):
    """
    DeepLOB model for limit order book prediction.

    This model combines convolutional layers for spatial feature extraction,
    inception modules for multi-scale feature processing, and LSTM for
    temporal modeling.

    Input shape: (batch_size, 1, time_steps, features)
    Output shape: (batch_size, output_size)

    Example:
        >>> config = DeepLOBConfig(input_size=40, output_size=3)
        >>> model = DeepLOBModel.from_config(config)
        >>> x = torch.randn(32, 1, 100, 40)  # batch=32, timesteps=100, features=40
        >>> output = model(x)  # shape: (32, 3)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 3,
        conv_filters: int = 32,
        inception_filters: int = 64,
        lstm_hidden_size: int = 64,
        dropout: float = 0.2,
        leaky_relu_slope: float = 0.01,
    ):
        """
        Initialize DeepLOB model.

        Args:
            input_size: Number of input features (typically 40 for 10-level LOB)
            output_size: Number of output classes
            conv_filters: Number of filters in convolutional blocks
            inception_filters: Number of filters in inception module
            lstm_hidden_size: Hidden size of LSTM layer
            dropout: Dropout rate (not used in original architecture)
            activation: Activation function (not used, fixed to match original)
            leaky_relu_slope: Negative slope for LeakyReLU
            use_batch_norm: Whether to use batch normalization (always True in original)
        """
        super().__init__(input_size, output_size)

        self.conv_filters = conv_filters
        self.inception_filters = inception_filters
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout = dropout

        # First convolutional block: LeakyReLU activation
        # Input: (batch, 1, T, 40) -> (batch, 32, T, 20)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_filters, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
        )

        # Second convolutional block: Tanh activation
        # (batch, 32, T-6, 20) -> (batch, 32, T-6, 10)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=conv_filters,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            nn.Tanh(),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(conv_filters),
        )

        # Third convolutional block: LeakyReLU activation
        # (batch, 32, T-12, 10) -> (batch, 32, T-18, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
            nn.Conv2d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(conv_filters),
        )

        # Inception module - three parallel paths
        # Path 1: 1x1 conv -> 3x1 conv
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
            nn.Conv2d(
                in_channels=inception_filters,
                out_channels=inception_filters,
                kernel_size=(3, 1),
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
        )

        # Path 2: 1x1 conv -> 5x1 conv
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
            nn.Conv2d(
                in_channels=inception_filters,
                out_channels=inception_filters,
                kernel_size=(5, 1),
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
        )

        # Path 3: maxpool -> 1x1 conv
        self.inp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=conv_filters,
                out_channels=inception_filters,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.BatchNorm2d(inception_filters),
        )

        # After concatenating 3 inception paths: 64*3 = 192 channels
        inception_output_size = inception_filters * 3

        # Dropout layers for regularization
        self.dropout_conv = nn.Dropout2d(p=dropout)  # 2D dropout for conv features
        self.dropout_inception = nn.Dropout2d(p=dropout)  # 2D dropout after inception
        self.dropout_lstm = nn.Dropout(p=dropout)  # 1D dropout after LSTM

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=inception_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Output layer
        self.fc1 = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DeepLOB.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
               OR (batch_size, channels, time_steps, features)
               For LOB data: typically (batch_size, 100, 40)

        Returns:
            Output tensor of shape (batch_size, output_size) with softmax applied
        """
        # Handle both 3D and 4D input tensors
        if x.dim() == 3:
            # Input is (batch, time, features) - add channel dimension
            # Reshape to (batch, 1, time, features)
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(
                f"Expected 3D (batch, time, features) or 4D (batch, channels, time, features) input, "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )

        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.lstm_hidden_size).to(x.device)

        # Convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout_conv(x)  # Dropout after final conv block

        # Inception module - three parallel paths
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        # Concatenate along channel dimension
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = self.dropout_inception(x)  # Dropout after inception module

        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        # LSTM layer
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.dropout_lstm(x)  # Dropout after LSTM

        # Output layer
        x = self.fc1(x)

        return x
