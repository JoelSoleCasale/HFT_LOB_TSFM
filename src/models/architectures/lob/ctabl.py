"""
CTABL: Temporal Attention-Augmented Bilinear Network for LOB data

Reference:
  Tran, D. T., Iosifidis, A., Kanniainen, J., & Gabbouj, M. (2018).
  Temporal Attention-Augmented Bilinear Network for Financial Time-Series Data Analysis.
  IEEE Transactions on Neural Networks and Learning Systems.
  arXiv: https://arxiv.org/abs/1712.00975

Architecture:
  - One or more Bilinear layers that project along feature/time modes
  - Final TABL layer with temporal attention applied before second-mode projection
  - Default template from paper/examples: [40x10] -> [120x5] -> [3x1]
"""

from dataclasses import dataclass
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ModelArchitectureConfig, FinancialTimeSeriesModel


class BilinearLayer(nn.Module):
    """Bilinear layer projecting a 2D tensor along both modes.

    Input: (batch, D1, D2)
    Output: (batch, d1, d2)
    """

    def __init__(self, in_dims: Tuple[int, int], out_dims: Tuple[int, int]):
        super().__init__()
        D1, D2 = in_dims
        d1, d2 = out_dims

        self.D1 = D1
        self.D2 = D2
        self.d1 = d1
        self.d2 = d2

        self.W1 = nn.Parameter(torch.empty(D1, d1))
        self.W2 = nn.Parameter(torch.empty(D2, d2))
        self.bias = nn.Parameter(torch.zeros(d1, d2))

        self.reset_parameters()

    def reset_parameters(self):
        # He/Kaiming uniform initialization similar to Keras he_uniform
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.D1 or x.shape[2] != self.D2:
            raise ValueError(
                f"Expected input of shape (batch, {self.D1}, {self.D2}), got {tuple(x.shape)}"
            )

        x = torch.einsum("bjd,jk->bkd", x, self.W1)
        x = torch.einsum("bkd,dl->bkl", x, self.W2)
        x = x + self.bias
        return x


class TABLLayer(nn.Module):
    """Temporal Attention-Augmented Bilinear Layer (TABL).

    Applies first-mode projection, computes temporal attention, mixes attended
    features via a learnable alpha, then applies second-mode projection.
    """

    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        alpha_init: float = 0.5,
    ):
        super().__init__()
        D1, D2 = in_dims
        d1, d2 = out_dims

        self.D1 = D1
        self.D2 = D2
        self.d1 = d1
        self.d2 = d2

        self.W1 = nn.Parameter(torch.empty(D1, d1))
        self.W2 = nn.Parameter(torch.empty(D2, d2))
        # Temporal attention matrix (D2 x D2)
        self.W_attn = nn.Parameter(torch.empty(D2, D2))
        # Alpha in logit space so sigmoid keeps it in [0,1]; init to give alpha_init after sigmoid
        alpha_logit = math.log(alpha_init / (1 - alpha_init)) if 0 < alpha_init < 1 else 0.0
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit))
        self.bias = nn.Parameter(torch.zeros(1, d1, d2))

        self.reset_parameters()

    def reset_parameters(self):
        # He/Kaiming uniform initialization with fan-in scaling
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        # Initialize attention matrix with small values scaled by D2
        nn.init.normal_(self.W_attn, mean=0.0, std=1.0 / math.sqrt(self.D2))
        # Set diagonal to 1/D2 for identity-like initialization
        with torch.no_grad():
            self.W_attn.diagonal().fill_(1.0 / self.D2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.D1 or x.shape[2] != self.D2:
            raise ValueError(
                f"Expected input of shape (batch, {self.D1}, {self.D2}), got {tuple(x.shape)}"
            )

        x = torch.einsum("bjd,jk->bkd", x, self.W1)

        # Enforce constant diagonal of attention matrix to 1/D2 for stability
        Id = torch.eye(self.D2, device=x.device, dtype=x.dtype)
        W = self.W_attn - self.W_attn * Id + Id * (1.0 / self.D2)

        attn_scores = torch.einsum("bkd,dl->bkl", x, W)
        attention = F.softmax(attn_scores, dim=-1)

        alpha = torch.sigmoid(self.alpha_logit)
        # alpha=1 keeps original x; alpha=0 uses attended x entirely
        x = alpha * x + (1.0 - alpha) * (x * attention)

        x = torch.einsum("bkd,dl->bkl", x, self.W2)
        x = x + self.bias
        return x


@dataclass
class CTABLConfig(ModelArchitectureConfig):
    """Configuration for CTABL model.

    Default template (from paper/examples):
      input:  (40, 10)
      hidden: (120, 5)
      output: (3, 1)
    """

    model_type: str = "ctabl"
    # Temporal length (second mode of input)
    time_steps: int = 10
    # Hidden bilinear layers (sequence of (d1, d2))
    hidden_dims: tuple[tuple[int, int], ...] = ((120, 5),)
    # Output bilinear dims (d1, d2); d2 should be 1 for classification logits
    output_dims: Tuple[int, int] = (3, 1)
    # Activation between bilinear layers
    activation: str = "relu"


class CTABLModel(FinancialTimeSeriesModel):
    """CTABL model combining Bilinear layers and a final TABL layer.

    Input shape expected by forward: (batch, time, features)
    Internally transposed to (batch, features, time) for bilinear operations.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 3,
        time_steps: int = 10,
        hidden_dims: tuple[tuple[int, int], ...] = ((120, 5),),
        output_dims: Tuple[int, int] = (3, 1),
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__(input_size, output_size)

        if output_dims[0] != output_size:
            raise ValueError(
                f"output_dims first element ({output_dims[0]}) must equal output_size ({output_size})"
            )

        self.D1 = input_size
        self.D2 = time_steps
        self.dropout_p = dropout
        self.activation = activation

        layers: list[nn.Module] = []
        in_dims = (self.D1, self.D2)
        for hd in hidden_dims:
            layers.append(BilinearLayer(in_dims, hd))
            in_dims = hd
        self.hidden_layers = nn.ModuleList(layers)

        self.tabl = TABLLayer(in_dims, output_dims)
        self.dropout = nn.Dropout(self.dropout_p)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Accepts either (batch, time, features) or (batch, features, time).
        Returns raw logits of shape (batch, output_size).
        """
        if x.dim() != 3:
            raise ValueError(
                f"CTABL expects a 3D tensor (batch, time, features) or (batch, features, time), got {x.shape}"
            )

        if x.shape[1] == self.D2 and x.shape[2] == self.D1:
            x = x.transpose(1, 2)
        elif not (x.shape[1] == self.D1 and x.shape[2] == self.D2):
            raise ValueError(
                f"Input shape must be (batch, {self.D2}, {self.D1}) or (batch, {self.D1}, {self.D2}), got {tuple(x.shape)}"
            )

        for layer in self.hidden_layers:
            x = layer(x)
            x = self._activate(x)
            x = self.dropout(x)

        x = self.tabl(x)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
