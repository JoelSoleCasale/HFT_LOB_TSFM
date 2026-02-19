"""
Axial-LOB: High-Frequency Trading with Axial Attention

Reference:
    Berti, L., et al. (2022).
    Axial-LOB: High-Frequency Trading with Axial Attention.
    arXiv: https://arxiv.org/abs/2212.01807

Architecture:
    1. Convolutional layer (1x1) to project input to c_in channels
    2. Axial Attention blocks (Width and Height attention)
    3. Residual connections
    4. Final convolution and pooling
    5. Fully connected output layer
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ..base import ModelArchitectureConfig, FinancialTimeSeriesModel


@dataclass
class AxialLOBConfig(ModelArchitectureConfig):
    """Configuration for AxialLOB model."""

    model_type: str = "axial_lob"

    # Architecture parameters
    c_in: int = 32
    c_out: int = 32
    c_final: int = 4
    n_heads: int = 4
    pool_kernel: Tuple[int, int] = (1, 4)
    pool_stride: Tuple[int, int] = (1, 4)
    sequence_length: int = 40  # Added to calculate linear layer size


def _conv1d1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm1d(out_channels),
    )


class GatedAxialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dim, flag):
        assert (in_channels % heads == 0) and (out_channels % heads == 0)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dim_head_v = out_channels // heads
        self.flag = flag  # if flag then we do the attention along width
        self.dim = dim
        self.dim_head_qk = self.dim_head_v // 2
        self.qkv_channels = self.dim_head_v + self.dim_head_qk * 2

        # Multi-head self attention
        self.to_qkv = _conv1d1x1(in_channels, self.heads * self.qkv_channels)
        self.bn_qkv = nn.BatchNorm1d(self.heads * self.qkv_channels)
        self.bn_similarity = nn.BatchNorm2d(heads * 3)
        self.bn_output = nn.BatchNorm1d(self.heads * self.qkv_channels)

        # Gating mechanism
        self.f_qr = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(
            torch.randn(self.dim_head_v * 2, dim * 2 - 1), requires_grad=True
        )
        query_index = torch.arange(dim).unsqueeze(0)
        key_index = torch.arange(dim).unsqueeze(1)
        relative_index = key_index - query_index + dim - 1
        self.register_buffer("flatten_index", relative_index.view(-1))

        self.reset_parameters()

    def forward(self, x):
        if self.flag:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        x = self.to_qkv(x)

        qkv = self.bn_qkv(x)
        q, k, v = torch.split(
            qkv.reshape(N * W, self.heads, self.dim_head_v * 2, H),
            [self.dim_head_v // 2, self.dim_head_v // 2, self.dim_head_v],
            dim=2,
        )

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(
            self.dim_head_v * 2, self.dim, self.dim
        )
        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings,
            [self.dim_head_qk, self.dim_head_qk, self.dim_head_v],
            dim=0,
        )
        qr = torch.einsum("bgci,cij->bgij", q, q_embedding)
        kr = torch.einsum("bgci,cij->bgij", k, k_embedding).transpose(2, 3)
        qk = torch.einsum("bgci, bgcj->bgij", q, k)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = (
            self.bn_similarity(stacked_similarity).view(N * W, 3, self.heads, H, H).sum(dim=1)
        )
        # (N, heads, H, H, W)
        similarity = torch.softmax(stacked_similarity, dim=3)
        sv = torch.einsum("bgij,bgcj->bgci", similarity, v)
        sve = torch.einsum("bgij,cij->bgci", similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_channels * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_channels, 2, H).sum(dim=-2)

        if self.flag:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        return output

    def reset_parameters(self):
        nn.init.normal_(self.relative, 0.0, math.sqrt(1.0 / self.dim_head_v))


class AxialLOBModel(FinancialTimeSeriesModel):
    """
    AxialLOB model for limit order book prediction.

    Input shape: (batch_size, 1, time_steps, features)
    Output shape: (batch_size, output_size)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 3,
        c_in: int = 32,
        c_out: int = 32,
        c_final: int = 4,
        n_heads: int = 4,
        pool_kernel: Tuple[int, int] = (1, 4),
        pool_stride: Tuple[int, int] = (1, 4),
        sequence_length: int = 40,  # Added to calculate linear layer size
        dropout: float = 0.0,  # Added to match Config, unused in architecture
    ):
        super().__init__(input_size, output_size)

        self.c_in = c_in
        self.c_out = c_out
        self.c_final = c_final
        self.sequence_length = sequence_length

        # Dimensions for Axial Attention
        # Input is (N, 1, T, F). We permute to (N, 1, F, T) in forward.
        # So W = features (input_size), H = time_steps (sequence_length)
        W = input_size
        H = sequence_length

        self.CNN_in = nn.Conv2d(in_channels=1, out_channels=c_in, kernel_size=1)
        self.CNN_out = nn.Conv2d(in_channels=c_out, out_channels=c_final, kernel_size=1)
        self.CNN_res2 = nn.Conv2d(in_channels=c_out, out_channels=c_final, kernel_size=1)
        self.CNN_res1 = nn.Conv2d(in_channels=1, out_channels=c_out, kernel_size=1)

        self.norm = nn.BatchNorm2d(c_in)
        self.res_norm2 = nn.BatchNorm2d(c_final)
        self.res_norm1 = nn.BatchNorm2d(c_out)
        self.norm2 = nn.BatchNorm2d(c_final)

        # Note: In (N, C, F, T), F is Height, T is Width.
        # axial_height (flag=False) attends over Height (F), so dim should be W (input_size)
        # axial_width (flag=True) attends over Width (T), so dim should be H (sequence_length)
        self.axial_height_1 = GatedAxialAttention(c_out, c_out, n_heads, W, flag=False)
        self.axial_width_1 = GatedAxialAttention(c_out, c_out, n_heads, H, flag=True)
        self.axial_height_2 = GatedAxialAttention(c_out, c_out, n_heads, W, flag=False)
        self.axial_width_2 = GatedAxialAttention(c_out, c_out, n_heads, H, flag=True)

        self.activation = nn.ReLU()
        self.pooling = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

        # Calculate linear layer input size
        # Output of pooling is (N, c_final, W_out, H_out)
        # W_out = (W - kernel[0]) / stride[0] + 1
        # H_out = (H - kernel[1]) / stride[1] + 1
        # Assuming padding=0 (default for AvgPool2d)

        # For W (features), kernel is 1, stride is 1 (from (1, 4)) -> W_out = W
        # For H (time), kernel is 4, stride is 4 -> H_out = H / 4

        w_out = int((W - pool_kernel[0]) / pool_stride[0] + 1)
        h_out = int((H - pool_kernel[1]) / pool_stride[1] + 1)

        linear_input_size = c_final * w_out * h_out
        self.linear = nn.Linear(linear_input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # x shape: (batch, 1, time_steps, features)
        # AxialLOB expects (batch, 1, features, time_steps)
        x = x.permute(0, 1, 3, 2)

        # up branch
        # first convolution before the attention
        y = self.CNN_in(x)
        y = self.norm(y)
        y = self.activation(y)

        # attention mechanism through gated multi head axial layer
        y = self.axial_width_1(y)
        y = self.axial_height_1(y)

        # lower branch
        x_res = self.CNN_res1(x)
        x_res = self.res_norm1(x_res)
        x_res = self.activation(x_res)

        # first residual
        y = y + x_res
        z = y.detach().clone()

        # second axial layer
        y = self.axial_width_2(y)
        y = self.axial_height_2(y)

        # second convolution
        y = self.CNN_out(y)
        y = self.res_norm2(y)
        y = self.activation(y)

        # lower branch
        z = self.CNN_res2(z)
        z = self.norm2(z)
        z = self.activation(z)

        # second res connection
        y = y + z

        # final part
        y = self.pooling(y)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        # forecast_y = torch.softmax(y, dim=1) # Softmax is usually handled by CrossEntropyLoss
        return y
