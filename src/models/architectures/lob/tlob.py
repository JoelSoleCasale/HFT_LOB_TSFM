"""
TLOB: Transformer for Limit Order Book

Reference:
  Berti, L., Rognone, L., & Macchiavello, A. (2025).
  TLOB: Transformer for Limit Order Book
  arXiv: https://arxiv.org/pdf/2502.15757

Architecture:
  - Bilinear Normalization (BiN) for input features
  - Alternating feature-wise and temporal-wise transformer layers
  - Sinusoidal or learned positional embeddings
  - Final MLP projection head
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from einops import rearrange

from ..base import ModelArchitectureConfig, FinancialTimeSeriesModel


class BiN(nn.Module):
    """
    Bilinear Normalization layer for LOB data.

    Normalizes along both feature and temporal dimensions.
    """

    def __init__(self, num_features: int, seq_size: int):
        super().__init__()
        self.num_features = num_features
        self.seq_size = seq_size
        # Learnable parameters for normalization
        self.gamma = nn.Parameter(torch.ones(1, num_features, seq_size))
        self.beta = nn.Parameter(torch.zeros(1, num_features, seq_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, num_features, seq_size)

        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance across batch dimension
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        # Apply affine transformation
        return self.gamma * x_norm + self.beta


class ComputeQKV(nn.Module):
    """Compute Query, Key, Value projections for multi-head attention."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


class TransformerLayer(nn.Module):
    """
    Transformer layer with multi-head attention and feed-forward network.

    Can operate on either feature or temporal dimension.
    """

    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(hidden_dim * num_heads, num_heads, batch_first=True)
        # MLP: hidden_dim -> hidden_dim*4 -> final_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, final_dim),
        )
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim)

        Returns:
            Transformed tensor and attention weights
        """
        res = x
        q, k, v = self.qkv(x)
        x, att = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        x = self.w0(x)
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        # Residual connection if dimensions match
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, att


def sinusoidal_positional_embedding(
    token_sequence_size: int, token_embedding_dim: int, n: float = 10000.0
) -> torch.Tensor:
    """
    Generate sinusoidal positional embeddings.

    Args:
        token_sequence_size: Sequence length
        token_embedding_dim: Embedding dimension (must be even)
        n: Base for positional encoding

    Returns:
        Positional embeddings of shape (token_sequence_size, token_embedding_dim)
    """
    if token_embedding_dim % 2 != 0:
        raise ValueError(
            f"Sinusoidal positional embedding requires even embedding dimension "
            f"(got dim={token_embedding_dim})"
        )

    T = token_sequence_size
    d = token_embedding_dim

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    # 10000^(2i/d_model), i is the index of embedding
    denominators = torch.pow(n, 2 * torch.arange(0, d // 2) / d)
    embeddings[:, 0::2] = torch.sin(positions / denominators)
    embeddings[:, 1::2] = torch.cos(positions / denominators)

    return embeddings


@dataclass
class TLOBConfig(ModelArchitectureConfig):
    """Configuration for TLOB model."""

    model_type: str = "tlob"
    # Hidden dimension for transformer
    hidden_dim: int = 64
    # Number of transformer layer pairs (feature + temporal)
    num_layers: int = 2
    # Number of attention heads
    num_heads: int = 4
    # Use sinusoidal positional embeddings (True) or learned (False)
    is_sin_emb: bool = True
    # Dropout rate
    dropout: float = 0.0

    sequence_length: int = 100


class TLOBModel(FinancialTimeSeriesModel):
    """
    TLOB: Transformer for Limit Order Book.

    Alternates between feature-wise and temporal-wise transformer layers,
    processing LOB data with bilinear normalization and positional embeddings.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        is_sin_emb: bool = True,
        dropout: float = 0.0,
        sequence_length: int = 100,
    ):
        """
        Args:
            input_size: Number of input features (e.g., 40 for 10 LOB levels)
            output_size: Number of output classes
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layer pairs
            num_heads: Number of attention heads
            is_sin_emb: Use sinusoidal (True) or learned (False) positional embeddings
            dropout: Dropout rate
            sequence_length: Temporal sequence length
        """
        super().__init__(input_size, output_size)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_sin_emb = is_sin_emb
        self.seq_size = sequence_length
        self.num_heads = num_heads
        self.num_features = input_size
        self.dropout = dropout

        # Bilinear normalization layer
        self.norm_layer = BiN(input_size, sequence_length)

        # Feature embedding layer
        self.emb_layer = nn.Linear(input_size, hidden_dim)

        # Positional encoding
        if is_sin_emb:
            self.register_buffer(
                "pos_encoder",
                sinusoidal_positional_embedding(sequence_length, hidden_dim),
            )
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, sequence_length, hidden_dim))

        # Transformer layers (alternating feature and temporal)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                # Feature-wise transformer
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                # Temporal-wise transformer
                self.layers.append(TransformerLayer(sequence_length, num_heads, sequence_length))
            else:
                # Last layer: reduce dimensions
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim // 4))
                self.layers.append(
                    TransformerLayer(sequence_length, num_heads, sequence_length // 4)
                )

        # Final projection layers
        total_dim = (hidden_dim // 4) * (sequence_length // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)

        Returns:
            Logits of shape (batch, output_size)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, sequence, features), got shape {x.shape}")

        batch_size, seq_len, features = x.shape
        if seq_len != self.seq_size:
            raise ValueError(f"Expected sequence length {self.seq_size}, got {seq_len}")
        if features != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {features}")

        x = rearrange(x, "b s f -> b f s")
        x = self.norm_layer(x)
        x = rearrange(x, "b f s -> b s f")

        x = self.emb_layer(x)

        if self.is_sin_emb:
            x = x + self.pos_encoder.unsqueeze(0)
        else:
            x = x + self.pos_encoder

        for i in range(len(self.layers)):
            x, att = self.layers[i](x)
            # Transpose for next layer (alternates between feature and temporal processing)
            x = x.permute(0, 2, 1)

        x = rearrange(x, "b s f -> b (f s)")

        for layer in self.final_layers:
            x = layer(x)

        return x
