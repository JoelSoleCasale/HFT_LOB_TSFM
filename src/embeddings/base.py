"""Base classes for embedding generators"""

from abc import ABC, abstractmethod
from typing import Any
import polars as pl
import numpy as np


class BaseEmbeddingGenerator(ABC):
    """Abstract base class for all embedding generators"""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the embedding generator.

        Args:
            config: Configuration dictionary for the generator
        """
        self.config = config or {}
        self.embedding_dim: int | None = None

    @abstractmethod
    def generate_embedding(
        self, features: np.ndarray, context_length: int | None = None
    ) -> np.ndarray:
        """
        Generate embedding from feature array.

        Args:
            features: Feature array of shape (seq_len, n_features)
            context_length: Optional context length to use (if None, uses all available data)

        Returns:
            Embedding array of shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def generate_embeddings_batch(
        self, features_df: pl.LazyFrame, context_length: int
    ) -> pl.LazyFrame:
        """
        Generate embeddings for all positions in a feature dataframe.

        Args:
            features_df: LazyFrame with time series features (must have 'timestamp' column)
            context_length: Number of samples to use as context for each embedding

        Returns:
            LazyFrame with embeddings for each position, indexed by timestamp
        """
        pass

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by this generator.

        Returns:
            Embedding dimension
        """
        if self.embedding_dim is None:
            raise ValueError(
                "Embedding dimension not set. Call generate_embedding or generate_embeddings_batch first."
            )
        return self.embedding_dim

    def validate_config(self) -> bool:
        """
        Validate the configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        return True
