"""
PCA dimensionality reduction for embeddings.

This module provides tools for applying incremental PCA to large embedding datasets
that don't fit in memory, using Polars LazyFrames for efficient processing.
"""

from embeddings.pca.config import PCAConfig
from embeddings.pca.incremental_pca import IncrementalPCAProcessor

__all__ = ["PCAConfig", "IncrementalPCAProcessor"]
