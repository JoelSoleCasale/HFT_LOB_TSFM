"""
Embedding generation module

Provides base classes, concrete generators, and pipelines
for generating embeddings from extracted features.
"""

from .base import BaseEmbeddingGenerator
from .pipeline import EmbeddingPipeline
from .registry import EmbeddingGeneratorRegistry
from .aggregator import EmbeddingAggregator
from .utils import (
    compute_patch_statistics,
    compute_differenced_sequence,
    extract_feature_columns,
)
from .chronos import ChronosEmbeddingGenerator

__all__ = [
    "BaseEmbeddingGenerator",
    "EmbeddingPipeline",
    "EmbeddingGeneratorRegistry",
    "EmbeddingAggregator",
    "compute_patch_statistics",
    "compute_differenced_sequence",
    "extract_feature_columns",
    "ChronosEmbeddingGenerator",
]
