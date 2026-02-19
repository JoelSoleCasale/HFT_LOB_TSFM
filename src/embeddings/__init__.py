"""
Embedding generation module

Provides base classes, concrete generators, and pipelines
for generating embeddings from extracted features.
"""

# Base classes
from .base import BaseEmbeddingGenerator

# Pipelines
from .pipeline import EmbeddingPipeline

# Registries
from .registry import EmbeddingGeneratorRegistry

# Aggregators
from .aggregator import EmbeddingAggregator

# Utilities
from .utils import (
    compute_patch_statistics,
    compute_differenced_sequence,
    extract_feature_columns,
)

# Generators (import to register them)
try:
    from .chronos import ChronosEmbeddingGenerator
except ImportError:
    # Chronos not available, but that's okay
    ChronosEmbeddingGenerator = None

__all__ = [
    # Base
    "BaseEmbeddingGenerator",
    # Pipelines
    "EmbeddingPipeline",
    # Registries
    "EmbeddingGeneratorRegistry",
    # Aggregators
    "EmbeddingAggregator",
    # Utilities
    "compute_patch_statistics",
    "compute_differenced_sequence",
    "extract_feature_columns",
    # Generators
    "ChronosEmbeddingGenerator",
]
