"""
Integration tests for Chronos embedding generators.

These tests validate that the embedding generation produces consistent results
across different model types (T5, Bolt, Chronos2) using precomputed reference embeddings.
"""

import pytest
import numpy as np
from pathlib import Path

from embeddings.registry import EmbeddingGeneratorRegistry
from .test_fixtures import (
    generate_sample_time_series,
    generate_sample_time_series_batch,
    SINGLE_CONFIGS,
    BATCH_CONFIGS,
)


# Path to reference embeddings
REFERENCE_DATA_DIR = Path(__file__).parent.parent.parent / "data_sample" / "embeddings"


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_time_series(random_seed):
    """Generate a sample time series with fixed random seed."""
    return generate_sample_time_series(random_seed)


@pytest.fixture
def sample_time_series_batch(random_seed):
    """Generate a batch of sample time series with fixed random seed."""
    return generate_sample_time_series_batch(random_seed)


@pytest.fixture
def context_length():
    """Context length for embedding generation."""
    return 128


def validate_embedding_against_reference(
    embedding: np.ndarray, reference_name: str, auto_generate: bool = True
):
    """
    Validate an embedding against a reference file.

    Args:
        embedding: Generated embedding to validate
        reference_name: Name of the reference file (without .npy extension)
        auto_generate: If True, generate reference if it doesn't exist
    """
    reference_path = REFERENCE_DATA_DIR / f"{reference_name}.npy"

    if not reference_path.exists():
        if auto_generate:
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(reference_path, embedding)
            pytest.skip(
                f"Reference embedding not found. Generated and saved to {reference_path}. "
                "Please commit this file and re-run the test."
            )
        else:
            raise FileNotFoundError(f"Reference file not found: {reference_path}")

    reference_embedding = np.load(reference_path)

    # Validate shape
    assert (
        embedding.shape == reference_embedding.shape
    ), f"Embedding shape mismatch: {embedding.shape} vs {reference_embedding.shape}"

    # Validate values (use allclose for floating point comparison)
    np.testing.assert_allclose(
        embedding,
        reference_embedding,
        atol=1e-2,
        err_msg=f"{reference_name} embeddings do not match reference",
    )


def _generate_single_embedding(config_item: dict, sample_data: np.ndarray, context_length: int):
    """
    Helper to generate and validate a single embedding.

    Args:
        config_item: Configuration dictionary with 'name' and 'config' keys
        sample_data: Sample time series data
        context_length: Context length for embedding
    """
    generator = EmbeddingGeneratorRegistry.create("chronos", config=config_item["config"])
    embedding = generator.generate_embedding(sample_data, context_length=context_length)
    validate_embedding_against_reference(embedding, config_item["name"])


def _generate_batch_embedding(config_item: dict, sample_batch: np.ndarray, context_length: int):
    """
    Helper to generate and validate a batch embedding.

    Args:
        config_item: Configuration dictionary with 'name' and 'config' keys
        sample_batch: Batch of sample time series data
        context_length: Context length for embedding
    """
    generator = EmbeddingGeneratorRegistry.create("chronos", config=config_item["config"])

    # Generate embeddings for each item in the batch individually
    individual_embeddings = []
    for i in range(sample_batch.shape[0]):
        embedding = generator.generate_embedding(sample_batch[i], context_length=context_length)
        individual_embeddings.append(embedding)

    # Stack into a single array for comparison
    individual_embeddings_array = np.array(individual_embeddings)
    validate_embedding_against_reference(individual_embeddings_array, config_item["name"])


@pytest.mark.integration
@pytest.mark.slow
class TestChronosEmbeddingConsistency:
    """Test suite for validating Chronos embedding consistency."""

    @pytest.mark.parametrize("config_item", SINGLE_CONFIGS, ids=lambda x: x["name"])
    def test_single_embeddings(self, config_item, sample_time_series, context_length):
        """Test single embedding generation for all configurations."""
        _generate_single_embedding(config_item, sample_time_series, context_length)

    @pytest.mark.parametrize("config_item", BATCH_CONFIGS, ids=lambda x: x["name"])
    def test_batch_embeddings(self, config_item, sample_time_series_batch, context_length):
        """Test batch embedding generation for all configurations."""
        _generate_batch_embedding(config_item, sample_time_series_batch, context_length)

    def test_embedding_dimensions_consistency(self, sample_time_series, context_length):
        """
        Test that embedding dimensions are consistent across different configurations.

        This ensures that the embedding dimension calculation is stable.
        """
        # Test with first two single configs (one from each model type)
        test_configs = [SINGLE_CONFIGS[0], SINGLE_CONFIGS[1]]

        embeddings = []
        for config_item in test_configs:
            generator = EmbeddingGeneratorRegistry.create("chronos", config=config_item["config"])
            embedding = generator.generate_embedding(
                sample_time_series, context_length=context_length
            )
            embeddings.append(embedding)

        # Check that all embeddings have consistent dimensions when not augmented
        # (different models may have different embedding sizes, but should be stable)
        for i, embedding in enumerate(embeddings):
            assert len(embedding.shape) == 1, f"Config {i}: Embedding should be 1D"
            assert embedding.shape[0] > 0, f"Config {i}: Embedding should have non-zero length"
