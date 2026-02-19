"""
Abstract base test class for data leakage testing in embedding generators.

This module provides a framework for ensuring that embedding generators
do not leak future information when generating embeddings for a given timestamp.
The test validates the causal property: embeddings at time t should only depend
on data from times <= t.

## Test Methodology

The data leakage tests work by creating two time series:
1. An original time series with data from t=0 to t=T
2. A modified time series that matches the original up to some cutoff point t=C,
   but has completely different values for t > C

For each cutoff point:
- We generate embeddings using both the full original and full modified series
- The embeddings MUST be identical up to position C (inclusive)
- These embeddings SHOULD differ after position C because they include the modified data

This validates two properties:
1. **Causality**: Embeddings at time t don't use future information
2. **Test sensitivity**: The test can actually detect differences when data changes

## Additional Tests

- **Context length isolation**: Verifies that only the most recent context_length
  samples affect the embedding, not older history
- **Augmentation features**: Tests that patch statistics and differencing features
  also respect causality

## Usage for Other Generators

To test a new embedding generator:
1. Create a test class inheriting from BaseDataLeakageTest
2. Implement the get_generator() method
3. The base class will automatically test for data leakage

Example:
```python
class TestMyGeneratorDataLeakage(BaseDataLeakageTest):
    def get_generator(self) -> BaseEmbeddingGenerator:
        return MyGenerator(config={...})
```
"""

import pytest
import numpy as np
from abc import ABC, abstractmethod

from embeddings.base import BaseEmbeddingGenerator


class BaseDataLeakageTest(ABC):
    """
    Abstract base class for data leakage tests.

    This class provides a template method for testing whether an embedding
    generator leaks future information. Subclasses should implement the
    get_generator method to specify which generator to test.
    """

    @abstractmethod
    def get_generator(self) -> BaseEmbeddingGenerator:
        """
        Get the embedding generator to test.

        Returns:
            BaseEmbeddingGenerator: The generator instance to test
        """
        pass

    def test_no_data_leakage(self):
        """
        Test that embeddings do not leak future information.

        This test verifies the causal property of embeddings: the embedding
        at position i should only depend on data from positions 0 to i (inclusive),
        not on any future data.

        The test works by:
        1. Generating embeddings for the full original time series
        2. For several cutoff points, creating a modified series that matches
           the original up to that point but differs afterwards
        3. Generating embeddings for the full modified time series
        4. Verifying that embeddings up to the cutoff point are identical
        5. Verifying that embeddings after the cutoff point are different

        This ensures that the embedding generator respects temporal ordering
        and does not use future information.
        """
        import polars as pl

        np.random.seed(42)
        generator = self.get_generator()

        # Generate random feature array
        seq_len = 256
        n_features = 5
        context_length = 64  # Use a smaller context length for testing

        # Create original time series
        original_data = np.cumsum(np.random.randn(seq_len, n_features) * 0.1, axis=0)
        original_data += np.array([100.0, 50.0, 25.0, 10.0, 5.0])
        original_data = original_data.astype(np.float32)

        # Create DataFrame for batch processing
        timestamps = np.arange(seq_len)
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        original_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                **{feature_cols[i]: original_data[:, i] for i in range(n_features)},
            }
        ).lazy()

        # Generate embeddings for the FULL original sequence
        original_embeddings_df = generator.generate_embeddings_batch(
            original_df, context_length
        ).collect()

        # Extract embedding column names (exclude timestamp)
        embedding_cols = [col for col in original_embeddings_df.columns if col != "timestamp"]

        # Test cutoff points where we have enough context
        test_positions = [
            context_length,  # First position with full context
            context_length + 32,  # Midway position
            context_length + 64,  # Another position
            seq_len - 32,  # Near the end
        ]

        for cutoff_idx in test_positions:
            # Create modified data: same up to cutoff, different after
            modified_data = original_data.copy()
            # Replace future data with completely different values
            np.random.seed(123 + cutoff_idx)  # Different seed for each cutoff
            modified_data[cutoff_idx + 1 :] = np.cumsum(
                np.random.randn(seq_len - cutoff_idx - 1, n_features) * 0.5, axis=0
            )
            modified_data[cutoff_idx + 1 :] += np.array([200.0, 100.0, 50.0, 20.0, 10.0])

            # Verify that data before cutoff is identical
            np.testing.assert_array_equal(
                original_data[: cutoff_idx + 1],
                modified_data[: cutoff_idx + 1],
                err_msg=f"Test setup error: data before cutoff {cutoff_idx} should be identical",
            )

            # Verify that data after cutoff is different
            if cutoff_idx + 1 < seq_len:
                assert not np.allclose(
                    original_data[cutoff_idx + 1 :],
                    modified_data[cutoff_idx + 1 :],
                    atol=1e-5,
                ), f"Test setup error: data after cutoff {cutoff_idx} should be different"

            # Create DataFrame for modified data
            modified_df = pl.DataFrame(
                {
                    "timestamp": timestamps,
                    **{feature_cols[i]: modified_data[:, i] for i in range(n_features)},
                }
            ).lazy()

            # Generate embeddings for the FULL modified sequence
            modified_embeddings_df = generator.generate_embeddings_batch(
                modified_df, context_length
            ).collect()

            # Extract embeddings at positions up to and including cutoff
            # Note: embeddings start at index context_length - 1
            if cutoff_idx >= context_length - 1:
                # Get embeddings for positions up to cutoff_idx
                original_embeddings_before = (
                    original_embeddings_df.filter(pl.col("timestamp") <= cutoff_idx)
                    .select(embedding_cols)
                    .to_numpy()
                )

                modified_embeddings_before = (
                    modified_embeddings_df.filter(pl.col("timestamp") <= cutoff_idx)
                    .select(embedding_cols)
                    .to_numpy()
                )

                # CRITICAL TEST: Embeddings up to cutoff should be IDENTICAL
                # because they only use data up to that position
                np.testing.assert_allclose(
                    original_embeddings_before,
                    modified_embeddings_before,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Data leakage detected at cutoff {cutoff_idx}! "
                    f"Embeddings at or before time {cutoff_idx} changed when future data "
                    f"(after index {cutoff_idx}) was modified. This indicates the "
                    f"generator is using future information.",
                )

            # Test that embeddings at future positions ARE different
            # This validates that our test is actually sensitive to changes
            if cutoff_idx + context_length < seq_len:
                future_start = cutoff_idx + context_length // 2

                # Get embeddings for positions after cutoff
                original_embeddings_after = (
                    original_embeddings_df.filter(pl.col("timestamp") >= future_start)
                    .select(embedding_cols)
                    .to_numpy()
                )

                modified_embeddings_after = (
                    modified_embeddings_df.filter(pl.col("timestamp") >= future_start)
                    .select(embedding_cols)
                    .to_numpy()
                )

                # These SHOULD be different (validates test sensitivity)
                # Use a more lenient check since some aggregation might reduce differences
                embeddings_are_different = not np.allclose(
                    original_embeddings_after, modified_embeddings_after, rtol=1e-3, atol=1e-4
                )

                assert embeddings_are_different, (
                    f"Test validation error at cutoff {cutoff_idx}: "
                    f"Embeddings at future positions (>= {future_start}) should differ "
                    f"when input data differs, but they are too similar. "
                    f"This suggests the test might not be sensitive enough or "
                    f"the embedding aggregation is too strong."
                )

    def test_context_length_isolation(self):
        """
        Test that only the most recent context_length samples affect the embedding.

        This test verifies that changing data outside the context window
        (more than context_length steps in the past) does not affect the
        current embedding.
        """
        np.random.seed(42)
        generator = self.get_generator()

        seq_len = 256
        n_features = 5
        context_length = 64

        # Create original time series
        original_data = np.cumsum(np.random.randn(seq_len, n_features) * 0.1, axis=0)
        original_data += np.array([100.0, 50.0, 25.0, 10.0, 5.0])
        original_data = original_data.astype(np.float32)

        # Pick a position with full context available
        test_position = seq_len - 1

        # Generate embedding at test position
        original_embedding = generator.generate_embedding(
            original_data[: test_position + 1], context_length=context_length
        )

        # Create modified data: change only the distant past (outside context window)
        modified_data = original_data.copy()
        # Modify data that's well outside the context window
        early_cutoff = test_position - context_length - 10
        if early_cutoff > 0:
            np.random.seed(999)
            modified_data[:early_cutoff] = np.cumsum(
                np.random.randn(early_cutoff, n_features) * 0.5, axis=0
            )
            modified_data[:early_cutoff] += np.array([150.0, 75.0, 35.0, 15.0, 8.0])

            # Verify the context window data is still identical
            context_start = test_position - context_length + 1
            np.testing.assert_array_equal(
                original_data[context_start : test_position + 1],
                modified_data[context_start : test_position + 1],
                err_msg="Test setup error: context window should be identical",
            )

            # Generate embedding with modified history
            modified_embedding = generator.generate_embedding(
                modified_data[: test_position + 1], context_length=context_length
            )

            # Embeddings should be IDENTICAL because the context window hasn't changed
            np.testing.assert_allclose(
                original_embedding,
                modified_embedding,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Context window isolation violated! "
                f"Embedding changed when data outside the context window "
                f"(before index {context_start}) was modified. "
                f"This suggests the generator is using more history than specified by context_length.",
            )


@pytest.mark.integration
@pytest.mark.slow
class TestChronosDataLeakage(BaseDataLeakageTest):
    """
    Test data leakage for Chronos embedding generators.

    This test suite validates that Chronos models (T5, Bolt, Chronos2)
    do not leak future information when generating embeddings.
    """

    @pytest.fixture
    def generator_config(self):
        """
        Configuration for Chronos generator to test.

        Uses a lightweight configuration (bolt-mini on CPU) for faster testing.
        """
        return {
            "model_type": "bolt",
            "model_size": "mini",
            "device": "cpu",
            "seq_aggregation": "last",
            "feat_aggregation": "concat",
            "augment_with_statistics": False,
            "use_differencing": False,
            "disable_tqdm": True,
        }

    def get_generator(self) -> BaseEmbeddingGenerator:
        """Get Chronos generator for testing."""
        from embeddings.registry import EmbeddingGeneratorRegistry

        return EmbeddingGeneratorRegistry.create("chronos", config=self._generator_config)

    def test_no_data_leakage(self, generator_config):
        """Test that Chronos embeddings do not leak future information."""
        # Override to inject fixture
        self._generator_config = generator_config
        super().test_no_data_leakage()

    def test_context_length_isolation(self, generator_config):
        """Test that only recent context affects Chronos embeddings."""
        # Override to inject fixture
        self._generator_config = generator_config
        super().test_context_length_isolation()

    @pytest.mark.parametrize(
        "model_config",
        [
            {
                "model_type": "bolt",
                "model_size": "mini",
                "seq_aggregation": "mean",
                "feat_aggregation": "mean",
            },
            {
                "model_type": "chronos2",
                "model_size": "base",
                "seq_aggregation": "last",
                "feat_aggregation": "concat",
            },
        ],
        ids=["bolt-mean-aggregation", "chronos2"],
    )
    def test_different_configurations_no_leakage(self, model_config):
        """
        Test data leakage for different Chronos configurations.

        This test validates that various model types and aggregation strategies
        maintain the causal property.
        """
        config = {
            **model_config,
            "device": "cpu",
            "augment_with_statistics": False,
            "use_differencing": False,
            "disable_tqdm": True,
        }

        self._generator_config = config

        # Run the base data leakage test
        super().test_no_data_leakage()

    @pytest.mark.parametrize(
        "model_config",
        [
            {
                "model_type": "bolt",
                "model_size": "mini",
                "seq_aggregation": "last",
                "feat_aggregation": "concat",
            },
        ],
        ids=["bolt-last-aggregation"],
    )
    def test_augmentation_no_leakage(self, model_config):
        """
        Test that augmentation features (statistics, differencing) do not leak data.

        This is especially important because patch statistics and differencing
        could potentially use future information if implemented incorrectly.
        """
        config = {
            **model_config,
            "device": "cpu",
            "augment_with_statistics": True,
            "k": 8,
            "use_differencing": True,
            "disable_tqdm": True,
        }

        self._generator_config = config

        # Run the base data leakage test with augmentation enabled
        super().test_no_data_leakage()
        super().test_context_length_isolation()
