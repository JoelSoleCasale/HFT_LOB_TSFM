"""
Shared fixtures and configurations for Chronos embedding tests.

This module provides reusable test data generation and configuration definitions
to avoid duplication across test files and the reference generation script.
"""

import numpy as np


def generate_sample_time_series(seed: int = 42):
    """
    Generate a sample time series with fixed random seed.

    Args:
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (seq_len=512, n_features=5)
    """
    np.random.seed(seed)
    seq_len = 512
    n_features = 5

    # Generate realistic-looking time series data
    # Use cumulative sum of random values to create trends
    data = np.cumsum(np.random.randn(seq_len, n_features) * 0.1, axis=0)

    # Add some baseline values
    data += np.array([100.0, 50.0, 25.0, 10.0, 5.0])

    return data.astype(np.float32)


def generate_sample_time_series_batch(seed: int = 42):
    """
    Generate a batch of sample time series with fixed random seed.

    Args:
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Shape (batch_size=4, seq_len=512, n_features=5)
    """
    batch_size = 4
    seq_len = 512
    n_features = 5

    # Generate multiple time series with different characteristics
    batch_data = []
    for i in range(batch_size):
        # Use different seed per batch item for variety
        np.random.seed(seed + i)
        data = np.cumsum(np.random.randn(seq_len, n_features) * 0.1, axis=0)
        data += np.array([100.0, 50.0, 25.0, 10.0, 5.0]) * (1 + i * 0.1)
        batch_data.append(data.astype(np.float32))

    return np.array(batch_data)


# Base configuration templates
BASE_CONFIG_CPU = {
    "device": "cpu",
    "disable_tqdm": True,
}

# Single embedding test configurations
SINGLE_CONFIGS = [
    {
        "name": "chronos_bolt_mini_reference",
        "config": {
            **BASE_CONFIG_CPU,
            "model_type": "bolt",
            "model_size": "mini",
            "seq_aggregation": "last",
            "feat_aggregation": "concat",
            "augment_with_statistics": False,
            "use_differencing": False,
        },
    },
    {
        "name": "chronos2_reference",
        "config": {
            **BASE_CONFIG_CPU,
            "model_type": "chronos2",
            "model_size": "base",
            "seq_aggregation": "last",
            "feat_aggregation": "concat",
            "augment_with_statistics": False,
            "use_differencing": False,
        },
    },
    {
        "name": "chronos_bolt_mini_augmented_reference",
        "config": {
            **BASE_CONFIG_CPU,
            "model_type": "bolt",
            "model_size": "mini",
            "seq_aggregation": "mean",
            "feat_aggregation": "mean",
            "augment_with_statistics": True,
            "k": 8,
            "use_differencing": True,
        },
    },
    {
        "name": "chronos2_augmented_reference",
        "config": {
            **BASE_CONFIG_CPU,
            "model_type": "chronos2",
            "model_size": "base",
            "seq_aggregation": "mean",
            "feat_aggregation": "mean",
            "augment_with_statistics": True,
            "k": 8,
            "use_differencing": True,
        },
    },
]

# Batch processing test configurations
BATCH_CONFIGS = [
    {
        "name": "chronos_bolt_mini_batch_reference",
        "config": {
            **BASE_CONFIG_CPU,
            "model_type": "bolt",
            "model_size": "mini",
            "seq_aggregation": "last",
            "feat_aggregation": "concat",
            "augment_with_statistics": False,
            "use_differencing": False,
        },
    },
    {
        "name": "chronos2_batch_reference",
        "config": {
            **BASE_CONFIG_CPU,
            "model_type": "chronos2",
            "model_size": "base",
            "seq_aggregation": "last",
            "feat_aggregation": "concat",
            "augment_with_statistics": False,
            "use_differencing": False,
        },
    },
    {
        "name": "chronos_bolt_mini_batch_augmented_reference",
        "config": {
            **BASE_CONFIG_CPU,
            "model_type": "bolt",
            "model_size": "mini",
            "seq_aggregation": "mean",
            "feat_aggregation": "mean",
            "augment_with_statistics": True,
            "k": 8,
            "use_differencing": True,
        },
    },
    {
        "name": "chronos2_batch_augmented_reference",
        "config": {
            **BASE_CONFIG_CPU,
            "model_type": "chronos2",
            "model_size": "base",
            "seq_aggregation": "mean",
            "feat_aggregation": "mean",
            "augment_with_statistics": True,
            "k": 8,
            "use_differencing": True,
        },
    },
]
