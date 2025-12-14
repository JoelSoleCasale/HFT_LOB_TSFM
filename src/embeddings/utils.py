"""Utility functions for embedding generation"""

import numpy as np
import polars as pl
from typing import Literal


def compute_patch_statistics(data: np.ndarray, k: int, normalize: bool = True) -> np.ndarray:
    """
    Compute mean, std, min, and max for k non-overlapping patches of the input data.

    Args:
        data: Input array of shape (seq_len, n_features)
        k: Number of non-overlapping patches
        normalize: Whether to normalize data before computing statistics (default: True)

    Returns:
        Array of shape (k * 4 * n_features,) containing statistics for all patches
    """
    if data.size == 0:
        raise ValueError("Input data cannot be empty")

    # Normalize data if requested
    if normalize:
        data_mean = data.mean(axis=0, keepdims=True)
        data_std = data.std(axis=0, keepdims=True)
        # Avoid division by zero
        data_std = np.where(data_std == 0, 1, data_std)
        data = (data - data_mean) / data_std

    seq_len, n_features = data.shape
    patch_size = seq_len // k

    if patch_size == 0:
        raise ValueError(f"Sequence length ({seq_len}) must be at least k ({k})")

    # Truncate data to ensure equal-sized patches
    truncated_len = patch_size * k
    data = data[:truncated_len]

    # Reshape into patches: (k, patch_size, n_features)
    patches = data.reshape(k, patch_size, n_features)

    # Compute statistics for each patch
    means = patches.mean(axis=1)  # (k, n_features)
    stds = patches.std(axis=1)  # (k, n_features)
    mins = patches.min(axis=1)  # (k, n_features)
    maxs = patches.max(axis=1)  # (k, n_features)

    # Concatenate all statistics and flatten
    statistics = np.concatenate([means, stds, mins, maxs], axis=0)  # (4*k, n_features)

    return statistics.flatten()  # (4*k*n_features,)


def compute_differenced_sequence(data: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Compute differenced time series.

    Args:
        data: Input array of shape (seq_len, n_features)
        order: Order of differencing (default: 1)

    Returns:
        Differenced array of shape (seq_len - order, n_features)
    """
    if order < 1:
        raise ValueError("Differencing order must be at least 1")

    if len(data) <= order:
        raise ValueError(f"Sequence length ({len(data)}) must be greater than order ({order})")

    result = data.copy()
    for _ in range(order):
        result = np.diff(result, axis=0)

    return result


def aggregate_sequence(
    data: np.ndarray, method: Literal["last", "mean", "max", "min"] = "last"
) -> np.ndarray:
    """
    Aggregate a sequence along the time dimension.

    Args:
        data: Input array of shape (seq_len, n_features)
        method: Aggregation method ("last", "mean", "max", "min")

    Returns:
        Aggregated array of shape (n_features,)
    """
    if method == "last":
        return data[-1]
    elif method == "mean":
        return data.mean(axis=0)
    elif method == "max":
        return data.max(axis=0)
    elif method == "min":
        return data.min(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def extract_feature_columns(df: pl.LazyFrame, exclude_cols: list[str] | None = None) -> list[str]:
    """
    Extract feature column names from a LazyFrame, excluding specified columns.

    Args:
        df: Input LazyFrame
        exclude_cols: List of column names to exclude (default: ["timestamp", "time"])

    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ["timestamp", "time"]

    schema = df.collect_schema()
    return [col for col in schema.keys() if col not in exclude_cols]
