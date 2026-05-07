"""Embedding aggregation utilities."""

import torch


class EmbeddingAggregator:
    """Handles aggregation operations for embeddings along specified dimensions."""

    # Supported aggregation methods
    SUPPORTED_AGGREGATIONS = frozenset(["last", "mean", "max", "min", "concat"])

    def __init__(self, aggregation_method: str):
        """Initialize aggregator with specified method."""
        if not self.is_valid(aggregation_method):
            raise ValueError(
                f"Invalid aggregation method: {aggregation_method}. "
                f"Must be one of {sorted(self.SUPPORTED_AGGREGATIONS)}"
            )

        self.aggregation_method = aggregation_method

    def aggregate(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Aggregate tensor along the specified dimension.

        For "concat", the specified dimension is flattened with the next dimension (dim + 1).
        If dim is the last dimension, it's flattened with the previous dimension instead.
        """
        if self.aggregation_method == "last":
            return tensor.select(dim, -1)
        elif self.aggregation_method == "mean":
            return tensor.mean(dim=dim)
        elif self.aggregation_method == "max":
            return tensor.max(dim=dim)[0]
        elif self.aggregation_method == "min":
            return tensor.min(dim=dim)[0]
        elif self.aggregation_method == "concat":
            if dim < 0:
                dim = tensor.ndim + dim

            if dim == tensor.ndim - 1:
                flatten_start = dim - 1
                flatten_end = dim
            else:
                flatten_start = dim
                flatten_end = dim + 1

            return torch.flatten(tensor, start_dim=flatten_start, end_dim=flatten_end)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

    @classmethod
    def is_valid(cls, aggregation_method: str) -> bool:
        """Check if an aggregation method is valid."""
        return aggregation_method in cls.SUPPORTED_AGGREGATIONS
