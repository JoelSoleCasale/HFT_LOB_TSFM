"""Embedding aggregation utilities."""

import torch


class EmbeddingAggregator:
    """Handles aggregation operations for embeddings along specified dimensions."""

    # Supported aggregation methods
    SUPPORTED_AGGREGATIONS = frozenset(["last", "mean", "max", "min", "concat"])

    def __init__(self, aggregation_method: str):
        """
        Initialize aggregator with specified method.

        Args:
            aggregation_method: Aggregation method to use
                - "last": Take the last element along the dimension
                - "mean": Average across the dimension
                - "max": Maximum value across the dimension
                - "min": Minimum value across the dimension
                - "concat": Flatten the dimension with the next dimension
        """
        if not self.is_valid(aggregation_method):
            raise ValueError(
                f"Invalid aggregation method: {aggregation_method}. "
                f"Must be one of {sorted(self.SUPPORTED_AGGREGATIONS)}"
            )

        self.aggregation_method = aggregation_method

    def aggregate(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Aggregate tensor along the specified dimension.

        Args:
            tensor: Input tensor
            dim: Dimension to aggregate along (can be negative for indexing from end)

        Returns:
            Aggregated tensor with the specified dimension reduced or flattened

        Note:
            For "concat" aggregation, the specified dimension is flattened with the
            next dimension (dim + 1). If dim is the last dimension, it's flattened
            with the previous dimension instead.
        """
        if self.aggregation_method == "last":
            # Select last element along dimension
            return tensor.select(dim, -1)
        elif self.aggregation_method == "mean":
            return tensor.mean(dim=dim)
        elif self.aggregation_method == "max":
            return tensor.max(dim=dim)[0]
        elif self.aggregation_method == "min":
            return tensor.min(dim=dim)[0]
        elif self.aggregation_method == "concat":
            # Flatten specified dimension with the next dimension
            # Normalize negative dimension
            if dim < 0:
                dim = tensor.ndim + dim

            # Determine which dimensions to flatten
            if dim == tensor.ndim - 1:
                # If last dimension, flatten with previous
                flatten_start = dim - 1
                flatten_end = dim
            else:
                # Otherwise flatten with next
                flatten_start = dim
                flatten_end = dim + 1

            # Use torch.flatten to combine the dimensions
            return torch.flatten(tensor, start_dim=flatten_start, end_dim=flatten_end)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

    @classmethod
    def is_valid(cls, aggregation_method: str) -> bool:
        """
        Check if an aggregation method is valid.

        Args:
            aggregation_method: Method to validate

        Returns:
            True if valid, False otherwise
        """
        return aggregation_method in cls.SUPPORTED_AGGREGATIONS
