"""Chronos embedding generator implementation"""

from typing import Any, Literal
import numpy as np
import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from embeddings.base import BaseEmbeddingGenerator
from embeddings.registry import EmbeddingGeneratorRegistry
from embeddings.utils import (
    compute_patch_statistics,
    compute_differenced_sequence,
    extract_feature_columns,
)

from chronos import ChronosPipeline, BaseChronosPipeline, Chronos2Pipeline


# Cache for Chronos pipelines by model size
_PIPELINE_CACHE: dict[str, Any] = {}


@EmbeddingGeneratorRegistry.register("chronos")
class ChronosEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Embedding generator using Chronos time series foundation models.

    Supports T5, Bolt, and Chronos 2 model types, various aggregation methods,
    and augmentation with statistics and differencing.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize Chronos embedding generator.

        Args:
            config: Configuration dictionary with the following keys:
                - model_type: Type of Chronos model ("t5", "bolt", or "chronos2", default: "t5")
                - model_size: Size of the model ("mini", "small", "base", "large", default: "mini")
                  Note: For chronos2, only base size is available (model_size is ignored)
                - seq_aggregation: How to aggregate across sequence ("last" or "mean", default: "last")
                - feat_aggregation: How to aggregate across features ("concat" or "mean", default: "concat")
                - augment_with_statistics: Whether to augment with patch statistics (default: False)
                - k: Number of patches for statistics (default: 8)
                - use_differencing: Whether to augment with differenced embeddings (default: False)
                - device: Device to use ("cuda" or "cpu", default: "cuda")
                - batch_size: Number of samples to process in parallel (default: 32)
                - disable_tqdm: Whether to disable progress bars (default: False)
        """
        super().__init__(config)

        # Set defaults
        self.model_type: str = self.config.get("model_type", "t5")
        self.model_size: str = self.config.get("model_size", "mini")
        self.seq_aggregation: Literal["last", "mean"] = self.config.get("seq_aggregation", "last")
        self.feat_aggregation: Literal["concat", "mean"] = self.config.get(
            "feat_aggregation", "concat"
        )
        self.augment_with_statistics: bool = self.config.get("augment_with_statistics", False)
        self.k: int = self.config.get("k", 8)
        self.use_differencing: bool = self.config.get("use_differencing", False)
        self.device: str = self.config.get("device", "cuda")
        self.batch_size: int = self.config.get("batch_size", 32)
        self.disable_tqdm: bool = self.config.get("disable_tqdm", False)

        # Validate configuration
        self.validate_config()

        # Initialize pipeline (lazy loading)
        self._pipeline: Any = None

    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        if self.model_type not in ["t5", "bolt", "chronos2"]:
            raise ValueError(
                f"Invalid model_type: {self.model_type}. Must be 't5', 'bolt', or 'chronos2'"
            )

        if self.model_type != "chronos2" and self.model_size not in [
            "mini",
            "small",
            "base",
            "large",
        ]:
            raise ValueError(
                f"Invalid model_size: {self.model_size}. "
                "Must be 'mini', 'small', 'base', or 'large'"
            )

        if self.seq_aggregation not in ["last", "mean"]:
            raise ValueError(
                f"Invalid seq_aggregation: {self.seq_aggregation}. Must be 'last' or 'mean'"
            )

        if self.feat_aggregation not in ["concat", "mean"]:
            raise ValueError(
                f"Invalid feat_aggregation: {self.feat_aggregation}. Must be 'concat' or 'mean'"
            )

        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")

        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}. Must be 'cuda' or 'cpu'")

        return True

    @property
    def pipeline(self):
        """Lazy load and cache the Chronos pipeline."""
        if self._pipeline is None:
            cache_key = f"{self.model_type}_{self.model_size}_{self.device}"
            if cache_key not in _PIPELINE_CACHE:
                if self.model_type == "chronos2":
                    logger.info(f"Loading Chronos 2 on {self.device}")
                    _PIPELINE_CACHE[cache_key] = Chronos2Pipeline.from_pretrained(
                        "amazon/chronos-2",
                        device_map=self.device,
                    )
                elif self.model_type == "bolt":
                    logger.info(
                        f"Loading Chronos {self.model_type}-{self.model_size} on {self.device}"
                    )
                    _PIPELINE_CACHE[cache_key] = BaseChronosPipeline.from_pretrained(
                        f"amazon/chronos-bolt-{self.model_size}",
                        device_map=self.device,
                        dtype=torch.bfloat16,
                    )
                else:  # t5
                    logger.info(
                        f"Loading Chronos {self.model_type}-{self.model_size} on {self.device}"
                    )
                    _PIPELINE_CACHE[cache_key] = ChronosPipeline.from_pretrained(
                        f"amazon/chronos-t5-{self.model_size}",
                        device_map=self.device,
                        dtype=torch.bfloat16,
                    )
            self._pipeline = _PIPELINE_CACHE[cache_key]
        return self._pipeline

    def _embed_with_chronos(self, context: torch.Tensor) -> torch.Tensor:
        """
        Wrapper to call pipeline.embed() with consistent output format.

        For Chronos 1 (T5/Bolt): Flattens batch dimension to treat each feature independently,
        then reshapes output to match Chronos 2 format.

        For Chronos 2: Passes input directly.

        Args:
            context: Input tensor of shape (batch_size, n_features, seq_len)

        Returns:
            Embeddings tensor of shape (batch_size, n_features, new_seq_len, embedding_size)
        """
        if self.model_type == "chronos2":
            # Chronos 2: Input shape (batch_size, n_features, seq_len)
            # Output is a list of tensors, convert to single tensor
            # Output shape: (batch_size, n_features, new_seq_len, embedding_size)
            embeddings_list, _ = self.pipeline.embed(context)
            embeddings = (
                torch.stack(embeddings_list)
                if isinstance(embeddings_list, list)
                else embeddings_list
            )
            return embeddings
        else:
            # Chronos 1 (T5/Bolt): Each feature is treated as an independent batch element
            batch_size, n_features, seq_len = context.shape

            # Flatten to (batch_size * n_features, seq_len)
            context_flat = context.reshape(batch_size * n_features, seq_len)

            # Get embeddings: (batch_size * n_features, new_seq_len, embedding_size)
            embeddings_flat, _ = self.pipeline.embed(context_flat)

            # Reshape to match Chronos 2 format: (batch_size, n_features, new_seq_len, embedding_size)
            _, new_seq_len, embedding_size = embeddings_flat.shape
            embeddings = embeddings_flat.reshape(
                batch_size, n_features, new_seq_len, embedding_size
            )

            return embeddings

    def _process_embedding_batch(self, data_batch: np.ndarray) -> torch.Tensor:
        """
        Generate and process embeddings from a batch of data.

        Args:
            data_batch: Input array of shape (batch_size, seq_len, n_features)

        Returns:
            Processed embedding tensor of shape (batch_size, embedding_dim)
        """
        batch_size = data_batch.shape[0]

        # Convert to tensor with shape (batch_size, n_features, seq_len)
        # Note: data_batch is (batch_size, seq_len, n_features)
        context = torch.tensor(
            np.transpose(data_batch, (0, 2, 1)), dtype=torch.float32
        )  # (batch_size, n_features, seq_len)

        # Get embeddings: (batch_size, n_features, new_seq_len, embedding_size)
        embeddings = self._embed_with_chronos(context)

        # Aggregate across sequence dimension using torch operations
        # Shape: (batch_size, n_features, new_seq_len, embedding_size)
        if self.seq_aggregation == "last":
            emb: torch.Tensor = embeddings[:, :, -1, :]  # (batch_size, n_features, embedding_size)
        elif self.seq_aggregation == "mean":
            emb = embeddings.mean(dim=2)  # (batch_size, n_features, embedding_size)
        else:
            raise ValueError(f"Invalid sequential aggregation: {self.seq_aggregation}")

        # Aggregate across feature dimension
        if self.feat_aggregation == "concat":
            # Flatten feature and embedding dimensions for each batch element
            emb = emb.reshape(batch_size, -1)  # (batch_size, n_features * embedding_size)
        elif self.feat_aggregation == "mean":
            emb = emb.mean(dim=1)  # (batch_size, embedding_size)
        else:
            raise ValueError(f"Invalid feature aggregation: {self.feat_aggregation}")

        return emb.float()

    def _process_embedding_single(self, data: np.ndarray) -> torch.Tensor:
        """
        Generate and process embedding from a single data sample.

        Args:
            data: Input array of shape (seq_len, n_features)

        Returns:
            Processed embedding tensor of shape (embedding_dim,)
        """
        # Add batch dimension and process
        data_batch = np.expand_dims(data, axis=0)  # (1, seq_len, n_features)
        emb_batch = self._process_embedding_batch(data_batch)  # (1, embedding_dim)
        return emb_batch.squeeze(0)  # (embedding_dim,)

    def generate_embedding(
        self, features: np.ndarray, context_length: int | None = None
    ) -> np.ndarray:
        """
        Generate embedding from feature array.

        Args:
            features: Feature array of shape (seq_len, n_features)
            context_length: Optional context length to use (if None, uses all available data)

        Returns:
            Embedding array
        """
        if len(features) == 0:
            raise ValueError("Features array cannot be empty")

        # Truncate to context length if specified
        if context_length is not None:
            features = features[-context_length:]

        # Generate embedding from original time series
        embedding = self._process_embedding_single(features).cpu().numpy()

        # Augment with differenced embedding if enabled
        if self.use_differencing and len(features) > 1:
            differenced_data = compute_differenced_sequence(features, order=1)
            differenced_embedding = self._process_embedding_single(differenced_data).cpu().numpy()
            embedding = np.concatenate([embedding, differenced_embedding])

        # Augment with patch statistics if enabled
        if self.augment_with_statistics and len(features) >= self.k:
            patch_stats = compute_patch_statistics(features, self.k, normalize=True)
            embedding = np.concatenate([embedding, patch_stats])

        # Set embedding dimension if not set
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)

        return embedding

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
        if context_length < 1:
            raise ValueError(f"context_length must be at least 1, got {context_length}")

        # Collect to get schema and validate
        features_collected = features_df.collect()

        if "timestamp" not in features_collected.columns:
            raise ValueError("Features dataframe must have a 'timestamp' column")

        # Extract feature columns (exclude timestamp)
        feature_cols = extract_feature_columns(features_collected.lazy())

        if len(feature_cols) == 0:
            raise ValueError("No feature columns found in dataframe")

        # Convert to numpy for processing
        timestamps = features_collected["timestamp"].to_numpy()
        feature_data = features_collected.select(feature_cols).to_numpy()

        # Calculate start index: we need at least context_length samples
        start_index = context_length - 1  # 0-based index

        if start_index >= len(feature_data):
            raise ValueError(
                f"Not enough data: need at least {context_length} samples, "
                f"but only have {len(feature_data)}"
            )

        # Prepare all context windows
        all_contexts = []
        valid_indices = []

        for i in range(start_index, len(feature_data)):
            context_data = feature_data[i - context_length + 1 : i + 1]
            all_contexts.append(context_data)
            valid_indices.append(i)

        # Process in batches for efficiency
        embeddings_list = []

        for batch_start in tqdm(
            range(0, len(all_contexts), self.batch_size),
            desc="Generating embeddings",
            leave=False,
            disable=self.disable_tqdm,
            total=len(all_contexts),
            unit="sample",
            unit_scale=self.batch_size,
        ):
            batch_end = min(batch_start + self.batch_size, len(all_contexts))
            batch_contexts = all_contexts[batch_start:batch_end]

            # Stack contexts into a batch: (batch_size, seq_len, n_features)
            batch_array = np.stack(batch_contexts, axis=0)

            try:
                # Process batch of original time series
                batch_embeddings = self._process_embedding_batch(batch_array).cpu().numpy()

                # Handle differencing if enabled
                if self.use_differencing and context_length > 1:
                    # Compute differenced sequences for the batch
                    differenced_batch = np.array(
                        [compute_differenced_sequence(ctx, order=1) for ctx in batch_contexts]
                    )
                    diff_embeddings = (
                        self._process_embedding_batch(differenced_batch).cpu().numpy()
                    )
                    batch_embeddings = np.concatenate([batch_embeddings, diff_embeddings], axis=1)

                # Handle statistics augmentation if enabled
                if self.augment_with_statistics and context_length >= self.k:
                    # Compute patch statistics for each sample in the batch
                    stats_batch = np.array(
                        [
                            compute_patch_statistics(ctx, self.k, normalize=True)
                            for ctx in batch_contexts
                        ]
                    )
                    batch_embeddings = np.concatenate([batch_embeddings, stats_batch], axis=1)

                # Add batch embeddings to the list
                embeddings_list.extend(batch_embeddings)

            except Exception as e:
                logger.warning(
                    f"Failed to generate embeddings for batch starting at {batch_start}: {e}"
                )
                # Fall back to processing individually for this batch
                for ctx in batch_contexts:
                    try:
                        embedding = self.generate_embedding(ctx, context_length=context_length)
                        embeddings_list.append(embedding)
                    except Exception as e2:
                        logger.warning(f"Failed to generate individual embedding: {e2}")
                        continue

        if len(embeddings_list) == 0:
            raise ValueError("No embeddings were successfully generated")

        # Determine embedding dimension
        if self.embedding_dim is None:
            self.embedding_dim = len(embeddings_list[0])

        # Create embedding column names
        embedding_cols = [f"embedding_{i}" for i in range(self.embedding_dim)]

        # Create result dataframe
        embeddings_array = np.array(embeddings_list)
        result_dict = {
            "timestamp": timestamps[valid_indices],
            **{col: embeddings_array[:, i] for i, col in enumerate(embedding_cols)},
        }

        result_df = pl.DataFrame(result_dict)

        return result_df.lazy()
