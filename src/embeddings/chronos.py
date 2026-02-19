"""Chronos embedding generator implementation"""

from typing import Any
import numpy as np
import polars as pl
import torch
from loguru import logger
from tqdm import tqdm

from embeddings.base import BaseEmbeddingGenerator
from embeddings.registry import EmbeddingGeneratorRegistry
from embeddings.aggregator import EmbeddingAggregator
from embeddings.utils import (
    compute_patch_statistics,
    compute_differenced_sequence,
    extract_feature_columns,
)

from chronos import ChronosPipeline, BaseChronosPipeline, Chronos2Pipeline


# Cache for Chronos pipelines
_PIPELINE_CACHE: dict[str, Any] = {}

# Model type configurations
_MODEL_CONFIGS = {
    "t5": {
        "sizes": ["mini", "small", "base", "large"],
        "pipeline_class": ChronosPipeline,
        "model_name_template": "amazon/chronos-t5-{size}",
    },
    "bolt": {
        "sizes": ["mini", "small", "base", "large"],
        "pipeline_class": BaseChronosPipeline,
        "model_name_template": "amazon/chronos-bolt-{size}",
    },
    "chronos2": {
        "sizes": ["base"],
        "pipeline_class": Chronos2Pipeline,
        "model_name_template": "amazon/chronos-2",
    },
    "autogluon-chronos2": {
        "sizes": ["small", "synth"],
        "pipeline_class": Chronos2Pipeline,
        "model_name_template": "autogluon/chronos-2-{size}",
    },
}


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
                - model_type: Type of Chronos model ("t5", "bolt", "chronos2", "autogluon-chronos2", default: "t5")
                - model_size: Size of the model ("mini", "small", "base", "large", default: "mini")
                  Note: For chronos2, only base size is available (model_size is ignored)
                - seq_aggregation: How to aggregate across sequence dimension ("last", "mean", "max", "min", or "concat", default: "last")
                - feat_aggregation: How to aggregate across feature dimension ("last", "mean", "max", "min", or "concat", default: "concat")
                - augment_with_statistics: Whether to augment with patch statistics (default: False)
                - k: Number of patches for statistics (default: 8)
                - use_differencing: Whether to augment with differenced embeddings (default: False)
                - device: Device to use ("cuda" or "cpu", default: "cuda")
                - batch_size: Number of samples to process in parallel (default: 32)
                - stride: Generate embeddings only for timestamps where timestamp % stride == 0 (default: 1, i.e., all samples)
                - disable_tqdm: Whether to disable progress bars (default: False)
        """
        super().__init__(config)

        # Set defaults
        self.model_type: str = self.config.get("model_type", "t5")
        self.model_size: str = self.config.get("model_size", "mini")
        self.seq_aggregation: str = self.config.get("seq_aggregation", "last")
        self.feat_aggregation: str = self.config.get("feat_aggregation", "concat")
        self.augment_with_statistics: bool = self.config.get("augment_with_statistics", False)
        self.k: int = self.config.get("k", 8)
        self.use_differencing: bool = self.config.get("use_differencing", False)
        self.device: str = self.config.get("device", "cuda")
        self.batch_size: int = self.config.get("batch_size", 32)
        self.stride: int = self.config.get("stride", 1)
        self.disable_tqdm: bool = self.config.get("disable_tqdm", False)

        self.validate_config()

        self.seq_aggregator = EmbeddingAggregator(self.seq_aggregation)
        self.feat_aggregator = EmbeddingAggregator(self.feat_aggregation)

        self._pipeline: Any = None

    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        if self.model_type not in _MODEL_CONFIGS:
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                f"Must be one of {sorted(_MODEL_CONFIGS.keys())}"
            )

        available_sizes = _MODEL_CONFIGS[self.model_type]["sizes"]
        if self.model_size not in available_sizes:
            raise ValueError(
                f"Invalid model_size: {self.model_size} for model_type: {self.model_type}. "
                f"Must be one of {available_sizes}"
            )

        if self.k < 1:
            raise ValueError(f"k must be at least 1, got {self.k}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")

        if self.stride < 1:
            raise ValueError(f"stride must be at least 1, got {self.stride}")

        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}. Must be 'cuda' or 'cpu'")

        return True

    @property
    def pipeline(self):
        """Lazy load and cache the Chronos pipeline."""
        if self._pipeline is None:
            cache_key = f"{self.model_type}_{self.model_size}_{self.device}"
            if cache_key not in _PIPELINE_CACHE:
                model_config = _MODEL_CONFIGS[self.model_type]
                pipeline_class = model_config["pipeline_class"]
                model_name_template = model_config["model_name_template"]

                # Format model name (chronos2 ignores size parameter)
                if self.model_type == "chronos2":
                    model_name = model_name_template
                    logger.info(f"Loading Chronos 2 on {self.device}")
                else:
                    model_name = model_name_template.format(size=self.model_size)
                    logger.info(
                        f"Loading Chronos {self.model_type}-{self.model_size} on {self.device}"
                    )

                # Load pipeline with appropriate dtype
                if self.model_type in ["chronos2", "autogluon-chronos2"]:
                    _PIPELINE_CACHE[cache_key] = pipeline_class.from_pretrained(
                        model_name,
                        device_map=self.device,
                    )
                else:
                    _PIPELINE_CACHE[cache_key] = pipeline_class.from_pretrained(
                        model_name,
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
        if self.model_type in ["chronos2", "autogluon-chronos2"]:
            # Chronos 2: Input shape (batch_size, n_features, seq_len)
            # Output is a list of tensors, convert to single tensor
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

            # Reshape to match Chronos 2 format
            _, new_seq_len, embedding_size = embeddings_flat.shape
            embeddings = embeddings_flat.reshape(
                batch_size, n_features, new_seq_len, embedding_size
            )

            # Output shape: (batch_size, n_features, new_seq_len, embedding_size)
            return embeddings

    def _process_embedding_batch(self, data_batch: np.ndarray) -> torch.Tensor:
        """
        Generate and process embeddings from a batch of data.

        Args:
            data_batch: Input array of shape (batch_size, seq_len, n_features)

        Returns:
            Processed embedding tensor of shape (batch_size, embedding_dim)
        """
        # Convert to tensor with shape (batch_size, n_features, seq_len)
        context = torch.tensor(
            np.transpose(data_batch, (0, 2, 1)), dtype=torch.float32
        )  # (batch_size, n_features, seq_len)

        # Get embeddings: (batch_size, n_features, new_seq_len, embedding_size)
        embeddings = self._embed_with_chronos(context)

        # Aggregate across sequence dimension (dim=2)
        # Shape: (batch_size, n_features, new_seq_len, embedding_size) -> (batch_size, n_features, embedding_size)
        emb = self.seq_aggregator.aggregate(embeddings, dim=2)

        # Aggregate across feature dimension (dim=1)
        # Shape: (batch_size, n_features, embedding_size) -> (batch_size, embedding_dim)
        emb = self.feat_aggregator.aggregate(emb, dim=1)

        return emb.float()

    def _process_embedding_single(self, data: np.ndarray) -> torch.Tensor:
        """
        Generate and process embedding from a single data sample.

        Args:
            data: Input array of shape (seq_len, n_features)

        Returns:
            Processed embedding tensor of shape (embedding_dim,)
        """
        data_batch = np.expand_dims(data, axis=0)  # (1, seq_len, n_features)
        emb_batch = self._process_embedding_batch(data_batch)  # (1, embedding_dim)
        return emb_batch.squeeze(0)  # (embedding_dim,)

    def _generate_embedding_column_names(
        self, feature_cols: list[str], base_emb_size: int
    ) -> list[str]:
        """
        Generate meaningful column names for embeddings.

        Args:
            feature_cols: List of feature column names from the input data
            base_emb_size: Size of the base embedding (before augmentation)

        Returns:
            List of column names indicating the source and type of each embedding dimension
        """
        n_features = len(feature_cols)
        column_names = []

        # Determine embedding size per feature based on aggregation strategy
        if self.feat_aggregation == "concat":
            # Each feature contributes equally to the base embedding
            emb_per_feature = base_emb_size // n_features
            for feat_name in feature_cols:
                for i in range(emb_per_feature):
                    column_names.append(f"emb_{feat_name}_{i}")
        else:
            # Features are aggregated, so we can't attribute to specific features
            for i in range(base_emb_size):
                column_names.append(f"emb_agg_{i}")

        # Add differenced embedding columns if enabled
        if self.use_differencing:
            if self.feat_aggregation == "concat":
                emb_per_feature = base_emb_size // n_features
                for feat_name in feature_cols:
                    for i in range(emb_per_feature):
                        column_names.append(f"emb_diff_{feat_name}_{i}")
            else:
                for i in range(base_emb_size):
                    column_names.append(f"emb_diff_agg_{i}")

        # Add patch statistics columns if enabled
        if self.augment_with_statistics:
            # Statistics are ordered as: [means, stds, mins, maxs] for k patches
            stat_names = ["mean", "std", "min", "max"]
            for stat_name in stat_names:
                for patch_idx in range(self.k):
                    for feat_name in feature_cols:
                        column_names.append(f"{stat_name}_p{patch_idx}_{feat_name}")

        return column_names

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

        # Prepare context windows for samples matching stride criterion
        all_contexts = []
        valid_indices = []

        for i in range(start_index, len(feature_data)):
            # Only generate embeddings for timestamps where timestamp % stride == 0
            if i % self.stride == 0:
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
            unit="sample",
            unit_scale=self.batch_size,
        ):
            batch_end = min(batch_start + self.batch_size, len(all_contexts))
            batch_contexts = all_contexts[batch_start:batch_end]

            # Stack contexts into a batch: (batch_size, seq_len, n_features)
            batch_array = np.stack(batch_contexts, axis=0)

            try:
                batch_embeddings = self._process_embedding_batch(batch_array).cpu().numpy()

                if self.use_differencing and context_length > 1:
                    # Compute differenced sequences for the batch
                    differenced_batch = np.array(
                        [compute_differenced_sequence(ctx, order=1) for ctx in batch_contexts]
                    )
                    diff_embeddings = (
                        self._process_embedding_batch(differenced_batch).cpu().numpy()
                    )
                    batch_embeddings = np.concatenate([batch_embeddings, diff_embeddings], axis=1)

                if self.augment_with_statistics and context_length >= self.k:
                    # Compute patch statistics for each sample in the batch
                    stats_batch = np.array(
                        [
                            compute_patch_statistics(ctx, self.k, normalize=True)
                            for ctx in batch_contexts
                        ]
                    )
                    batch_embeddings = np.concatenate([batch_embeddings, stats_batch], axis=1)

                embeddings_list.extend(batch_embeddings)

            except Exception as e:
                logger.warning(
                    f"Failed to generate embeddings for batch starting at {batch_start}: {e}"
                )
                for ctx in batch_contexts:
                    try:
                        embedding = self.generate_embedding(ctx, context_length=context_length)
                        embeddings_list.append(embedding)
                    except Exception as e2:
                        logger.warning(f"Failed to generate individual embedding: {e2}")
                        continue

        if len(embeddings_list) == 0:
            raise ValueError("No embeddings were successfully generated")

        if self.embedding_dim is None:
            self.embedding_dim = len(embeddings_list[0])

        # Calculate base embedding size (before augmentation)
        base_emb_size = self._process_embedding_batch(all_contexts[0:1]).shape[1]

        # Create meaningful embedding column names
        embedding_cols = self._generate_embedding_column_names(feature_cols, base_emb_size)

        # Create result dataframe
        embeddings_array = np.array(embeddings_list)
        result_dict = {
            "timestamp": timestamps[valid_indices],
            **{col: embeddings_array[:, i] for i, col in enumerate(embedding_cols)},
        }

        result_df = pl.DataFrame(result_dict)

        return result_df.lazy()
