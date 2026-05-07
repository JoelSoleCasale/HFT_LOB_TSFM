"""Incremental PCA processor for large LazyFrame datasets."""

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from embeddings.pca.config import PCAConfig
from utils import iter_slices


class IncrementalPCAProcessor:
    """
    Incremental PCA processor that works with Polars LazyFrames.

    This class handles fitting and transforming large datasets that don't fit in memory
    by processing data in chunks. It's designed to work seamlessly with LazyFrames
    to maintain memory efficiency.

    Attributes:
        config: PCA configuration
        pca: Fitted sklearn IncrementalPCA object (None until fitted)
        feature_columns: List of feature column names used for fitting
        is_fitted: Whether PCA has been fitted
    """

    def __init__(self, config: PCAConfig):
        """Initialize PCA processor.

        Args:
            config: PCA configuration
        """
        self.config = config
        self.pca: IncrementalPCA | None = None
        self.feature_columns: list[str] | None = None
        self.is_fitted = False

    def fit(
        self,
        data: pl.LazyFrame,
        feature_columns: list[str] | None = None,
        stride: int = 5,
        verbose: bool = True,
    ) -> "IncrementalPCAProcessor":
        """
        Fit incremental PCA on all provided data.

        Args:
            data: LazyFrame containing features and timestamp column
            feature_columns: List of feature column names. If None, uses all columns except 'timestamp'
            stride: Stride value for gather_every to reduce memory usage
            verbose: Whether to show progress bar

        Returns:
            Self for method chaining
        """
        # Determine feature columns
        if feature_columns is None:
            all_cols = data.collect_schema().names()
            feature_columns = [col for col in all_cols if col != "timestamp"]

        self.feature_columns = feature_columns

        # Get total data size
        total_rows = data.select(pl.len()).collect().item()

        if verbose:
            logger.info(
                f"Fitting IncrementalPCA with {self.config.n_components} components "
                f"on {total_rows:,} rows..."
            )

        # Initialize IncrementalPCA
        self.pca = IncrementalPCA(
            n_components=self.config.n_components, whiten=self.config.whiten, batch_size=None
        )

        n_chunks = math.ceil(total_rows / self.config.chunk_size)
        iterator = iter_slices(data, n_rows=self.config.chunk_size)

        if verbose:
            iterator = tqdm(iterator, total=n_chunks, desc="Fitting PCA")

        for chunk in iterator:
            if isinstance(chunk, pl.LazyFrame):
                chunk = chunk.collect(engine="streaming").gather_every(stride)

            X_chunk = chunk.select(self.feature_columns).to_numpy()
            self.pca.partial_fit(X_chunk)

        self.is_fitted = True

        # Log explained variance
        total_variance = self.pca.explained_variance_ratio_.sum()
        if verbose:
            logger.info(f"PCA fitting complete. Explained variance ratio: {total_variance:.4f}")

        return self

    def transform(
        self, data: pl.LazyFrame, keep_timestamp: bool = True, verbose: bool = True
    ) -> pl.LazyFrame:
        """
        Transform data using fitted PCA.

        Args:
            data: LazyFrame containing features to transform
            keep_timestamp: Whether to include timestamp column in output
            verbose: Whether to show progress bar

        Returns:
            LazyFrame with PCA-transformed features and optionally timestamp

        Raises:
            RuntimeError: If PCA hasn't been fitted yet
        """
        if not self.is_fitted or self.pca is None:
            raise RuntimeError("PCA must be fitted before transforming. Call fit() first.")

        if self.feature_columns is None:
            raise RuntimeError("Feature columns not set. Call fit() first.")

        total_rows = data.select(pl.len()).collect().item()

        if verbose:
            logger.info(f"Transforming {total_rows:,} rows with fitted PCA...")

        X_transformed_chunks = []
        timestamps_chunks = []

        n_chunks = (total_rows // self.config.chunk_size) + (
            1 if total_rows % self.config.chunk_size > 0 else 0
        )
        iterator = iter_slices(data, n_rows=self.config.chunk_size)

        if verbose:
            iterator = tqdm(iterator, total=n_chunks, desc="Transforming with PCA")

        for chunk in iterator:
            if isinstance(chunk, pl.LazyFrame):
                chunk = chunk.collect()
            X_chunk = chunk.select(self.feature_columns).to_numpy()
            X_transformed = self.pca.transform(X_chunk)

            X_transformed_chunks.append(X_transformed)

            if keep_timestamp:
                timestamps = chunk.select("timestamp").to_numpy()
                timestamps_chunks.append(timestamps)

        X_pca = np.vstack(X_transformed_chunks)

        if verbose:
            logger.info(f"PCA transformation complete. New shape: {X_pca.shape}")

        pca_col_names = [f"pca_{i}" for i in range(self.config.n_components)]
        result_dict: dict[str, Any] = {}

        if keep_timestamp:
            timestamps_combined = np.vstack(timestamps_chunks)
            result_dict["timestamp"] = timestamps_combined.squeeze()

        for i, col_name in enumerate(pca_col_names):
            result_dict[col_name] = X_pca[:, i]

        return pl.LazyFrame(result_dict)

    def fit_transform(
        self,
        data: pl.LazyFrame,
        feature_columns: list[str] | None = None,
        keep_timestamp: bool = True,
        verbose: bool = True,
    ) -> pl.LazyFrame:
        """
        Fit PCA and transform data in one call.

        Args:
            data: LazyFrame containing features
            feature_columns: List of feature column names. If None, uses all except 'timestamp'
            keep_timestamp: Whether to include timestamp in output
            verbose: Whether to show progress

        Returns:
            LazyFrame with PCA-transformed features
        """
        self.fit(data, feature_columns, verbose)
        return self.transform(data, keep_timestamp, verbose)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio for each component.

        Returns:
            Array of explained variance ratios

        Raises:
            RuntimeError: If PCA hasn't been fitted yet
        """
        if not self.is_fitted or self.pca is None:
            raise RuntimeError("PCA must be fitted first. Call fit().")

        return self.pca.explained_variance_ratio_

    def get_total_explained_variance(self) -> float:
        """
        Get total explained variance across all components.

        Returns:
            Total explained variance ratio (sum across components)

        Raises:
            RuntimeError: If PCA hasn't been fitted yet
        """
        return float(self.get_explained_variance_ratio().sum())

    def get_n_samples_seen(self) -> int:
        """
        Get number of samples seen during fitting.

        Returns:
            Number of samples used for fitting

        Raises:
            RuntimeError: If PCA hasn't been fitted yet
        """
        if not self.is_fitted or self.pca is None:
            raise RuntimeError("PCA must be fitted first. Call fit().")

        return int(self.pca.n_samples_seen_)

    def save(self, path: Path | str | None = None) -> Path:
        """
        Save fitted PCA model to disk.

        Args:
            path: Path to save model. If None, uses config.save_path

        Returns:
            Path where model was saved

        Raises:
            RuntimeError: If PCA hasn't been fitted yet
            ValueError: If no save path provided
        """
        if not self.is_fitted or self.pca is None:
            raise RuntimeError("PCA must be fitted before saving. Call fit() first.")

        save_path = Path(path) if path is not None else self.config.save_path

        if save_path is None:
            raise ValueError("No save path provided. Set config.save_path or pass path argument.")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "pca": self.pca,
            "config": self.config,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved PCA model to {save_path}")

        return save_path

    @classmethod
    def load(cls, path: Path | str) -> "IncrementalPCAProcessor":
        """
        Load fitted PCA model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded IncrementalPCAProcessor with fitted model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"PCA model not found at {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        processor = cls(config=state["config"])
        processor.pca = state["pca"]
        processor.feature_columns = state["feature_columns"]
        processor.is_fitted = state["is_fitted"]

        logger.info(f"Loaded PCA model from {path}")

        return processor

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"IncrementalPCAProcessor("
            f"n_components={self.config.n_components}, "
            f"status={status})"
        )
