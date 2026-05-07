"""PCA processor for precomputing and caching PCA models on embeddings."""

from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger

from definitions import ROOT_DIR
from embeddings.pca import IncrementalPCAProcessor, PCAConfig
from utils import date_range


class PCAProcessor:
    """
    Processor for precomputing and caching PCA models on embedding data.

    This class handles:
    - Loading embedding data for specified date ranges
    - Fitting PCA models with specified configurations
    - Caching fitted models to disk
    - Checking for existing models to avoid redundant computation
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        embedding_code: str,
        base_path: str | Path = "embeddings",
        pca_model_path: str | Path | None = None,
    ):
        """
        Initialize PCA processor.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            embedding_code: Embedding configuration code
            base_path: Base directory containing embedding data
            pca_model_path: Directory to save PCA models. If None, uses ROOT_DIR/data/embeddings/pca
        """
        self.exchange = exchange
        self.symbol = symbol
        self.embedding_code = embedding_code
        self.base_path = Path(base_path)

        if pca_model_path is None:
            self.pca_model_path = ROOT_DIR / "data" / "embeddings" / "pca"
        else:
            self.pca_model_path = Path(pca_model_path)

        self.pca_model_path.mkdir(parents=True, exist_ok=True)

    def get_embedding_dir(self) -> Path:
        """Get directory containing embedding data."""
        return self.base_path / self.exchange / self.symbol / self.embedding_code

    def get_model_save_path(
        self,
        n_components: int,
        start_date: date,
        end_date: date,
    ) -> Path:
        """
        Get path where PCA model should be saved.

        Args:
            n_components: Number of PCA components
            start_date: Start date of training data
            end_date: End date of training data

        Returns:
            Path to save PCA model
        """
        filename = (
            f"pca_{self.exchange}_{self.symbol}_{self.embedding_code}_"
            f"n{n_components}_"
            f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.pkl"
        )
        return self.pca_model_path / filename

    def model_exists(
        self,
        n_components: int,
        start_date: date,
        end_date: date,
    ) -> bool:
        """
        Check if PCA model already exists for given configuration.

        Args:
            n_components: Number of PCA components
            start_date: Start date of training data
            end_date: End date of training data

        Returns:
            True if model exists, False otherwise
        """
        model_path = self.get_model_save_path(n_components, start_date, end_date)
        return model_path.exists()

    def load_embeddings(
        self,
        start_date: date,
        end_date: date,
        max_samples: int | None = None,
    ) -> pl.LazyFrame:
        """
        Load embeddings for specified date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            max_samples: Maximum number of samples to load (None for all)

        Returns:
            LazyFrame with embeddings and timestamps

        Raises:
            FileNotFoundError: If embedding directory doesn't exist
        """
        embedding_dir = self.get_embedding_dir()

        if not embedding_dir.exists():
            raise FileNotFoundError(
                f"Embedding directory not found: {embedding_dir}\n"
                f"Please ensure embeddings have been generated for "
                f"{self.exchange}/{self.symbol}/{self.embedding_code}"
            )

        # Get all parquet files in date range
        date_strs = [d.strftime("%Y-%m-%d") for d in date_range(start_date, end_date)]
        paths = []

        for date_str in date_strs:
            # Look for files matching date pattern
            matching_files = list(embedding_dir.glob(f"{date_str}*.parquet"))
            if matching_files:
                paths.extend(matching_files)
            else:
                logger.warning(f"No embedding file found for date {date_str}")

        if not paths:
            raise FileNotFoundError(
                f"No embedding files found in date range {start_date} to {end_date} "
                f"at {embedding_dir}"
            )

        logger.info(f"Loading {len(paths)} embedding file(s) from {embedding_dir}")

        embeddings = pl.scan_parquet(sorted(paths))

        # If max_samples specified, sample uniformly using gather_every
        if max_samples is not None:
            total_rows = embeddings.select(pl.len()).collect().item()
            if max_samples < total_rows:
                stride = max(1, total_rows // max_samples + 1)
                actual_samples = (total_rows + stride - 1) // stride

                logger.info(
                    f"Sampling uniformly from {total_rows:,} rows using stride {stride}."
                    f" Requested: {max_samples:,}, Actual: {actual_samples:,}"
                )

                embeddings = embeddings.gather_every(stride)
            else:
                logger.info(
                    f"Requested {max_samples:,} samples, but only {total_rows:,} available. Using all."
                )

        return embeddings

    def fit_pca(
        self,
        config: PCAConfig,
        start_date: date,
        end_date: date,
        max_samples: int | None = None,
        force_recompute: bool = False,
        verbose: bool = True,
    ) -> IncrementalPCAProcessor:
        """
        Fit PCA model on embeddings for specified date range.

        Args:
            config: PCA configuration
            start_date: Start date for training data (inclusive)
            end_date: End date for training data (inclusive)
            max_samples: Maximum number of samples to use (None for all)
            force_recompute: If True, recompute even if model exists
            verbose: Whether to show progress

        Returns:
            Fitted IncrementalPCAProcessor
        """
        # Set save path in config if not already set
        if config.save_path is None:
            config.save_path = self.get_model_save_path(
                config.n_components,
                start_date,
                end_date,
            )

        # Check if model already exists
        if not force_recompute and self.model_exists(config.n_components, start_date, end_date):
            logger.info(
                f"PCA model already exists at {config.save_path}. "
                "Use force_recompute=True to recompute."
            )
            return IncrementalPCAProcessor.load(config.save_path)

        # Load embeddings
        logger.info(
            f"Loading embeddings for {self.exchange}/{self.symbol}/{self.embedding_code} "
            f"from {start_date} to {end_date}"
        )
        embeddings = self.load_embeddings(start_date, end_date, max_samples)

        # Initialize processor and fit
        processor = IncrementalPCAProcessor(config)

        logger.info(
            f"Fitting PCA with {config.n_components} components (chunk_size={config.chunk_size:,})"
        )

        processor.fit(embeddings, verbose=verbose)

        processor.save()

        logger.info(
            f"PCA model saved to {config.save_path}\n"
            f"  Explained variance: {processor.get_total_explained_variance():.4f}\n"
            f"  Samples seen: {processor.get_n_samples_seen():,}"
        )

        return processor

    def load_pca(
        self,
        n_components: int,
        start_date: date,
        end_date: date,
    ) -> IncrementalPCAProcessor:
        """
        Load previously fitted PCA model.

        Args:
            n_components: Number of PCA components
            start_date: Start date used for training
            end_date: End date used for training

        Returns:
            Loaded IncrementalPCAProcessor

        Raises:
            FileNotFoundError: If model doesn't exist
        """
        model_path = self.get_model_save_path(n_components, start_date, end_date)

        if not model_path.exists():
            raise FileNotFoundError(
                f"PCA model not found at {model_path}\n"
                f"Please run fit_pca() first to compute the model."
            )

        return IncrementalPCAProcessor.load(model_path)

    def list_available_models(self) -> list[Path]:
        """
        List all available PCA models for this configuration.

        Returns:
            List of paths to available PCA models
        """
        pattern = f"pca_{self.exchange}_{self.symbol}_{self.embedding_code}_*.pkl"
        return sorted(self.pca_model_path.glob(pattern))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PCAProcessor("
            f"exchange={self.exchange}, "
            f"symbol={self.symbol}, "
            f"embedding_code={self.embedding_code})"
        )
