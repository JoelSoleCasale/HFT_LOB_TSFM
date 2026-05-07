from pathlib import Path
import polars as pl
from loguru import logger

from embeddings import EmbeddingPipeline, EmbeddingGeneratorRegistry
from core.orderbook import OrderBook
from features import InputSpace, FeaturePipeline, FeatureExtractorRegistry


class EmbeddingProcessor:
    """Processes orderbook data to generate embeddings with hourly granularity."""

    def __init__(
        self,
        base_folder: str | Path,
        embedding_type: str,
        embedding_config: dict,
        context_length: int,
        feature_extractors: list[str],
        orderbook_levels: int = 5,
        sample_time_delta: int = 100_000_000,
        interpolate: bool = True,
        orderbook_subfolder: str = "orderbook_snapshots",
        output_subfolder: str = "embeddings",
    ):
        """
        Initialize the embedding processor.

        Args:
            base_folder: Root data directory
            embedding_type: Type of embedding generator (e.g., "chronos")
            embedding_config: Configuration dict for the embedding generator
            context_length: Number of samples to use as context for embeddings
            feature_extractors: List of feature extractor names from registry
            orderbook_levels: Number of orderbook levels to use
            sample_time_delta: Time delta for sampling in nanoseconds
            interpolate: Whether to interpolate missing values
            orderbook_subfolder: Subfolder containing orderbook files (under base_folder)
            output_subfolder: Subfolder for saving embeddings (under base_folder)
        """
        self.base_folder = Path(base_folder)
        self.orderbook_folder = self.base_folder / orderbook_subfolder
        self.output_folder = self.base_folder / output_subfolder
        self.context_length = context_length
        self.orderbook_levels = orderbook_levels
        self.sample_time_delta = sample_time_delta
        self.interpolate = interpolate
        self.embedding_type = embedding_type
        self.embedding_config = embedding_config

        self.feature_pipeline = FeaturePipeline()
        for extractor_name in feature_extractors:
            extractor = FeatureExtractorRegistry.create(extractor_name)
            self.feature_pipeline.add_extractor(extractor)
            logger.info(f"Added feature extractor: {extractor_name}")

        embedding_generator = EmbeddingGeneratorRegistry.create(
            embedding_type, config=embedding_config
        )
        self.embedding_pipeline = EmbeddingPipeline().add_generator(embedding_generator)

        logger.info(f"Initialized EmbeddingProcessor with {embedding_type} embeddings")
        logger.info(f"Context length: {context_length}")
        logger.info(f"Orderbook levels: {orderbook_levels}")
        logger.info(f"Sample time delta: {sample_time_delta}ns")
        logger.info(f"Orderbook folder: {self.orderbook_folder}")
        logger.info(f"Output folder: {self.output_folder}")

    def _get_config_string(self) -> str:
        """
        Generate a configuration string for the embedding output path.

        Returns:
            String encoding key configuration parameters
        """
        model_type = self.embedding_config.get("model_type", "unknown")
        model_size = self.embedding_config.get("model_size", "unknown")
        seq_agg = self.embedding_config.get("seq_aggregation", "last")
        use_diff = self.embedding_config.get("use_differencing", False)
        stride = self.embedding_config.get("stride", 1)

        # Format: type-size_ctx<length>_seq<agg>[_diff][_s<stride>]
        config_parts = [
            f"{model_type}-{model_size}",
            f"ctx{self.context_length}",
            f"seq{seq_agg}",
        ]

        if use_diff:
            config_parts.append("diff")

        if stride > 1:
            config_parts.append(f"s{stride}")

        return "_".join(config_parts)

    def _get_orderbook_path(self, symbol: str, exchange: str, date_str: str) -> Path:
        """Get path to orderbook parquet file."""
        return self.orderbook_folder / exchange / symbol / f"{date_str}_L20.parquet"

    def _get_output_path(self, symbol: str, exchange: str, date_str: str, hour: int) -> Path:
        """
        Get path to output embedding parquet file with configuration details.

        Path format: <output_folder>/<exchange>/<symbol>/<config_string>/<date>-<hour>.parquet
        """
        config_str = self._get_config_string()
        return (
            self.output_folder / exchange / symbol / config_str / f"{date_str}-{hour:02d}.parquet"
        )

    def _load_and_extract_features(
        self, symbol: str, exchange: str, date_str: str
    ) -> pl.LazyFrame | None:
        """
        Load orderbook data and extract features as a LazyFrame.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            date_str: Date string in YYYY-MM-DD format

        Returns:
            LazyFrame with features or None if file doesn't exist
        """
        orderbook_path = self._get_orderbook_path(symbol, exchange, date_str)

        if not orderbook_path.exists():
            logger.warning(f"Orderbook file not found: {orderbook_path}")
            return None

        logger.debug(f"Loading orderbook from {orderbook_path}")

        # Load orderbook with preprocessing (following notebook pattern)
        orderbook = (
            OrderBook.from_parquet([orderbook_path], lazy=True)
            .select_levels(self.orderbook_levels)
            .sample_by_time(time_delta=self.sample_time_delta, interpolate=self.interpolate)
        )

        # Create input space
        input_space = InputSpace(orderbook_snapshots=orderbook)

        # Extract features
        logger.debug(f"Extracting features for {date_str}")
        features_lf = self.feature_pipeline.extract_all(input_space)

        return features_lf

    def _get_hour_boundaries(
        self, features_lf: pl.LazyFrame, date_str: str
    ) -> list[tuple[int, int, int]]:
        """
        Get timestamp boundaries for each hour in nanoseconds (UTC).

        Args:
            features_lf: LazyFrame with features
            date_str: Date string in YYYY-MM-DD format

        Returns:
            List of (hour, start_ns, end_ns) tuples
        """
        from datetime import datetime, timezone

        # Parse the date as UTC to avoid timezone issues
        base_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        # Get min and max timestamps from data
        bounds = features_lf.select(
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
        ).collect()

        min_ts = bounds["min_ts"][0]
        max_ts = bounds["max_ts"][0]

        # Convert to human-readable for debugging
        min_ts_dt = datetime.fromtimestamp(min_ts / 1e9, tz=timezone.utc)
        max_ts_dt = datetime.fromtimestamp(max_ts / 1e9, tz=timezone.utc)

        logger.debug("Data timestamp range (UTC):")
        logger.debug(f"  Min: {min_ts_dt.isoformat()} ({min_ts} ns)")
        logger.debug(f"  Max: {max_ts_dt.isoformat()} ({max_ts} ns)")

        boundaries = []
        skipped_hours = []

        for hour in range(24):
            # Calculate hour boundaries in nanoseconds using UTC
            hour_start_ns = int(base_date.timestamp() * 1e9) + hour * int(3600 * 1e9)
            hour_end_ns = int(base_date.timestamp() * 1e9) + (hour + 1) * int(3600 * 1e9)

            # Only include hours that overlap with actual data
            if hour_end_ns >= min_ts and hour_start_ns <= max_ts:
                boundaries.append((hour, hour_start_ns, hour_end_ns))
                logger.debug(f"Hour {hour:02d} (UTC): {hour_start_ns} to {hour_end_ns} - INCLUDED")
            else:
                skipped_hours.append(hour)
                logger.debug(
                    f"Hour {hour:02d} (UTC): {hour_start_ns} to {hour_end_ns} - SKIPPED "
                    f"(no data overlap)"
                )

        if skipped_hours:
            logger.info(
                f"Skipped {len(skipped_hours)} hours due to no data: "
                f"{', '.join(f'{h:02d}' for h in skipped_hours)}"
            )

        return boundaries

    def _process_hour(
        self,
        features_lf: pl.LazyFrame,
        hour: int,
        hour_start_ns: int,
        hour_end_ns: int,
        context_start_ns: int,
    ) -> pl.LazyFrame | None:
        """
        Process a single hour of data to generate embeddings.

        Args:
            features_lf: Full feature LazyFrame for the day
            hour: Hour of day (0-23)
            hour_start_ns: Start timestamp for the hour (nanoseconds)
            hour_end_ns: End timestamp for the hour (nanoseconds)
            context_start_ns: Start timestamp for context window (nanoseconds)

        Returns:
            LazyFrame with embeddings for this hour, or None if no data
        """
        # Filter to context window + current hour using lazy operations
        # This includes context_length samples before the hour if available
        hour_with_context = features_lf.filter(
            (pl.col("timestamp") >= context_start_ns) & (pl.col("timestamp") < hour_end_ns)
        )

        hour_df = hour_with_context.collect()

        if hour_df.height == 0:
            logger.warning(f"Hour {hour:02d}: No data found in time range")
            return None

        logger.debug(
            f"Hour {hour:02d}: Processing {hour_df.height} samples "
            f"(context window: {context_start_ns} to {hour_end_ns})"
        )

        # Generate embeddings using the pipeline
        embeddings_lf = self.embedding_pipeline.generate(
            hour_df.lazy(), context_length=self.context_length
        )

        # Filter to only keep embeddings within the actual hour (not context)
        embeddings_hour = embeddings_lf.filter(
            (pl.col("timestamp") >= hour_start_ns) & (pl.col("timestamp") < hour_end_ns)
        )

        # Save it as Float32 to save space
        embeddings_hour = embeddings_hour.cast({pl.Float64: pl.Float32})

        return embeddings_hour

    def _get_context_from_previous_day(
        self, symbol: str, exchange: str, previous_date_str: str
    ) -> pl.LazyFrame | None:
        """
        Load context samples from the last hour of the previous day.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            previous_date_str: Date string of previous day in YYYY-MM-DD format

        Returns:
            LazyFrame with context samples or None if not available
        """
        # Try to load the last hour (23) from previous day
        previous_output_path = self._get_output_path(symbol, exchange, previous_date_str, 23)

        if not previous_output_path.exists():
            logger.warning(
                f"Previous day's last hour not found: {previous_output_path}. "
                "First hour will have limited context."
            )
            return None

        logger.debug(f"Loading context from previous day: {previous_output_path}")
        return pl.scan_parquet(previous_output_path)

    def process_day(
        self,
        symbol: str,
        exchange: str,
        date_str: str,
        skip_existing: bool = True,
    ) -> bool:
        """
        Process a single day of feature data to generate hourly embedding files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            date_str: Date string in YYYY-MM-DD format
            skip_existing: If True, skip hours that already have output files

        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"Processing day: {date_str} for {symbol} on {exchange}")

        # Load orderbook and extract features lazily
        features_lf = self._load_and_extract_features(symbol, exchange, date_str)
        if features_lf is None:
            return False

        # Get hour boundaries
        try:
            hour_boundaries = self._get_hour_boundaries(features_lf, date_str)
        except Exception as e:
            logger.error(f"Failed to get hour boundaries: {e}")
            return False

        if not hour_boundaries:
            logger.warning(f"No valid hour boundaries found for {date_str}")
            return False

        logger.info(f"Found {len(hour_boundaries)} hours to process")

        # Track skipped hours with reasons
        skipped_existing = []
        skipped_no_data = []
        skipped_errors = []
        processed = []

        # Process each hour
        for hour, hour_start_ns, hour_end_ns in hour_boundaries:
            output_path = self._get_output_path(symbol, exchange, date_str, hour)

            # Skip if exists and skip_existing is True
            if skip_existing and output_path.exists():
                logger.debug(f"Hour {hour:02d}: Skipping (file already exists at {output_path})")
                skipped_existing.append(hour)
                continue

            # Calculate context start time
            # We need to estimate the time for context_length samples
            # For safety, we'll fetch more data and let the embedding generator handle it
            # Assume a reasonable sampling rate (e.g., 10 samples/second = 100ms intervals)
            estimated_context_duration_ns = self.context_length * int(
                100_000_000
            )  # 100ms per sample

            context_start_ns = max(
                hour_start_ns - estimated_context_duration_ns,
                hour_boundaries[0][1],  # First hour start
            )

            # For first hour with limited context, log a warning
            if hour == hour_boundaries[0][0]:
                available_context_samples = (hour_start_ns - context_start_ns) // int(100_000_000)
                if available_context_samples < self.context_length:
                    logger.warning(
                        f"Hour {hour:02d}: Limited context available "
                        f"(~{available_context_samples} samples vs {self.context_length} requested). "
                        "Embeddings may use zero-padding for missing context."
                    )

            try:
                # Process the hour
                embeddings_hour = self._process_hour(
                    features_lf, hour, hour_start_ns, hour_end_ns, context_start_ns
                )

                if embeddings_hour is None:
                    logger.warning(f"Hour {hour:02d}: No embeddings generated (no data in range)")
                    skipped_no_data.append(hour)
                    continue

                # Create output directory
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Stream to disk using sink_parquet (efficient for large data)
                embeddings_hour.sink_parquet(output_path)

                # Get count for logging
                count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
                logger.info(f"Hour {hour:02d}: Saved {count} embeddings to {output_path}")
                processed.append(hour)

            except Exception as e:
                logger.error(f"Hour {hour:02d}: Failed with error: {e}", exc_info=True)
                skipped_errors.append(hour)
                continue

        logger.info(
            f"Done {date_str}: {len(processed)} processed, "
            f"{len(skipped_existing)} skipped, {len(skipped_no_data)} no data, "
            f"{len(skipped_errors)} errors"
        )
        return True
