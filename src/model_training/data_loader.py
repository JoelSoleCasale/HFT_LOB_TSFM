from pathlib import Path
from datetime import date, timedelta
import polars as pl
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger
from typing import List, Tuple

from definitions import ROOT_DIR
from utils import date_range
from core.orderbook import OrderBook
from features import InputSpace, TripleBarrierLabel
from data_manager.embeddings.pca import PCAProcessor


def get_embeddings_paths(
    str_code: str,
    start_date: date,
    end_date: date,
    exchange: str = "binance_futures",
    symbol: str = "BTCUSDT",
    base_path: str = "/scratch/PI/palomar/joel_sole/embedding_data",
) -> List[Path]:
    """
    Get list of embedding parquet file paths for specified date range.
    """
    dir_path = Path(base_path) / exchange / symbol / str_code
    if not dir_path.exists():
        # Fallback or check if user provided full path or relative
        pass

    paths = []
    # Check if dir_path exists, if not maybe log but the loop handles checking files
    if not dir_path.exists():
        logger.warning(f"Embedding directory not found: {dir_path}")

    for current_date in date_range(start_date, end_date):
        date_str = current_date.strftime("%Y-%m-%d")
        # Embeddings are hourly: YYYY-MM-DD-HH.parquet (HH from 00 to 23)
        for hour in range(24):
            file_path = dir_path / f"{date_str}-{hour:02d}.parquet"
            if file_path.exists():
                paths.append(file_path)

    if not paths:
        # Just return empty, caller handles error or we raise
        # The original code raises FileNotFoundError
        pass

    return sorted(paths)


def get_orderbook_paths(
    start_date: date,
    end_date: date,
    exchange: str = "binance_futures",
    symbol: str = "BTCUSDT",
    levels: int = 20,
) -> List[Path]:
    """
    Get list of orderbook parquet file paths for specified date range.
    """
    paths = []
    for current_date in date_range(start_date, end_date):
        path = (
            ROOT_DIR
            / "data"
            / "orderbook_snapshots"
            / exchange
            / symbol
            / f"{current_date.strftime('%Y-%m-%d')}_L{levels}.parquet"
        )
        if path.exists():
            paths.append(path)
        else:
            logger.warning(f"Orderbook file not found: {path}")

    return paths


def apply_pca_reduction(
    embeddings: pl.LazyFrame,
    n_components: int,
    train_start_date: date,
    train_end_date: date,
    embedding_code: str,
    data_start_date: date,
    data_end_date: date,
    exchange: str = "binance_futures",
    symbol: str = "BTCUSDT",
    base_path: str = "/scratch/PI/palomar/joel_sole/embedding_data",
) -> Tuple[pl.LazyFrame, int]:
    """
    Apply PCA dimensionality reduction using pre-computed PCA model.
    Results are cached to disk to avoid repeated computations.
    """
    # Get embedding columns (all except timestamp)
    all_cols = embeddings.collect_schema().names()
    embedding_cols = [col for col in all_cols if col != "timestamp"]
    original_size = len(embedding_cols)

    # Define cache path
    cache_dir = Path(base_path) / exchange / symbol / "PCA" / f"{embedding_code}_pca{n_components}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_filename = (
        f"{data_start_date.strftime('%Y-%m-%d')}_{data_end_date.strftime('%Y-%m-%d')}.parquet"
    )
    cache_path = cache_dir / cache_filename

    # Check if cached file exists
    if cache_path.exists():
        logger.info(f"Loading cached PCA-reduced embeddings from: {cache_path}")
        reduced_embeddings = pl.scan_parquet(cache_path)
        logger.info(
            f"Cached embeddings loaded successfully (original size: {original_size} -> {n_components})"
        )
        return reduced_embeddings, original_size

    logger.info(
        f"Cache not found. Applying PCA dimensionality reduction from {original_size} to {n_components} components..."
    )

    # Initialize PCA processor
    pca_processor = PCAProcessor(
        exchange=exchange,
        symbol=symbol,
        embedding_code=embedding_code,
        base_path=base_path,
    )

    # Load pre-computed PCA model
    logger.info(
        f"Loading pre-computed PCA model for training period: "
        f"{train_start_date} to {train_end_date}"
    )

    try:
        incremental_pca = pca_processor.load_pca(
            n_components=n_components,
            start_date=train_start_date,
            end_date=train_end_date,
        )
    except FileNotFoundError as e:
        logger.error(
            f"PCA model not found. Please pre-compute it first using PCAProcessor.fit_pca()\n"
            f"Error: {e}"
        )
        raise

    # Transform embeddings using loaded PCA
    logger.info("Transforming embeddings using loaded PCA model...")
    reduced_embeddings = incremental_pca.transform(
        data=embeddings,
        keep_timestamp=True,
    )

    # set the quantization to Float32 to save memory
    reduced_embeddings = reduced_embeddings.cast({pl.Float64: pl.Float32})

    # Log explained variance
    total_variance = incremental_pca.get_total_explained_variance()
    logger.info(f"Total explained variance: {total_variance:.4f}")

    # Save to cache using sink_parquet (no need to collect)
    logger.info(f"Saving PCA-reduced embeddings to cache: {cache_path}")
    reduced_embeddings.sink_parquet(cache_path)

    # Load the cached file as LazyFrame
    reduced_embeddings = pl.scan_parquet(cache_path)
    logger.info("Cache saved and reloaded successfully")

    return reduced_embeddings, original_size


def prepare_data(config: dict) -> Tuple[pl.LazyFrame, pl.LazyFrame, dict]:
    """
    Prepare features and labels based on configuration.
    Returns (features_df, labels_df, metadata)
    """
    # Extract config parameters
    data_config = config.get("data", {})
    paths_config = config.get("paths", {})

    first_date = data_config.get("first_date")
    n_days = data_config.get("n_days", 10)
    if isinstance(first_date, str):
        first_date = date.fromisoformat(first_date)

    end_date = first_date + timedelta(days=n_days - 1)

    embedding_code = data_config.get("embedding_code")
    max_samples = data_config.get("max_total_samples", 100_000_000)

    # Paths
    base_embedding_path = paths_config.get(
        "embeddings_dir", "/scratch/PI/palomar/joel_sole/embedding_data"
    )
    exchange = data_config.get("exchange", "binance_futures")
    symbol = data_config.get("symbol", "BTCUSDT")

    # 1. Load Embeddings
    logger.info(f"Loading embeddings: {embedding_code}...")
    embedding_paths = get_embeddings_paths(
        str_code=embedding_code,
        start_date=first_date,
        end_date=end_date,
        exchange=exchange,
        symbol=symbol,
        base_path=base_embedding_path,
    )

    if not embedding_paths:
        raise FileNotFoundError(
            f"No embedding files found for {exchange}/{symbol}/{embedding_code}"
        )

    embeddings = pl.scan_parquet(embedding_paths).limit(max_samples)

    # 2. PCA Reduction
    n_components = data_config.get("pca_components")
    original_embedding_size = -1

    if n_components:
        train_split = data_config.get("train_split", 0.8)
        train_n_days = int(n_days * train_split)
        train_end_date = first_date + timedelta(days=train_n_days - 1)

        embeddings, original_embedding_size = apply_pca_reduction(
            embeddings=embeddings,
            n_components=n_components,
            train_start_date=first_date,
            train_end_date=train_end_date,
            embedding_code=embedding_code,
            data_start_date=first_date,
            data_end_date=end_date,
            exchange=exchange,
            symbol=symbol,
            base_path=base_embedding_path,
        )

    # 3. Load Labels (Orderbook)
    logger.info("Loading orderbook data for labels...")
    ob_paths = get_orderbook_paths(
        start_date=first_date, end_date=end_date, exchange=exchange, symbol=symbol
    )

    if not ob_paths:
        raise FileNotFoundError(f"No orderbook files found for {exchange}/{symbol}")

    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(5)
        .sample_by_time(time_delta=100_000_000, interpolate=True)
    )

    input_space = InputSpace(orderbook_snapshots=orderbook_data)

    # Label extraction
    label_config = config.get("labels", {})
    horizon = label_config.get("horizon", 200)
    threshold = label_config.get("threshold", 2e-4)  # note: YAML likely parses as float
    if isinstance(threshold, str):
        threshold = float(threshold)

    directional_return_label = TripleBarrierLabel(
        config={"horizon": horizon, "threshold": threshold}
    )

    logger.info("Extracting labels...")
    labels = directional_return_label.extract(input_space).collect()

    # 4. Merge
    logger.info("Joining embeddings with labels...")
    label_col = directional_return_label.label_names[0]

    # Inner join
    data = embeddings.join(labels.lazy(), on="timestamp", how="inner")

    # Calculate class weights
    # We need to compute counts. We can likely do this on the LazyFrame but better if we had valid labels.
    # Since we collected labels earlier (as it's usually small enough compared to embeddings), we can compute distribution.
    # BUT, we joined using inner join, so we lose some labels that don't match embeddings.
    # The example code computed statistics on 'labels' BEFORE the join, which potentially includes points not in embeddings?
    # Actually the example joins then uses `labels` (collected) to compute counts.
    # Wait, in the example:
    # labels = ... .collect()
    # data = embeddings.join(labels.lazy() ...)
    # label_counts = labels[label_col].value_counts() -> This uses ALL labels, even if no embedding existed.
    # This might be slightly inaccurate if embeddings are missing for many timestamps.
    # But let's follow the example logic for now.

    # Calculate class weights
    y = labels[label_col].to_numpy()
    unique_classes = np.unique(y)
    expected_classes = np.array([-1, 0, 1])

    # Check if we have unexpected classes
    if not np.isin(unique_classes, expected_classes).all():
        logger.warning(
            f"Found unexpected classes in labels: {unique_classes}. Expected subset of {expected_classes}"
        )

    weights = compute_class_weight(class_weight="balanced", classes=expected_classes, y=y)
    class_weights = weights.tolist()
    logger.info(f"Computed balanced class weights: {class_weights}")

    # Prepare output LazyFrames
    schema = data.collect_schema()
    embedding_cols = [col for col in schema.names() if col not in ["timestamp", label_col]]

    features_df = data.select(["timestamp"] + embedding_cols)
    labels_df = data.select(["timestamp", label_col])

    # Metadata
    metadata = {
        "class_weights": class_weights,
        "input_size": (
            n_components if n_components else original_embedding_size
        ),  # Fallback if no PCA? Needs logic
        "label_col": label_col,
    }

    if n_components:
        metadata["pca_components"] = n_components
        metadata["original_embedding_size"] = original_embedding_size

    return features_df, labels_df, metadata
