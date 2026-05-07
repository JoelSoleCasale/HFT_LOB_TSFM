import os
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
from features import InputSpace, TripleBarrierLabel, FeaturePipeline, FeatureExtractorRegistry
from data_manager.embeddings.pca import PCAProcessor

_DEFAULT_EMBEDDING_DIR = os.environ.get(
    "EMBEDDING_DATA_DIR",
    str(ROOT_DIR / "data" / "embeddings"),
)


def get_embeddings_paths(
    str_code: str,
    start_date: date,
    end_date: date,
    exchange: str = "binance_futures",
    symbol: str = "BTCUSDT",
    base_path: str = _DEFAULT_EMBEDDING_DIR,
) -> List[Path]:
    """
    Get list of embedding parquet file paths for specified date range.
    """
    dir_path = Path(base_path) / exchange / symbol / str_code

    paths = []
    if not dir_path.exists():
        logger.warning(f"Embedding directory not found: {dir_path}")

    for current_date in date_range(start_date, end_date):
        date_str = current_date.strftime("%Y-%m-%d")
        # Embeddings are hourly: YYYY-MM-DD-HH.parquet (HH from 00 to 23)
        for hour in range(24):
            file_path = dir_path / f"{date_str}-{hour:02d}.parquet"
            if file_path.exists():
                paths.append(file_path)

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
    base_path: str = _DEFAULT_EMBEDDING_DIR,
) -> Tuple[pl.LazyFrame, int]:
    """
    Apply PCA dimensionality reduction using pre-computed PCA model.
    Results are cached to disk to avoid repeated computations.
    """
    all_cols = embeddings.collect_schema().names()
    embedding_cols = [col for col in all_cols if col != "timestamp"]
    original_size = len(embedding_cols)

    cache_dir = Path(base_path) / exchange / symbol / "PCA" / f"{embedding_code}_pca{n_components}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_filename = (
        f"{data_start_date.strftime('%Y-%m-%d')}_{data_end_date.strftime('%Y-%m-%d')}.parquet"
    )
    cache_path = cache_dir / cache_filename

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

    pca_processor = PCAProcessor(
        exchange=exchange,
        symbol=symbol,
        embedding_code=embedding_code,
        base_path=base_path,
    )

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

    logger.info("Transforming embeddings using loaded PCA model...")
    reduced_embeddings = incremental_pca.transform(
        data=embeddings,
        keep_timestamp=True,
    )

    reduced_embeddings = reduced_embeddings.cast({pl.Float64: pl.Float32})

    total_variance = incremental_pca.get_total_explained_variance()
    logger.info(f"Total explained variance: {total_variance:.4f}")

    logger.info(f"Saving PCA-reduced embeddings to cache: {cache_path}")
    reduced_embeddings.sink_parquet(cache_path)

    reduced_embeddings = pl.scan_parquet(cache_path)
    logger.info("Cache saved and reloaded successfully")

    return reduced_embeddings, original_size


def _prepare_embeddings_data(config: dict) -> Tuple[pl.LazyFrame, pl.LazyFrame, dict]:
    """
    Internal handler for embeddings data source.
    """
    data_config = config.get("data", {})
    paths_config = config.get("paths", {})

    first_date = data_config.get("first_date")
    n_days = data_config.get("n_days", 10)
    if isinstance(first_date, str):
        first_date = date.fromisoformat(first_date)

    end_date = first_date + timedelta(days=n_days - 1)

    embedding_code = data_config.get("embedding_code")
    max_samples = data_config.get("max_total_samples", 100_000_000)

    base_embedding_path = paths_config.get("embeddings_dir", _DEFAULT_EMBEDDING_DIR)
    exchange = data_config.get("exchange", "binance_futures")
    symbol = data_config["symbol"]

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
    else:
        schema = embeddings.collect_schema()
        original_embedding_size = len([c for c in schema.names() if c != "timestamp"])

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

    label_config = config.get("labels", {})
    horizon = label_config.get("horizon", 200)
    threshold = label_config.get("threshold", 2e-4)
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
    y = labels[label_col].to_numpy()
    unique_classes = np.unique(y)
    expected_classes = np.array([-1, 0, 1])

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
        "input_size": (n_components if n_components else original_embedding_size),
        "label_col": label_col,
    }

    if n_components:
        metadata["pca_components"] = n_components
        metadata["original_embedding_size"] = original_embedding_size

    return features_df, labels_df, metadata


def _prepare_orderbook_data(config: dict) -> Tuple[pl.LazyFrame, pl.LazyFrame, dict]:
    """
    Internal handler for raw orderbook data source with feature extraction pipeline.
    """
    data_config = config.get("data", {})

    first_date = data_config.get("first_date")
    n_days = data_config.get("n_days", 10)
    if isinstance(first_date, str):
        first_date = date.fromisoformat(first_date)

    end_date = first_date + timedelta(days=n_days - 1)

    exchange = data_config.get("exchange", "binance_futures")
    symbol = data_config["symbol"]

    # 1. Load Orderbook
    logger.info("Loading orderbook data for features and labels...")
    ob_paths = get_orderbook_paths(
        start_date=first_date, end_date=end_date, exchange=exchange, symbol=symbol
    )

    if not ob_paths:
        raise FileNotFoundError(f"No orderbook files found for {exchange}/{symbol}")

    # Standard loading for DeepLOB-like models
    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(data_config.get("levels", 10))
        .sample_by_time(time_delta=data_config.get("time_delta", 100_000_000), interpolate=True)
    )

    input_space = InputSpace(orderbook_snapshots=orderbook_data)

    # 2. Extract Features
    logger.info("Extracting features from pipeline...")
    pipeline = FeaturePipeline()
    feature_configs = data_config.get("features", [])

    # Handle case where config generator unwrapped the list (grid search artifact)
    if isinstance(feature_configs, dict):
        feature_configs = [feature_configs]

    if not feature_configs:
        # Default fallback if no features specified?
        # Or error out. Better invoke at least one extractor.
        logger.warning("No feature extractors specified in config 'data.features'.")

    for feat_conf in feature_configs:
        name = feat_conf.get("name")
        params = feat_conf.get("params", {})
        pipeline.add_extractor(FeatureExtractorRegistry.create(name, config=params))

    features = pipeline.extract_all(input_space)

    # Calculate input size
    schema = features.collect_schema()
    # Subtract timestamp
    input_size = len([c for c in schema.names() if c != "timestamp"])

    # 3. Extract Labels
    label_config = config.get("labels", {})
    horizon = label_config.get("horizon", 200)
    threshold = label_config.get("threshold", 2e-4)
    if isinstance(threshold, str):
        threshold = float(threshold)

    directional_return_label = TripleBarrierLabel(
        config={"horizon": horizon, "threshold": threshold}
    )

    logger.info("Extracting labels...")
    labels = directional_return_label.extract(input_space)

    # We collect labels to compute weights (DeepLOB example does this)
    # Note: features are kept lazy
    labels_collected = labels.collect()

    # 4. Calculate Class Weights
    label_col = directional_return_label.label_names[0]
    y = labels_collected[label_col].to_numpy()
    unique_classes = np.unique(y)
    expected_classes = np.array([-1, 0, 1])

    if not np.isin(unique_classes, expected_classes).all():
        logger.warning(f"Unexpected classes: {unique_classes}")

    weights = compute_class_weight(class_weight="balanced", classes=expected_classes, y=y)
    class_weights = weights.tolist()
    logger.info(f"Computed balanced class weights: {class_weights}")

    # Metadata
    metadata = {
        "class_weights": class_weights,
        "input_size": input_size,
        "label_col": label_col,
    }

    labels_lazy = labels_collected.lazy()

    # Join to enforce timestamp alignment, mirroring the embeddings path.
    data = features.join(labels_lazy, on="timestamp", how="inner")

    features_df = data.select(["timestamp"] + [c for c in schema.names() if c != "timestamp"])
    labels_df = data.select(["timestamp", label_col])

    return features_df, labels_df, metadata


def prepare_data(config: dict) -> Tuple[pl.LazyFrame, pl.LazyFrame, dict]:
    """
    Prepare features and labels based on configuration.
    Returns (features_df, labels_df, metadata)
    """
    # Extract config parameters
    data_config = config.get("data", {})

    data_source = data_config.get("source", "embeddings")

    if data_source == "embeddings":
        return _prepare_embeddings_data(config)
    elif data_source == "orderbook":
        return _prepare_orderbook_data(config)
    else:
        raise ValueError(
            f"Unknown data source in config: {data_source}. Expected 'embeddings' or 'orderbook'."
        )
