"""
Example script showing how to train a model on Chronos embeddings.

This script demonstrates how to:
1. Load pre-generated Chronos embeddings
2. Load orderbook data and generate labels
3. Train an MLP model on the embeddings (sequence_length=1)
4. Evaluate the trained model
"""

from pathlib import Path
import warnings
from datetime import date, timedelta
import argparse
import polars as pl

from features import (
    InputSpace,
    TripleBarrierLabel,
)
from models import (
    ModelConfig,
    MLPConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
    train_model,
)
from utils import date_range, setup_logging
from definitions import ROOT_DIR
from core.orderbook import OrderBook
from data_manager.embeddings.pca import PCAProcessor
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028


def get_embeddings_paths(
    str_code: str,
    start_date: date,
    end_date: date,
    exchange: str = "binance_futures",
    symbol: str = "BTCUSDT",
    base_path: str = "/scratch/PI/palomar/joel_sole/embedding_data",
) -> list[Path]:
    """
    Get list of embedding parquet file paths for specified date range.

    Args:
        str_code: Embedding configuration code (e.g., "chronos2-base_ctx512_seqlast_s5")
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        exchange: Exchange name
        symbol: Trading symbol
        base_path: Base directory containing embedding data

    Returns:
        List of paths to embedding parquet files
    """
    dir_path = Path(base_path) / exchange / symbol / str_code
    if not dir_path.exists():
        raise FileNotFoundError(f"Embedding directory not found: {dir_path}")

    paths = []
    for current_date in date_range(start_date, end_date):
        date_str = current_date.strftime("%Y-%m-%d")
        # Embeddings are hourly: YYYY-MM-DD-HH.parquet (HH from 00 to 23)
        for hour in range(24):
            file_path = dir_path / f"{date_str}-{hour:02d}.parquet"
            if file_path.exists():
                paths.append(file_path)

    if not paths:
        raise FileNotFoundError(
            f"No embedding files found for {exchange}/{symbol}/{str_code} "
            f"in date range {start_date} to {end_date}"
        )

    return sorted(paths)


def get_orderbook_paths(
    start_date: date,
    end_date: date,
    exchange: str = "binance_futures",
    symbol: str = "BTCUSDT",
    levels: int = 20,
) -> list[Path]:
    """
    Get list of orderbook parquet file paths for specified date range.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        exchange: Exchange name
        symbol: Trading symbol
        levels: Number of orderbook levels

    Returns:
        List of paths to orderbook parquet files
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

    if not paths:
        raise FileNotFoundError(
            f"No orderbook files found for {exchange}/{symbol} "
            f"in date range {start_date} to {end_date}"
        )

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
) -> tuple[pl.LazyFrame, int]:
    """
    Apply PCA dimensionality reduction using pre-computed PCA model.
    Results are cached to disk to avoid repeated computations.

    Args:
        embeddings: LazyFrame with embeddings and timestamp
        n_components: Number of PCA components
        train_start_date: Start date of training period used to fit PCA
        train_end_date: End date of training period used to fit PCA
        embedding_code: Embedding configuration code
        data_start_date: Start date of the data being reduced
        data_end_date: End date of the data being reduced
        exchange: Exchange name
        symbol: Trading symbol
        base_path: Base directory containing embedding data

    Returns:
        Tuple of (LazyFrame with PCA-transformed embeddings and timestamp, original embedding size)
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


def main():
    """Main function demonstrating model training on embeddings."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train MLP model on Chronos embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embedding-code",
        type=str,
        default="chronos2-base_ctx512_seqlast_s5",
        help="Embedding configuration code (e.g., chronos2-base_ctx512_seqlast_s5)",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[64],
        help="List of hidden layer sizes for MLP (e.g., --hidden-sizes 64 128 64). Use --hidden-sizes with no values for empty list (linear model)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=512,
        help="Number of PCA components to reduce embeddings to",
    )
    args = parser.parse_args()

    setup_logging(level="DEBUG")

    # Configuration
    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 10

    MAX_TOTAL_SAMPLES = 100_000_000  # Limit total samples to avoid memory issues

    # Embedding configuration from arguments
    EMBEDDING_CODE = args.embedding_code

    # Calculate date range
    END_DATE = FIRST_DATE + timedelta(days=N_DAYS - 1)

    print(f"Loading embeddings: {EMBEDDING_CODE}...")
    print(f"Date range: {FIRST_DATE} to {END_DATE}")

    # Get embedding paths for date range
    embedding_paths = get_embeddings_paths(
        str_code=EMBEDDING_CODE,
        start_date=FIRST_DATE,
        end_date=END_DATE,
    )
    print(f"Found {len(embedding_paths)} embedding files")

    # Load embeddings (keep as LazyFrame to avoid memory issues)
    embeddings = pl.scan_parquet(embedding_paths).limit(MAX_TOTAL_SAMPLES)
    print("Embeddings loaded (LazyFrame)")

    # ============================================================================
    # PCA Dimensionality Reduction (before joining with labels)
    # ============================================================================
    N_COMPONENTS = args.n_components  # Number of PCA components to keep
    TRAIN_SPLIT = 0.8  # Must match DataConfig train_split

    # Calculate training period end date (for loading pre-computed PCA)
    train_n_days = int(N_DAYS * TRAIN_SPLIT)
    train_end_date = FIRST_DATE + timedelta(days=train_n_days - 1)

    print(f"\nPCA training period: {FIRST_DATE} to {train_end_date} ({train_n_days} days)")

    # Apply PCA using pre-computed model (with caching)
    embeddings, original_embedding_size = apply_pca_reduction(
        embeddings=embeddings,
        n_components=N_COMPONENTS,
        train_start_date=FIRST_DATE,
        train_end_date=train_end_date,
        embedding_code=EMBEDDING_CODE,
        data_start_date=FIRST_DATE,
        data_end_date=END_DATE,
    )

    print(f"Original embedding dimension: {original_embedding_size}")
    print(f"Reduced embedding dimension: {N_COMPONENTS}")

    # ============================================================================
    # End PCA Reduction
    # ============================================================================

    print("\nLoading orderbook data for labels...")

    # Get orderbook paths for date range
    ob_paths = get_orderbook_paths(
        start_date=FIRST_DATE,
        end_date=END_DATE,
    )
    print(f"Found {len(ob_paths)} orderbook files")
    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(5)
        .sample_by_time(time_delta=100_000_000, interpolate=True)
    )

    # Create input space for label generation
    input_space = InputSpace(
        orderbook_snapshots=orderbook_data,
    )

    # Build label pipeline
    directional_return_label = TripleBarrierLabel(config={"horizon": 200, "threshold": 2e-4})

    print("Extracting labels...")
    labels = directional_return_label.extract(input_space).collect()
    print(f"Labels shape: {labels.shape}")

    # Join embeddings with labels on timestamp
    label_col = directional_return_label.label_names[0]

    print("\nJoining embeddings with labels...")

    # Verify all embedding timestamps exist in labels (anti join to find missing)
    missing = embeddings.join(labels.select(["timestamp"]).lazy(), on="timestamp", how="anti")
    missing_count = missing.select(pl.len()).collect().item()
    if missing_count > 0:
        sample_missing = missing.select("timestamp").head(5).collect()["timestamp"].to_list()
        print(f"Warning: Found {missing_count} embedding timestamps not present in labels.")
        print(f"Examples: {sample_missing}")

    # Join labels to match embedding timestamps (inner join to keep only aligned data)
    data = embeddings.join(labels.lazy(), on="timestamp", how="inner")
    print("Aligned data joined (LazyFrame)")

    # Print label distribution from collected labels
    label_counts = labels[label_col].value_counts()
    total_samples = len(labels)

    print("\nLabel distribution:")
    for label_value, count in label_counts.iter_rows():
        percentage = (count / total_samples) * 100
        print(f"  Label {label_value}: {count:,} samples ({percentage:.2f}%)")

    # Calculate class weights for loss function
    class_weights = [0.0, 0.0, 0.0]
    for label_value, count in label_counts.iter_rows():
        freq = count / total_samples
        class_weights[label_value + 1] = 1 / freq

    # Prepare features and labels as LazyFrames for the model
    # Separate features from labels (data is already a LazyFrame)
    # Get column names by collecting schema
    all_cols = data.collect_schema().names()
    embedding_cols = [col for col in all_cols if col not in ["timestamp", label_col]]
    features_df = data.select(["timestamp"] + embedding_cols)
    labels_df = data.select(["timestamp", label_col])

    # Input size is now the reduced PCA dimension
    input_size = N_COMPONENTS

    print(f"\nHidden sizes for MLP: {args.hidden_sizes}")

    # Create MLP configuration for embedding-based training
    mlp_config = MLPConfig(
        input_size=input_size,
        output_size=3,  # -1, 0, 1 for directional labels
        hidden_sizes=args.hidden_sizes,  # From command-line args
        dropout=0.2,
        sequence_length=1,  # Each embedding is a single timestamp
    )

    # Parse embedding code to extract components
    embedding_parts = EMBEDDING_CODE.split("_")
    embedding_info = {}
    for part in embedding_parts:
        if part.startswith("ctx"):
            embedding_info["context_length"] = int(part[3:])
        elif part.startswith("seq"):
            embedding_info["sequence_method"] = part[3:]
        elif part.startswith("s"):
            try:
                embedding_info["sampling_rate"] = int(part[1:])
            except ValueError:
                pass  # Not a sampling rate
        elif "-" in part:
            # This is likely the model name (e.g., "chronos2-base")
            embedding_info["embedding_model"] = part

    # Collect all relevant training metadata
    other_info = {
        "pca_components": N_COMPONENTS,
        "original_embedding_size": original_embedding_size,
        "embedding_code": EMBEDDING_CODE,
        **embedding_info,
        "first_date": FIRST_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "num_days": N_DAYS,
        "label_horizon": directional_return_label.config["horizon"],
        "label_threshold": directional_return_label.config["threshold"],
        "max_total_samples": MAX_TOTAL_SAMPLES,
    }

    config = ModelConfig(
        architecture=mlp_config,
        data=DataConfig(
            sequence_length=1,  # Only use current timestamp embedding
            batch_size=1024,
            stride=5,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            device="cuda",
        ),
        training=TrainingConfig(
            learning_rate=0.001,
            num_epochs=100,
            early_stopping_patience=10,
            optimizer="adamw",
            scheduler="cosine",
            loss_function="focal",
            loss_params={"alpha": class_weights, "gamma": 2.0},
            mixed_precision=False,
        ),
        logging=LoggingConfig(
            project_name="E_pca_mlp_training",
            experiment_name=f"mlp-{args.hidden_sizes}_pca-{args.n_components}_E-{EMBEDDING_CODE}",
            wandb_enabled=True,
            log_confusion_matrix=True,
            log_trade_accuracy_vs_threshold=True,
            lambda_value=directional_return_label.threshold,
            theta_values=[0.0, 1e-4, 4e-4],
        ),
        other=other_info,
    )

    print("\nStarting model training...")
    print(f"Model configuration: {config}")

    # Train the model
    trainer, results = train_model(
        features=features_df,
        labels=labels_df,
        config=config,
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
