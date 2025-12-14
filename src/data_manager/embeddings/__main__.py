"""
CLI for processing orderbook data to generate embeddings at scale.

This module provides a command-line interface for generating embeddings from
orderbook snapshots, extracting features and processing multiple days with
hourly granularity to handle large datasets efficiently.
"""

import yaml
from loguru import logger
from datetime import date, timedelta
import argparse

from data_manager.embeddings.embedding_processor import EmbeddingProcessor
from definitions import ROOT_DIR
from utils import setup_logging


def load_config() -> dict:
    """
    Load configuration from YAML files.

    Returns:
        Merged configuration dictionary
    """
    config_path = ROOT_DIR / "config" / "config.yaml"
    embeddings_config_path = ROOT_DIR / "config" / "data_manager" / "embeddings" / "default.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(embeddings_config_path, "r") as f:
        embeddings_config = yaml.safe_load(f)

    config["embeddings"] = embeddings_config
    return config


def parse_args(default_cfg: dict) -> argparse.Namespace:
    """
    Parse command-line arguments with defaults from configuration.

    Args:
        default_cfg: Default configuration from YAML

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate embeddings from feature data with hourly granularity."
    )

    # Set defaults from configuration
    parser.set_defaults(
        log_level=default_cfg.get("log_level", "INFO"),
        data_folder=str(ROOT_DIR / default_cfg.get("data_folder", "data")),
        embedding_type=default_cfg["embeddings"]["embedding_type"],
        context_length=default_cfg["embeddings"]["context_length"],
        feature_extractors=default_cfg["embeddings"]["feature_extractors"],
        orderbook_levels=default_cfg["embeddings"]["orderbook_levels"],
        sample_time_delta=default_cfg["embeddings"]["sample_time_delta"],
        interpolate=default_cfg["embeddings"]["interpolate"],
        symbol=default_cfg["embeddings"]["symbol"],
        exchange=default_cfg["embeddings"]["exchange"],
        orderbook_subfolder=default_cfg["embeddings"]["orderbook_subfolder"],
        output_subfolder=default_cfg["embeddings"]["output_subfolder"],
        start_date=default_cfg["embeddings"]["dates"]["start_date"],
        end_date=default_cfg["embeddings"]["dates"]["end_date"],
        overwrite_existing=default_cfg["embeddings"]["overwrite_existing"],
    )

    # General arguments
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Base folder for data",
    )

    # Embedding configuration
    parser.add_argument(
        "--embedding_type",
        type=str,
        help="Type of embedding generator (e.g., 'chronos')",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["t5", "bolt", "chronos2"],
        help="Model type for Chronos embeddings",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["mini", "small", "base", "large"],
        help="Model size for Chronos embeddings",
    )
    parser.add_argument(
        "--seq_aggregation",
        type=str,
        choices=["last", "mean"],
        help="Sequence aggregation method",
    )
    parser.add_argument(
        "--feat_aggregation",
        type=str,
        choices=["concat", "mean"],
        help="Feature aggregation method",
    )
    parser.add_argument(
        "--augment_with_statistics",
        action=argparse.BooleanOptionalAction,
        help="Augment embeddings with k-patch statistics",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of patches for statistics",
    )
    parser.add_argument(
        "--use_differencing",
        action=argparse.BooleanOptionalAction,
        help="Include differenced embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device for model inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of samples to process in parallel",
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="Generate embeddings only for timestamps where timestamp %% stride == 0",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        help="Number of samples to use as context",
    )

    # Feature extraction configuration
    parser.add_argument(
        "--feature_extractors",
        type=str,
        nargs="+",
        help="List of feature extractor names from registry",
    )
    parser.add_argument(
        "--orderbook_levels",
        type=int,
        help="Number of orderbook levels to use",
    )
    parser.add_argument(
        "--sample_time_delta",
        type=int,
        help="Time delta for sampling in nanoseconds",
    )
    parser.add_argument(
        "--interpolate",
        action=argparse.BooleanOptionalAction,
        help="Whether to interpolate missing values",
    )

    # Data configuration
    parser.add_argument(
        "--symbol",
        type=str,
        nargs="+",
        help="Trading symbols to process",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        nargs="+",
        help="Exchanges to process",
    )
    parser.add_argument(
        "--orderbook_subfolder",
        type=str,
        help="Subfolder containing orderbook files",
    )
    parser.add_argument(
        "--output_subfolder",
        type=str,
        help="Subfolder for output embeddings",
    )

    # Date range
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date in YYYY-MM-DD format (defaults to today if not provided)",
    )

    # Processing options
    parser.add_argument(
        "--overwrite_existing",
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing embedding files",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable progress bars when generating embeddings",
    )

    args = parser.parse_args()
    return args, default_cfg


def build_embedding_config(args: argparse.Namespace, default_cfg: dict) -> dict:
    """
    Build embedding configuration from arguments and defaults.

    Args:
        args: Parsed command-line arguments
        default_cfg: Default configuration from YAML

    Returns:
        Embedding configuration dictionary
    """
    base_config = default_cfg["embeddings"]["embedding_config"]

    # Override with command-line arguments if provided
    embedding_config = {
        "model_type": args.model_type if args.model_type else base_config["model_type"],
        "model_size": args.model_size if args.model_size else base_config["model_size"],
        "seq_aggregation": (
            args.seq_aggregation if args.seq_aggregation else base_config["seq_aggregation"]
        ),
        "feat_aggregation": (
            args.feat_aggregation if args.feat_aggregation else base_config["feat_aggregation"]
        ),
        "augment_with_statistics": (
            args.augment_with_statistics
            if args.augment_with_statistics is not None
            else base_config["augment_with_statistics"]
        ),
        "k": args.k if args.k else base_config["k"],
        "use_differencing": (
            args.use_differencing
            if args.use_differencing is not None
            else base_config["use_differencing"]
        ),
        "device": args.device if args.device else base_config["device"],
        "batch_size": args.batch_size if args.batch_size else base_config["batch_size"],
        "stride": args.stride if args.stride else base_config["stride"],
    }

    return embedding_config


def main() -> None:
    """
    Main function to run the embedding processor.
    """
    # Load configuration
    default_cfg = load_config()
    args, default_cfg = parse_args(default_cfg)

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 80)
    logger.info("Embedding Processor CLI")
    logger.info("=" * 80)

    # Build embedding configuration
    embedding_config = build_embedding_config(args, default_cfg)

    # Add disable_tqdm to embedding config if chronos
    if args.embedding_type == "chronos":
        embedding_config["disable_tqdm"] = args.no_tqdm

    logger.debug("Configuration:")
    logger.debug(f"  Embedding type: {args.embedding_type}")
    logger.debug(f"  Context length: {args.context_length}")
    logger.debug(f"  Embedding config: {yaml.dump(embedding_config, default_flow_style=False)}")
    logger.debug(f"  Feature extractors: {args.feature_extractors}")
    logger.debug(f"  Orderbook levels: {args.orderbook_levels}")
    logger.debug(f"  Sample time delta: {args.sample_time_delta}ns")
    logger.debug(f"  Interpolate: {args.interpolate}")
    logger.debug(f"  Symbols: {args.symbol}")
    logger.debug(f"  Exchanges: {args.exchange}")
    logger.debug(f"  Orderbook subfolder: {args.orderbook_subfolder}")
    logger.debug(f"  Output subfolder: {args.output_subfolder}")
    logger.debug(f"  Disable tqdm: {args.no_tqdm}")

    # Initialize processor
    processor = EmbeddingProcessor(
        base_folder=args.data_folder,
        embedding_type=args.embedding_type,
        embedding_config=embedding_config,
        context_length=args.context_length,
        feature_extractors=args.feature_extractors,
        orderbook_levels=args.orderbook_levels,
        sample_time_delta=args.sample_time_delta,
        interpolate=args.interpolate,
        orderbook_subfolder=args.orderbook_subfolder,
        output_subfolder=args.output_subfolder,
    )

    # Handle dates
    start_date_str = args.start_date
    end_date_str = args.end_date

    start_date = date.fromisoformat(start_date_str)
    end_date = date.today() if end_date_str is None else date.fromisoformat(end_date_str)

    # Generate list of dates to process
    dates = [
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end_date - start_date).days + 1)
    ]

    logger.info(f"Processing {len(dates)} days from {start_date_str} to {end_date}")
    logger.info(f"Symbols: {', '.join(args.symbol)}")
    logger.info(f"Exchanges: {', '.join(args.exchange)}")

    # Process all combinations of symbol, exchange, and date
    total_tasks = len(args.symbol) * len(args.exchange) * len(dates)
    completed = 0
    failed = 0

    for symbol in args.symbol:
        for exchange in args.exchange:
            for date_str in dates:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Task {completed + failed + 1}/{total_tasks}")
                logger.info(f"Symbol: {symbol}, Exchange: {exchange}, Date: {date_str}")
                logger.info(f"{'=' * 80}")

                try:
                    success = processor.process_day(
                        symbol=symbol,
                        exchange=exchange,
                        date_str=date_str,
                        skip_existing=not args.overwrite_existing,
                    )

                    if success:
                        completed += 1
                    else:
                        failed += 1
                        logger.warning(f"Failed to process {symbol}/{exchange}/{date_str}")

                except Exception as e:
                    failed += 1
                    logger.error(
                        f"Error processing {symbol}/{exchange}/{date_str}: {e}",
                        exc_info=True,
                    )

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Processing Complete")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {completed / total_tasks * 100:.1f}%")


if __name__ == "__main__":
    main()
