"""
CLI for precomputing PCA models on embeddings at scale.

This module provides a command-line interface for fitting and caching PCA models
on pre-generated embeddings, supporting multiple exchanges, symbols, and embedding
configurations with flexible date ranges.
"""

import argparse
from datetime import date

import yaml
from loguru import logger

from data_manager.embeddings.pca.pca_processor import PCAProcessor
from definitions import ROOT_DIR
from embeddings.pca import PCAConfig
from utils import setup_logging


def load_config() -> dict:
    """
    Load configuration from YAML files.

    Returns:
        Merged configuration dictionary
    """
    config_path = ROOT_DIR / "config" / "config.yaml"
    embeddings_config_path = ROOT_DIR / "config" / "data_manager" / "embeddings" / "default.yaml"
    pca_config_path = ROOT_DIR / "config" / "data_manager" / "embeddings" / "pca" / "default.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(embeddings_config_path, "r") as f:
        embeddings_config = yaml.safe_load(f)

    with open(pca_config_path, "r") as f:
        pca_config = yaml.safe_load(f)

    config["embeddings"] = embeddings_config
    config["pca"] = pca_config
    return config


def parse_args(default_cfg: dict) -> tuple[argparse.Namespace, dict]:
    """
    Parse command-line arguments with defaults from configuration.

    Args:
        default_cfg: Default configuration from YAML

    Returns:
        Tuple of (parsed arguments, default config)
    """
    parser = argparse.ArgumentParser(
        description="Precompute PCA models on embeddings for specified date ranges."
    )

    pca_cfg = default_cfg["pca"]
    embeddings_cfg = default_cfg["embeddings"]

    # Set defaults from configuration
    parser.set_defaults(
        log_level=default_cfg.get("log_level", "INFO"),
        n_components=pca_cfg["n_components"],
        chunk_size=pca_cfg["chunk_size"],
        whiten=pca_cfg["whiten"],
        exchanges=pca_cfg["exchanges"],
        symbols=pca_cfg["symbols"],
        embedding_codes=pca_cfg["embedding_codes"],
        start_date=pca_cfg["dates"]["start_date"],
        end_date=pca_cfg["dates"]["end_date"],
        base_path=embeddings_cfg["output_subfolder"],  # Use embeddings output path
        pca_model_path=pca_cfg["pca_model_path"],
        max_samples=pca_cfg["max_samples"],
        force_recompute=pca_cfg["force_recompute"],
        verbose=pca_cfg["verbose"],
    )

    # General arguments
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # PCA configuration
    parser.add_argument(
        "--n_components",
        type=int,
        help="Number of principal components to keep",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Number of rows to process per chunk",
    )
    parser.add_argument(
        "--whiten",
        action=argparse.BooleanOptionalAction,
        help="Whether to whiten the transformed components",
    )

    # Data configuration
    parser.add_argument(
        "--exchanges",
        type=str,
        nargs="+",
        help="Exchanges to process (e.g., binance_futures)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Trading symbols to process (e.g., BTCUSDT)",
    )
    parser.add_argument(
        "--embedding_codes",
        type=str,
        nargs="+",
        help="Embedding configuration codes to process",
    )

    # Date range
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for PCA training in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for PCA training in YYYY-MM-DD format (defaults to today if not provided)",
    )

    # Paths
    parser.add_argument(
        "--base_path",
        type=str,
        help="Base directory containing embedding data",
    )
    parser.add_argument(
        "--pca_model_path",
        type=str,
        help="Directory to save PCA models (defaults to ROOT_DIR/models/pca)",
    )

    # Processing options
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to use for fitting (None for all)",
    )
    parser.add_argument(
        "--force_recompute",
        action=argparse.BooleanOptionalAction,
        help="Force recomputation even if PCA model already exists",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="Show detailed progress information",
    )

    args = parser.parse_args()
    return args, default_cfg


def main() -> None:
    """
    Main function to run the PCA processor.
    """
    # Load configuration
    default_cfg = load_config()
    args, default_cfg = parse_args(default_cfg)

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 80)
    logger.info("PCA Processor CLI")
    logger.info("=" * 80)

    logger.debug("Configuration:")
    logger.debug(f"  n_components: {args.n_components}")
    logger.debug(f"  chunk_size: {args.chunk_size:,}")
    logger.debug(f"  whiten: {args.whiten}")
    logger.debug(f"  Exchanges: {args.exchanges}")
    logger.debug(f"  Symbols: {args.symbols}")
    logger.debug(f"  Embedding codes: {args.embedding_codes}")
    logger.debug(f"  Date range: {args.start_date} to {args.end_date or 'today'}")
    logger.debug(f"  Base path: {args.base_path}")
    logger.debug(f"  PCA model path: {args.pca_model_path or 'ROOT_DIR/data/embeddings/pca'}")
    logger.debug(f"  Max samples: {args.max_samples or 'all'}")
    logger.debug(f"  Force recompute: {args.force_recompute}")

    # Handle dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.today() if args.end_date is None else date.fromisoformat(args.end_date)

    logger.info(f"\nProcessing PCA for date range: {start_date} to {end_date}")
    logger.info(f"Exchanges: {', '.join(args.exchanges)}")
    logger.info(f"Symbols: {', '.join(args.symbols)}")
    logger.info(f"Embedding codes: {', '.join(args.embedding_codes)}")

    # Process all combinations of exchange, symbol, and embedding code
    total_tasks = len(args.exchanges) * len(args.symbols) * len(args.embedding_codes)
    completed = 0
    failed = 0
    skipped = 0

    for exchange in args.exchanges:
        for symbol in args.symbols:
            for embedding_code in args.embedding_codes:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Task {completed + failed + skipped + 1}/{total_tasks}")
                logger.info(f"Exchange: {exchange}, Symbol: {symbol}, Embedding: {embedding_code}")
                logger.info(f"{'=' * 80}")

                try:
                    # Initialize processor
                    processor = PCAProcessor(
                        exchange=exchange,
                        symbol=symbol,
                        embedding_code=embedding_code,
                        base_path=args.base_path,
                        pca_model_path=args.pca_model_path,
                    )

                    # Check if model already exists
                    if not args.force_recompute and processor.model_exists(
                        args.n_components, start_date, end_date
                    ):
                        model_path = processor.get_model_save_path(
                            args.n_components, start_date, end_date
                        )
                        logger.info(
                            f"PCA model already exists at {model_path}. Skipping. "
                            "Use --force_recompute to recompute."
                        )
                        skipped += 1
                        continue

                    # Create PCA configuration
                    pca_config = PCAConfig(
                        n_components=args.n_components,
                        chunk_size=args.chunk_size,
                        whiten=args.whiten,
                        save_path=None,  # Will be set by processor
                    )

                    # Fit PCA
                    pca_processor = processor.fit_pca(
                        config=pca_config,
                        start_date=start_date,
                        end_date=end_date,
                        max_samples=args.max_samples,
                        force_recompute=args.force_recompute,
                        verbose=args.verbose,
                    )

                    logger.info(
                        f"✓ Successfully fitted PCA for {exchange}/{symbol}/{embedding_code}"
                    )
                    logger.info(
                        f"  Explained variance: {pca_processor.get_total_explained_variance():.4f}"
                    )
                    logger.info(f"  Samples seen: {pca_processor.get_n_samples_seen():,}")

                    completed += 1

                except FileNotFoundError as e:
                    failed += 1
                    logger.error(f"File not found: {e}")

                except Exception as e:
                    failed += 1
                    logger.error(
                        f"Error processing {exchange}/{symbol}/{embedding_code}: {e}",
                        exc_info=True,
                    )

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Processing Complete")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Skipped (already exists): {skipped}")
    logger.info(f"Failed: {failed}")
    if total_tasks > 0:
        logger.info(
            f"Success rate: {(completed + skipped) / total_tasks * 100:.1f}% "
            f"({completed} new, {skipped} existing)"
        )


if __name__ == "__main__":
    main()
