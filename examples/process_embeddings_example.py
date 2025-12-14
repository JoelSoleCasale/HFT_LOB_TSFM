"""
Example script showing how to use the embedding processor programmatically.

This demonstrates the EmbeddingProcessor class for generating embeddings
from feature data at scale with hourly granularity.
"""

from loguru import logger

from data_manager.embeddings.embedding_processor import EmbeddingProcessor
from definitions import ROOT_DIR
from utils import setup_logging


def main():
    """Generate embeddings for a specific date range programmatically."""
    # Setup logging
    setup_logging(level="INFO")

    logger.info("=" * 80)
    logger.info("Embedding Generation Example")
    logger.info("=" * 80)

    # Configuration
    base_folder = ROOT_DIR / "data"
    symbol = "BTCUSDT"
    exchange = "binance_futures"
    dates = ["2025-07-01", "2025-07-02", "2025-07-03"]

    # Embedding configuration
    embedding_config = {
        "model_type": "t5",
        "model_size": "mini",  # Use "base" or "large" for better quality
        "seq_aggregation": "last",
        "feat_aggregation": "mean",
        "augment_with_statistics": False,
        "k": 5,
        "use_differencing": False,
        "device": "cuda",  # Change to "cpu" if no GPU
    }

    # Initialize processor
    processor = EmbeddingProcessor(
        base_folder=base_folder,
        embedding_type="chronos",
        embedding_config=embedding_config,
        context_length=512,  # Use 1024 or 2048 for more context
        feature_extractors=["advanced_orderbook"],  # Can add multiple extractors
        orderbook_levels=5,
        sample_time_delta=100_000_000,  # 100ms
        interpolate=True,
        orderbook_subfolder="orderbook_snapshots",
        output_subfolder="embeddings",
    )

    # Process each date
    successful = 0
    failed = 0

    for date_str in dates:
        logger.info(f"\nProcessing {date_str}...")

        try:
            success = processor.process_day(
                symbol=symbol,
                exchange=exchange,
                date_str=date_str,
                skip_existing=True,  # Skip if already processed
            )

            if success:
                successful += 1
                logger.info(f"✓ Successfully processed {date_str}")
            else:
                failed += 1
                logger.warning(f"✗ Failed to process {date_str}")

        except Exception as e:
            failed += 1
            logger.error(f"✗ Error processing {date_str}: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Processing Summary")
    logger.info("=" * 80)
    logger.info(f"Total dates: {len(dates)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    # Show output structure
    output_dir = base_folder / "embeddings" / exchange / symbol
    if output_dir.exists():
        logger.info(f"\nOutput directory: {output_dir}")
        logger.info("Generated hourly files:")
        for date_str in dates:
            date_dir = output_dir / date_str
            if date_dir.exists():
                hourly_files = sorted(date_dir.glob("*.parquet"))
                logger.info(f"  {date_str}: {len(hourly_files)} hourly files")


if __name__ == "__main__":
    main()
