import polars as pl
from datetime import date, timedelta
from loguru import logger

from definitions import ROOT_DIR
from utils import date_range, get_ob_path
from core.orderbook import OrderBook
from features import (
    InputSpace,
    FeaturePipeline,
    FeatureExtractorRegistry,
)
from embeddings import EmbeddingPipeline


def main() -> None:
    """Main function to generate embeddings."""
    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 1
    N_SAMPLES = 10_000
    CONTEXT_LENGTH = 512

    logger.info(f"Loading orderbook data from {FIRST_DATE} for {N_DAYS} day(s)...")
    ob_paths = [
        get_ob_path(d) for d in date_range(FIRST_DATE, FIRST_DATE + timedelta(days=N_DAYS))
    ]
    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(5)
        .sample_by_time(time_delta=100_000_000, interpolate=True)
    )

    # Skip first 10 samples (first second) via private attribute — OrderBook has no
    # public slice API that rewrites the internal frame without re-validation.
    orderbook_data._data._df = orderbook_data.df[10:]

    logger.info("Creating input space and feature pipeline...")
    input_space = InputSpace(orderbook_snapshots=orderbook_data)

    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(FeatureExtractorRegistry.create("advanced_orderbook"))

    logger.info("Extracting features...")
    features: pl.LazyFrame = feature_pipeline.extract_all(input_space)

    features_schema = features.collect_schema()
    logger.info(f"Features columns: {len(features_schema)}")

    max_samples = N_SAMPLES + CONTEXT_LENGTH
    features_subset = features.slice(0, max_samples)

    logger.info(f"Using up to {max_samples} samples for embedding generation")

    embed_config = {
        "context_length": CONTEXT_LENGTH,
        "model_type": "bolt",
        "model_size": "mini",
        "seq_aggregation": "mean",
        "augment_with_statistics": True,
        "k": 8,
        "use_differencing": False,
        "device": "cuda",
    }

    logger.info("Creating embedding pipeline...")
    logger.info(f"Embedding config: {embed_config}")

    embedding_pipeline = EmbeddingPipeline()
    embedding_pipeline.add_generator(
        "chronos",
        name="chronos_bolt",
        config={
            "model_type": embed_config["model_type"],
            "model_size": embed_config["model_size"],
            "seq_aggregation": embed_config["seq_aggregation"],
            "augment_with_statistics": embed_config["augment_with_statistics"],
            "k": embed_config["k"],
            "use_differencing": embed_config["use_differencing"],
            "device": embed_config["device"],
        },
    )

    logger.info("Generating embeddings...")
    embeddings = embedding_pipeline.generate(
        features_subset, context_length=embed_config["context_length"]
    )

    embeddings_df = embeddings.collect()

    logger.info(f"Generated {len(embeddings_df)} embeddings")
    logger.info(f"Embeddings shape: {embeddings_df.shape}")

    if "timestamp" not in embeddings_df.columns:
        logger.error("Embeddings do not contain 'timestamp' column!")

    output_dir = ROOT_DIR / "data" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_suffix = "_stats" if embed_config["augment_with_statistics"] else ""
    diff_suffix = "_diff" if embed_config["use_differencing"] else ""
    filename = (
        f"embeddings_N-{len(embeddings_df)}_chronos_{embed_config['model_type']}_"
        f"{embed_config['model_size']}_seq-{embed_config['seq_aggregation']}"
        f"_ctx-{embed_config['context_length']}{stats_suffix}_k-{embed_config['k']}{diff_suffix}.parquet"
    )
    output_path = output_dir / filename

    logger.info(f"Saving embeddings to {output_path}")
    embeddings_df.write_parquet(output_path)

    logger.info("Embedding generation complete!")
    logger.info(f"Saved {len(embeddings_df)} embeddings to {output_path}")
    logger.info(f"  - Number of embeddings: {len(embeddings_df)}")
    logger.info(f"  - Embedding dimension: {len(embeddings_df.columns) - 1}")
    logger.info(f"  - Feature columns: {len(features_schema) - 1}")

    return embeddings_df, output_path


if __name__ == "__main__":
    embeddings_df, output_path = main()
