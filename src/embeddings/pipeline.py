"""Pipeline for generating embeddings from features"""

import polars as pl
from typing import Any
from loguru import logger

from embeddings.base import BaseEmbeddingGenerator
from embeddings.registry import EmbeddingGeneratorRegistry


class EmbeddingPipeline:
    """
    Pipeline for generating embeddings from feature dataframes.

    Supports single or multiple embedding generators, with optional
    feature selection and post-processing.
    """

    def __init__(self):
        """Initialize an empty embedding pipeline."""
        self.generators: list[BaseEmbeddingGenerator] = []
        self.generator_names: list[str] = []

    def add_generator(
        self,
        generator: BaseEmbeddingGenerator | str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> "EmbeddingPipeline":
        """
        Add an embedding generator to the pipeline.

        Args:
            generator: Either a generator instance or a registered generator name
            name: Optional name for the generator (used for column naming)
            config: Configuration dictionary (only used if generator is a string)

        Returns:
            Self for method chaining

        Example:
            pipeline = EmbeddingPipeline()
            pipeline.add_generator("chronos", name="chronos_t5", config={"model_type": "t5"})
        """
        if isinstance(generator, str):
            if config is None:
                config = {}
            generator = EmbeddingGeneratorRegistry.create(generator, config)
            if name is None:
                name = generator.__class__.__name__.lower().replace("embeddinggenerator", "")

        if name is None:
            name = generator.__class__.__name__.lower().replace("embeddinggenerator", "")

        self.generators.append(generator)
        self.generator_names.append(name)

        return self

    def generate(self, features_df: pl.LazyFrame, context_length: int) -> pl.LazyFrame:
        """
        Generate embeddings using all generators in the pipeline.

        Args:
            features_df: LazyFrame with time series features (must have 'timestamp' column)
            context_length: Number of samples to use as context for each embedding

        Returns:
            LazyFrame with embeddings from all generators, indexed by timestamp
        """
        if not self.generators:
            raise ValueError("No generators in pipeline")

        if len(self.generators) == 1:
            return self.generators[0].generate_embeddings_batch(features_df, context_length)

        results = []
        for generator, name in zip(self.generators, self.generator_names):
            logger.info(f"Generating embeddings with {name}")
            embeddings = generator.generate_embeddings_batch(features_df, context_length)

            rename_mapping = {
                col: f"{name}_{col}" if col != "timestamp" else col
                for col in embeddings.schema.keys()
            }
            embeddings = embeddings.rename(rename_mapping)
            results.append(embeddings)

        result = results[0]
        for res in results[1:]:
            result = result.join(res, on="timestamp", how="inner")

        return result

    def get_all_generator_names(self) -> list[str]:
        """Return names of all generators in the pipeline."""
        return self.generator_names.copy()
