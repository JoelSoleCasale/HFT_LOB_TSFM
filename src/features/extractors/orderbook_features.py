"""Orderbook-specific feature extractors"""

from features.base.feature_extractor import BaseFeatureExtractor
from features.registry import FeatureExtractorRegistry
import polars as pl
from features.base.input_space import InputSpace


@FeatureExtractorRegistry.register("orderbook_imbalance")
class OrderbookImbalanceFeatures(BaseFeatureExtractor):
    """Extract orderbook imbalance features"""

    def __init__(self, config=None):
        super().__init__(config)
        self.levels = self.config.get("levels", [1, 5, 10])
        self.dependencies = ["orderbook_snapshots"]
        self.feature_names = [f"imbalance_L{level}" for level in self.levels]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        features = []
        for level in self.levels:
            bid_volume = (
                input_space.orderbook_snapshots.filter(pl.col("side") == "bid")
                .head(level)
                .select(pl.col("quantity").sum())
            )
            ask_volume = (
                input_space.orderbook_snapshots.filter(pl.col("side") == "ask")
                .head(level)
                .select(pl.col("quantity").sum())
            )
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            features.append(imbalance.alias(f"imbalance_L{level}"))
        return pl.concat(features, how="horizontal")

    def get_feature_names(self):
        return self.feature_names


@FeatureExtractorRegistry.register("spread")
class SpreadFeatures(BaseFeatureExtractor):
    """Extract spread features"""

    def __init__(self, config=None):
        super().__init__(config)
        self.dependencies = ["orderbook_snapshots"]
        self.feature_names = ["spread"]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        spread = input_space.orderbook_snapshots.filter(pl.col("level") == 0).select(
            ["timestamp", (pl.col("ask_price") - pl.col("bid_price")).alias("spread")]
        )
        return spread

    def get_feature_names(self):
        return self.feature_names
