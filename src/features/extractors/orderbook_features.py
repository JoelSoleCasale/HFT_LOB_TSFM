"""Orderbook-specific feature extractors"""

from features.base.feature_extractor import BaseFeatureExtractor
from features.registry import FeatureExtractorRegistry
import polars as pl
from features.base.input_space import InputSpace


@FeatureExtractorRegistry.register("mid_price")
class MidPriceFeatures(BaseFeatureExtractor):
    """Extract mid-price features from orderbook"""

    def __init__(self, config=None):
        super().__init__(config)
        self.dependencies = ["orderbook_snapshots"]
        self.feature_names = ["mid_price"]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        mid_price = input_space.orderbook_snapshots.get_mid_prices()
        return mid_price


@FeatureExtractorRegistry.register("spread")
class SpreadFeatures(BaseFeatureExtractor):
    """Extract spread features"""

    def __init__(self, config=None):
        super().__init__(config)
        self.dependencies = ["orderbook_snapshots"]
        self.feature_names = ["spread"]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        spread = input_space.orderbook_snapshots.get_spreads()
        return spread


@FeatureExtractorRegistry.register("orderbook_imbalance")
class OrderbookImbalanceFeatures(BaseFeatureExtractor):
    """Extract orderbook imbalance features"""

    def __init__(self, config=None):
        super().__init__(config)
        self.levels = self.config.get("levels", [1, 2, 5])
        self.dependencies = ["orderbook_snapshots"]
        self.feature_names = [f"imbalance_L{level}" for level in self.levels]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        df = input_space.orderbook_snapshots.df

        # Calculate imbalance for each level
        imbalance_exprs = [pl.col("timestamp")]
        for level in self.levels:
            # Sum bid quantities from level 1 to level
            bid_cols = [pl.col(f"bid{i}_qty") for i in range(1, level + 1)]
            bid_volume = pl.sum_horizontal(bid_cols)

            # Sum ask quantities from level 1 to level
            ask_cols = [pl.col(f"ask{i}_qty") for i in range(1, level + 1)]
            ask_volume = pl.sum_horizontal(ask_cols)

            # Calculate imbalance
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            imbalance_exprs.append(imbalance.alias(f"imbalance_L{level}"))

        return df.select(imbalance_exprs)
