"""Trade flow feature extractors"""

from features.base.feature_extractor import BaseFeatureExtractor
from features.registry import FeatureExtractorRegistry
import polars as pl
from features.base.input_space import InputSpace


@FeatureExtractorRegistry.register("trade_flow")
class TradeFlowFeatures(BaseFeatureExtractor):
    """Extract trade flow statistics"""

    def __init__(self, config=None):
        super().__init__(config)
        self.windows = self.config.get("windows", [10, 30, 60])  # seconds
        self.dependencies = ["trades"]
        self.feature_names = [
            f"{stat}_{window}s"
            for stat in ["trade_count", "buy_volume", "sell_volume", "vwap"]
            for window in self.windows
        ]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        features = []
        for window in self.windows:
            trade_count = input_space.trades.group_by_dynamic("timestamp", every=f"{window}s").agg(
                pl.count().alias(f"trade_count_{window}s")
            )
            buy_volume = (
                input_space.trades.filter(pl.col("side") == "buy")
                .group_by_dynamic("timestamp", every=f"{window}s")
                .agg(pl.col("quantity").sum().alias(f"buy_volume_{window}s"))
            )
            # ... similar for sell_volume and vwap
            features.extend([trade_count, buy_volume])
        return pl.concat(features, how="horizontal")

    def get_feature_names(self):
        return self.feature_names
