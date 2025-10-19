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
        raise NotImplementedError("TradeFlowFeatures.extract is not yet implemented")
