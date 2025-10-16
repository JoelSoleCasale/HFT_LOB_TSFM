"""Directional/classification label extractors"""

from features.base.label_extractor import BaseLabelExtractor
import polars as pl
from features.base.input_space import InputSpace
from features.labels.price_labels import MidPriceReturnLabel


class DirectionalLabel(BaseLabelExtractor):
    """Classification labels for price direction"""

    def __init__(self, config=None):
        super().__init__(config)
        self.horizon = self.config.get("horizon", 10)  # seconds
        self.threshold = self.config.get("threshold", 0.0001)  # 1 bps
        self.dependencies = ["orderbook_snapshots"]
        self.label_names = ["direction"]
        self.lookahead_window = self.horizon

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        mid_price_return_extractor = MidPriceReturnLabel(config={"horizons": [self.horizon]})
        returns = mid_price_return_extractor.extract(input_space)
        direction = (
            pl.when(returns.select(f"return_{self.horizon}s") > self.threshold)
            .then(1)
            .when(returns.select(f"return_{self.horizon}s") < -self.threshold)
            .then(-1)
            .otherwise(0)
            .alias("direction")
        )
        return pl.concat([returns.select("timestamp"), direction], how="horizontal")

    def get_label_names(self):
        return self.label_names
