"""Price-based label extractors"""

from features.base.label_extractor import BaseLabelExtractor
import polars as pl
from features.base.input_space import InputSpace


class MidPriceReturnLabel(BaseLabelExtractor):
    """Future mid-price returns as labels"""

    def __init__(self, config=None):
        super().__init__(config)
        self.horizons = self.config.get("horizons", [5, 10, 30])  # seconds
        self.dependencies = ["orderbook_snapshots"]
        self.label_names = [f"return_{h}s" for h in self.horizons]
        self.lookahead_window = max(self.horizons)

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        mid_price = input_space.orderbook_snapshots.filter(pl.col("level") == 0).select(
            ["timestamp", ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid")]
        )
        labels = []
        for horizon in self.horizons:
            future_return = (
                mid_price.select("mid").shift(-horizon) / mid_price.select("mid") - 1
            ).alias(f"return_{horizon}s")
            labels.append(future_return)
        return pl.concat([mid_price.select("timestamp")] + labels, how="horizontal")

    def get_label_names(self):
        return self.label_names
