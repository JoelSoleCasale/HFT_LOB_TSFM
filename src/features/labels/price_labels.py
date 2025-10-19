"""Price-based label extractors"""

from features.base.label_extractor import BaseLabelExtractor
import polars as pl
from features.base.input_space import InputSpace


class MidPriceReturnLabel(BaseLabelExtractor):
    """Future mid-price returns as labels"""

    def __init__(self, config=None):
        super().__init__(config)
        self.horizons = self.config.get("horizons", [5, 10, 30])  # samples
        self.dependencies = ["orderbook_snapshots"]
        self.label_names = [f"return_{h}" for h in self.horizons]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        mid_prices = input_space.orderbook_snapshots.get_mid_prices()

        def future_return_exprs(horizons):
            for h in horizons:
                yield (
                    (pl.col("mid_price").shift(-h) - pl.col("mid_price")) / pl.col("mid_price")
                ).alias(f"return_{h}")

        return mid_prices.with_columns(list(future_return_exprs(self.horizons))).select(
            ["timestamp"] + [f"return_{h}" for h in self.horizons]
        )
