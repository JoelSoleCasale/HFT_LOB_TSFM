"""Price-based label extractors"""

from features.base.label_extractor import BaseLabelExtractor
import polars as pl
from features.base.input_space import InputSpace


class MidPriceReturnLabel(BaseLabelExtractor):
    """Future mid-price returns as labels"""

    def __init__(self, config=None):
        super().__init__(config)
        self.horizon = self.config.get("horizon", 5)  # samples
        self.dependencies = ["orderbook_snapshots"]
        self.label_names = [f"return_{self.horizon}"]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        mid_prices = input_space.orderbook_snapshots.get_mid_prices()

        return mid_prices.with_columns(
            (
                (pl.col("mid_price").shift(-self.horizon) - pl.col("mid_price"))
                / pl.col("mid_price")
            ).alias(f"return_{self.horizon}")
        ).select(["timestamp", f"return_{self.horizon}"])
