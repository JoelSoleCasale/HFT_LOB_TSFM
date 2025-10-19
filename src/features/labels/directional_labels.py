"""Directional/classification label extractors"""

from features.base.label_extractor import BaseLabelExtractor
import polars as pl
from features.base.input_space import InputSpace


class DirectionalLabel(BaseLabelExtractor):
    """Directional labels based on return thresholds: -1 (down), 0 (flat), 1 (up)"""

    def __init__(self, config=None):
        super().__init__(config)
        self.horizons = self.config.get("horizons", [5, 10, 30])  # samples
        self.threshold = self.config.get("threshold", 0.0001)  # 0.01% default threshold
        self.dependencies = ["orderbook_snapshots"]
        self.label_names = [f"direction_{h}" for h in self.horizons]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        mid_prices = input_space.orderbook_snapshots.get_mid_prices()

        def future_direction_exprs(horizons, threshold):
            for h in horizons:
                ret = (pl.col("mid_price").shift(-h) - pl.col("mid_price")) / pl.col("mid_price")
                yield (
                    pl.when(ret > threshold).then(1).when(ret < -threshold).then(-1).otherwise(0)
                ).alias(f"direction_{h}")

        return mid_prices.with_columns(
            list(future_direction_exprs(self.horizons, self.threshold))
        ).select(["timestamp"] + [f"direction_{h}" for h in self.horizons])
