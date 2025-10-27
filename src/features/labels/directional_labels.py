"""Directional/classification label extractors"""

from features.base.label_extractor import BaseLabelExtractor
import polars as pl
from features.base.input_space import InputSpace


class DirectionalLabel(BaseLabelExtractor):
    """Directional labels based on return thresholds: -1 (down), 0 (flat), 1 (up)"""

    def __init__(self, config=None):
        super().__init__(config)
        self.horizon = self.config.get("horizon", 10)  # samples
        self.threshold = self.config.get("threshold", 0.0001)  # 0.01% default threshold
        self.dependencies = ["orderbook_snapshots"]
        self.label_names = [f"direction_{self.horizon}"]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        mid_prices = input_space.orderbook_snapshots.get_mid_prices()

        ret = (pl.col("mid_price").shift(-self.horizon) - pl.col("mid_price")) / pl.col(
            "mid_price"
        )
        direction_expr = (
            pl.when(ret > self.threshold).then(1).when(ret < -self.threshold).then(-1).otherwise(0)
        ).alias(f"direction_{self.horizon}")

        return mid_prices.with_columns(direction_expr).select(
            ["timestamp", f"direction_{self.horizon}"]
        )


class SmoothedDirectionalLabel(BaseLabelExtractor):
    """Smoothed directional labels based on moving average comparison: -1 (down), 0 (flat), 1 (up)

    Compares the forward moving average (next k samples) with the backward moving average
    (last k samples) to determine direction:

    pc(t) = (m_+(t) - m_-(t)) / m_-(t)

    where:
    - m_+(t) = (1/k) * sum(m(t+i) for i in 1..k)  # forward MA
    - m_-(t) = (1/k) * sum(m(t-i) for i in 0..k)  # backward MA
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.window = self.config.get("window", 10)  # k samples for averaging
        self.threshold = self.config.get("threshold", 0.0001)  # 0.01% default threshold
        self.dependencies = ["orderbook_snapshots"]
        self.label_names = [f"smoothed_direction_{self.window}"]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        mid_prices = input_space.orderbook_snapshots.get_mid_prices()

        # Backward moving average: average of current and past k samples
        # m_-(t) = (1/k) * sum(m(t-i) for i in 0..k)
        backward_ma = pl.col("mid_price").rolling_mean(
            window_size=self.window + 1, min_periods=self.window + 1
        )

        # Forward moving average: average of next k samples
        # m_+(t) = (1/k) * sum(m(t+i) for i in 1..k)
        # We shift backwards by -(window-1) to -(0) and average
        forward_ma = (
            pl.col("mid_price")
            .shift(-self.window)
            .rolling_mean(window_size=self.window, min_periods=self.window)
            .shift(self.window - 1)
        )

        # Percentage change: pc(t) = (m_+(t) - m_-(t)) / m_-(t)
        pc = ((forward_ma - backward_ma) / backward_ma).alias("pc")

        # Assign directional labels based on threshold
        direction_expr = (
            pl.when(pl.col("pc") > self.threshold)
            .then(1)
            .when(pl.col("pc") < -self.threshold)
            .then(-1)
            .otherwise(0)
        ).alias(f"smoothed_direction_{self.window}")

        return (
            mid_prices.with_columns(
                [backward_ma.alias("m_backward"), forward_ma.alias("m_forward"), pc]
            )
            .with_columns(direction_expr)
            .select(["timestamp", f"smoothed_direction_{self.window}"])
        )
