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

        imbalance_exprs = [pl.col("timestamp")]
        for level in self.levels:
            bid_cols = [pl.col(f"bid{i}_qty") for i in range(1, level + 1)]
            bid_volume = pl.sum_horizontal(bid_cols)

            ask_cols = [pl.col(f"ask{i}_qty") for i in range(1, level + 1)]
            ask_volume = pl.sum_horizontal(ask_cols)

            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            imbalance_exprs.append(imbalance.alias(f"imbalance_L{level}"))

        return df.select(imbalance_exprs)


@FeatureExtractorRegistry.register("advanced_orderbook")
class AdvancedOrderbookFeatures(BaseFeatureExtractor):
    """Extract advanced orderbook features including volumes, spreads, VWAPs, and log returns"""

    def __init__(self, config=None):
        super().__init__(config)
        self.levels = self.config.get("levels", 5)
        self.dependencies = ["orderbook_snapshots"]

        self.feature_names = [
            "buy_vol",
            "sell_vol",
            "vol",
            *[f"bid_s{i}" for i in range(1, self.levels + 1)],
            *[f"ask_s{i}" for i in range(1, self.levels + 1)],
            "wap1",
            "wap2",
            "wap_balance",
            "buy_sp",
            "sell_sp",
            "vol_imbalance",
            "price_sp",
            "buy_vwap",
            "sell_vwap",
            "log_return_bid",
            "log_return_ask",
            "log_return_wap1",
            "log_return_wap2",
        ]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        self.validate_input(input_space)
        df: pl.LazyFrame = input_space.orderbook_snapshots.df

        bid_qty_cols = [pl.col(f"bid{i}_qty") for i in range(1, self.levels + 1)]
        ask_qty_cols = [pl.col(f"ask{i}_qty") for i in range(1, self.levels + 1)]

        df = df.with_columns(
            [
                pl.sum_horizontal(bid_qty_cols).alias("buy_vol"),
                pl.sum_horizontal(ask_qty_cols).alias("sell_vol"),
            ]
        )

        df = df.with_columns(
            [
                (pl.col("buy_vol") + pl.col("sell_vol")).alias("vol"),
                *[pl.col(f"bid{i}_qty").alias(f"bid_s{i}") for i in range(1, self.levels + 1)],
                *[pl.col(f"ask{i}_qty").alias(f"ask_s{i}") for i in range(1, self.levels + 1)],
                (
                    (
                        pl.col("bid1_price") * pl.col("bid1_qty")
                        + pl.col("ask1_price") * pl.col("ask1_qty")
                    )
                    / (pl.col("bid1_qty") + pl.col("ask1_qty"))
                ).alias("wap1"),
                (
                    (
                        pl.col("bid2_price") * pl.col("bid2_qty")
                        + pl.col("ask2_price") * pl.col("ask2_qty")
                    )
                    / (pl.col("bid2_qty") + pl.col("ask2_qty"))
                ).alias("wap2"),
                (pl.col("bid1_price") - pl.col("bid5_price")).alias("buy_sp"),
                (pl.col("ask5_price") - pl.col("ask1_price")).alias("sell_sp"),
                (
                    (pl.col("buy_vol") - pl.col("sell_vol"))
                    / (pl.col("buy_vol") + pl.col("sell_vol"))
                ).alias("vol_imbalance"),
                (pl.col("ask1_price") - pl.col("bid1_price")).alias("price_sp"),
                (
                    pl.sum_horizontal(
                        [
                            pl.col(f"bid{i}_price") * pl.col(f"bid{i}_qty")
                            for i in range(1, self.levels + 1)
                        ]
                    )
                    / pl.col("buy_vol")
                ).alias("buy_vwap"),
                (
                    pl.sum_horizontal(
                        [
                            pl.col(f"ask{i}_price") * pl.col(f"ask{i}_qty")
                            for i in range(1, self.levels + 1)
                        ]
                    )
                    / pl.col("sell_vol")
                ).alias("sell_vwap"),
            ]
        )

        df = df.with_columns(
            [
                (pl.col("wap1") - pl.col("wap2")).alias("wap_balance"),
            ]
        )

        df = df.with_columns(
            [
                (pl.col("bid1_price") / pl.col("bid1_price").shift(1))
                .log()
                .fill_null(0)
                .alias("log_return_bid"),
                (pl.col("ask1_price") / pl.col("ask1_price").shift(1))
                .log()
                .fill_null(0)
                .alias("log_return_ask"),
                (pl.col("wap1") / pl.col("wap1").shift(1))
                .log()
                .fill_null(0)
                .alias("log_return_wap1"),
                (pl.col("wap2") / pl.col("wap2").shift(1))
                .log()
                .fill_null(0)
                .alias("log_return_wap2"),
            ]
        )

        final_cols = ["timestamp"] + self.feature_names
        return df.select(final_cols)
