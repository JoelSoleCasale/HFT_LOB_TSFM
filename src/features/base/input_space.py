"""Core data container for all raw inputs"""

from dataclasses import dataclass
import polars as pl

from core.orderbook import OrderBook


@dataclass
class InputSpace:
    """Unified container for all raw input data"""

    orderbook_snapshots: OrderBook | None = None
    trades: pl.LazyFrame | None = None
    liquidations: pl.LazyFrame | None = None
    funding_rates: pl.LazyFrame | None = None
    mark_prices: pl.LazyFrame | None = None
    open_interest: pl.LazyFrame | None = None
    sentiment: pl.LazyFrame | None = None

    metadata: dict | None = None  # timestamps, symbols, etc.

    def validate(self) -> bool:
        """Ensure required data is present"""
        pass
