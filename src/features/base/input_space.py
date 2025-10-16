"""Core data container for all raw inputs"""

from dataclasses import dataclass
from typing import Optional
import polars as pl


@dataclass
class InputSpace:
    """Unified container for all raw input data"""

    orderbook_snapshots: Optional[pl.LazyFrame] = None
    trades: Optional[pl.LazyFrame] = None
    liquidations: Optional[pl.LazyFrame] = None
    funding_rates: Optional[pl.LazyFrame] = None
    mark_prices: Optional[pl.LazyFrame] = None
    open_interest: Optional[pl.LazyFrame] = None
    sentiment: Optional[pl.LazyFrame] = None
    # Add more as needed
    metadata: Optional[dict] = None  # timestamps, symbols, etc.

    def validate(self) -> bool:
        """Ensure required data is present"""
        pass
