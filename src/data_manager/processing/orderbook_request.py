"""Data request implementations for orderbook processing."""

from pathlib import Path
from datetime import date
from typing import Literal
from custom_types import DataRequest
from dataclasses import dataclass


@dataclass
class OrderBookSnapshotRequest(DataRequest):
    """Request for orderbook sampling operations."""

    exchange: str
    symbol: str
    date: date
    levels: int = 5
    reference_ts: Literal["received_time", "event_time"] = "received_time"
    batch_size: int = 1_000_000
    ob_init_prev_day_rows: int = 100_000

    def get_path(self) -> Path:
        """Return the cache path for this request."""
        return Path(
            f"orderbook_snapshots/{self.exchange}/{self.symbol}/{self.date.strftime('%Y-%m-%d')}_L{self.levels}.parquet"
        )
