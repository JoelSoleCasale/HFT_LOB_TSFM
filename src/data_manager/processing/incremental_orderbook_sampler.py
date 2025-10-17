from datetime import date
from pathlib import Path
from typing import Literal
import itertools
from loguru import logger
from utils import date_range
from core.orderbook import OrderBook
from data_manager.processing.orderbook_request import OrderBookSnapshotRequest
from data_manager.processing.orderbook_cache_manager import OrderBookCacheManager


class IncrementalOrderBookSampler:
    """Simplified orderbook sampler with separated concerns."""

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root) if isinstance(cache_root, str) else cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_manager = OrderBookCacheManager(self.cache_root)

    def precompute_full_snapshots(
        self,
        exchange: str | list[str],
        symbol: str | list[str],
        start_date: date,
        end_date: date,
        levels: int = 5,
        force_regenerate: bool = False,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
        batch_size: int = 1_000_000,
        ob_init_prev_day_rows: int = 100_000,
    ):
        """
        Precompute full event-sampled orderbook snapshots for given parameters and cache them.

        It iterates through the Cartesian product of all provided lists.

        Args:
            exchange (str | list[str]): The exchange(s) to process.
            symbol (str | list[str]): The trading symbol(s) to process.
            start_date (date): The start date of the sampling period.
            end_date (date): The end date of the sampling period.
            levels (int, optional): Number of orderbook levels. Defaults to 5.
            force_regenerate (bool, optional): If true, force regeneration of cached data. Defaults to False.
            reference_ts (Literal["received_time", "event_time"]): Timestamp reference. Defaults to "received_time".
            batch_size (int): Batch size for processing. Defaults to 1_000_000.
            ob_init_prev_day_rows (int): Rows from previous day for initialization. Defaults to 100_000.
        """
        exchanges = [exchange] if isinstance(exchange, str) else exchange
        symbols = [symbol] if isinstance(symbol, str) else symbol

        dates = date_range(start_date, end_date)

        for ex, sym, d in itertools.product(exchanges, symbols, dates):
            logger.info(f"Processing data for {ex}/{sym} on {d.strftime('%Y-%m-%d')}")
            try:
                request = OrderBookSnapshotRequest(
                    exchange=ex,
                    symbol=sym,
                    date=d,
                    levels=levels,
                    reference_ts=reference_ts,
                    batch_size=batch_size,
                    ob_init_prev_day_rows=ob_init_prev_day_rows,
                )
                self.cache_manager.get_cached_orderbook(request, force_regenerate)
            except Exception as e:
                logger.error(
                    f"Failed to process data for {ex}/{sym} on {d.strftime('%Y-%m-%d')}: {e}"
                )

    def get_orderbook(
        self,
        exchange: str,
        symbol: str,
        date: date,
        levels: int = 5,
        force_regenerate: bool = False,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
        batch_size: int = 1_000_000,
        ob_init_prev_day_rows: int = 100_000,
    ) -> OrderBook:
        """
        Get orderbook data for a specific exchange, symbol, and date.

        Args:
            exchange (str): The exchange to process.
            symbol (str): The trading symbol to process.
            date (date): The date to process.
            levels (int, optional): Number of orderbook levels. Defaults to 5.
            force_regenerate (bool, optional): If true, force regeneration of cached data. Defaults to False.
            reference_ts (Literal["received_time", "event_time"]): Timestamp reference. Defaults to "received_time".
            batch_size (int): Batch size for processing. Defaults to 1_000_000.
            ob_init_prev_day_rows (int): Rows from previous day for initialization. Defaults to 100_000.

        Returns:
            OrderBook: The processed orderbook data.
        """
        request = OrderBookSnapshotRequest(
            exchange=exchange,
            symbol=symbol,
            date=date,
            levels=levels,
            reference_ts=reference_ts,
            batch_size=batch_size,
            ob_init_prev_day_rows=ob_init_prev_day_rows,
        )
        return self.cache_manager.get_cached_orderbook(request, force_regenerate)
