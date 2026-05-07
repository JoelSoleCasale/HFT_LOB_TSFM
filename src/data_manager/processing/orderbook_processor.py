"""Core orderbook processing logic."""

from datetime import date, timedelta
from pathlib import Path
import polars as pl
from tqdm import tqdm
from loguru import logger

from utils import iter_slices
from core.orderbook import OrderBook, OrderBookSnapshot
from data_manager.downloader.data_downloader_request import RawDataRequest
from data_manager.processing.orderbook_request import OrderBookSnapshotRequest


class OrderBookProcessor:
    """Handles the core orderbook processing logic."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root

    def initialize_orderbook(
        self,
        exchange: str,
        symbol: str,
        date: date,
        ob_init_prev_day_rows: int,
    ) -> OrderBookSnapshot:
        """Initialize orderbook from previous day's data if available."""
        ob = OrderBookSnapshot()

        prev_day_request = RawDataRequest("orderbook", symbol, exchange, date - timedelta(days=1))
        prev_day_path = self.cache_root / prev_day_request.get_path()

        if prev_day_path.exists():
            logger.info(f"Initializing orderbook from previous day: {prev_day_path}")
            df_prev = (
                pl.scan_parquet(prev_day_path)
                .select(["price", "quantity", "side"])
                .tail(ob_init_prev_day_rows)
                .collect()
            )

            for price, quantity, side in df_prev.iter_rows():
                ob.update(side, price, quantity)

        return ob

    def generate_full_event_sampled(
        self,
        request: OrderBookSnapshotRequest,
    ) -> OrderBook:
        """Generate full event-sampled orderbook data."""
        logger.info(
            f"Generating full event-sampled data for {request.exchange}/{request.symbol}/{request.date} "
            f"with {request.levels} levels"
        )

        raw_data_path = (
            self.cache_root
            / RawDataRequest(
                "orderbook", request.symbol, request.exchange, request.date
            ).get_path()
        )

        if not raw_data_path.exists():
            raise FileNotFoundError(f"Orderbook data not found at {raw_data_path}")

        df = pl.scan_parquet(raw_data_path).select(
            [request.reference_ts, "price", "quantity", "side"]
        )

        # Initialize orderbook from previous day if available
        ob = self.initialize_orderbook(
            request.exchange, request.symbol, request.date, request.ob_init_prev_day_rows
        )

        result = []
        _head = df.select(request.reference_ts).head(1).collect()
        if _head.is_empty():
            raise ValueError(
                f"Orderbook data is empty for {request.exchange}/{request.symbol}/{request.date}"
            )
        prev_ts = _head.item()

        total_rows = df.select(pl.len()).collect().item()

        with tqdm(total=total_rows, unit="item", unit_scale=True) as pbar:
            for batch_df in iter_slices(
                df.select([request.reference_ts, "price", "quantity", "side"]),
                n_rows=request.batch_size,
            ):
                if isinstance(batch_df, pl.LazyFrame):
                    batch_df = batch_df.collect()

                for ts, price, quantity, side in batch_df.iter_rows():

                    if ts == prev_ts:
                        ob.update(side, price, quantity)
                        continue

                    assert ts >= prev_ts, "Timestamps should be non-decreasing"

                    # All events for previous timestamp processed, record snapshot
                    row_data = ob.get_top_levels(request.levels)
                    row_data["timestamp"] = prev_ts
                    result.append(row_data)

                    # Now process the current event
                    ob.update(side, price, quantity)

                    prev_ts = ts

                pbar.update(batch_df.height)

        # Record final snapshot
        row_data = ob.get_top_levels(request.levels)
        row_data["timestamp"] = prev_ts
        result.append(row_data)

        logger.info(f"Generated {len(result)} orderbook snapshots")

        df = pl.DataFrame(result)

        # remove duplicated rows (rows identical to previous row)
        float_cols = [c for c in df.columns if c != "timestamp"]

        mask = pl.any_horizontal([pl.col(col) != pl.col(col).shift(1) for col in float_cols]) | (
            pl.arange(0, pl.len()) == 0
        )

        df = df.filter(mask)

        return OrderBook(df)
