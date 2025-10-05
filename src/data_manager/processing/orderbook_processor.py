"""Core orderbook processing logic."""

from datetime import date, timedelta
from pathlib import Path
import polars as pl
import numpy as np
import warnings
from tqdm import tqdm
from loguru import logger

from utils import iter_slices
from core.orderbook import OrderBook, OrderBookSnapshot, OrderBookData
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
                pl.scan_parquet(str(prev_day_path))
                .select(["price", "quantity", "side"])
                .with_columns(
                    [
                        pl.col("price").cast(pl.Float64),
                        pl.col("quantity").cast(pl.Float64),
                    ]
                )
                .tail(ob_init_prev_day_rows)
                .collect()
            )

            for price, quantity, side in df_prev.iter_rows():
                if quantity == 0:
                    ob[side].pop(price, None)
                else:
                    ob[side][price] = quantity

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

        df = pl.scan_parquet(str(raw_data_path)).select(
            [request.reference_ts, "price", "quantity", "side"]
        )
        df = df.with_columns(
            [pl.col("price").cast(pl.Float64), pl.col("quantity").cast(pl.Float64)]
        )

        # Initialize orderbook from previous day if available
        ob = self.initialize_orderbook(
            request.exchange, request.symbol, request.date, request.ob_init_prev_day_rows
        )

        formats_map = {
            pl.Int64: np.int64,
            pl.Float64: np.float64,
        }
        dtype = [
            (name, formats_map[pl_type])
            for name, pl_type in OrderBookData.get_orderbook_schema(request.levels)
        ]

        capacity = 1_000_000
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = np.full(capacity, np.nan, dtype=dtype)
        row_num = 0
        prev_ts = 0

        total_rows = df.select(pl.len()).collect().item()

        with tqdm(total=total_rows, unit="item", unit_scale=True) as pbar:
            for batch_df in iter_slices(
                df.select([request.reference_ts, "price", "quantity", "side"]),
                n_rows=request.batch_size,
            ):
                for ts, price, quantity, side in batch_df.iter_rows():
                    ob.update(side, price, quantity)

                    if ts == prev_ts:
                        continue

                    assert ts >= prev_ts, "Timestamps should be non-decreasing"

                    if row_num >= capacity:
                        new_capacity = capacity * 2
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            new_result = np.full(new_capacity, np.nan, dtype=dtype)
                        new_result[:capacity] = result
                        result = new_result
                        capacity = new_capacity

                    result[row_num]["timestamp"] = ts

                    # Add ask levels
                    for j, (ask_price, ask_qty) in enumerate(ob.ask.items()[: request.levels]):
                        result[row_num][f"ask{j+1}_price"] = ask_price
                        result[row_num][f"ask{j+1}_qty"] = ask_qty

                    # Add bid levels
                    for j, (bid_price, bid_qty) in enumerate(ob.bid.items()[: request.levels]):
                        result[row_num][f"bid{j+1}_price"] = bid_price
                        result[row_num][f"bid{j+1}_qty"] = bid_qty

                    prev_ts = ts

                    # Skip if identical to previous row (except timestamp)
                    if (
                        row_num > 0
                        and tuple(result[row_num])[1:] == tuple(result[row_num - 1])[1:]
                    ):
                        continue

                    row_num += 1

                pbar.update(batch_df.height)

        final_result = result[:row_num]
        logger.info(f"Generated {len(final_result)} orderbook snapshots")
        return OrderBook(pl.DataFrame(final_result))
