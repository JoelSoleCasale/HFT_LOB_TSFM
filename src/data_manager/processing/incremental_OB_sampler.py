from datetime import date, timedelta
from pathlib import Path
from sortedcontainers import SortedDict
from typing import Literal
import itertools
import numpy as np
import polars as pl
from tqdm import tqdm
from pandas import date_range
from loguru import logger
import warnings
from utils import iter_slices


class IncrementalOBSampler:
    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root) if isinstance(cache_root, str) else cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def precompute_full_snapshots(
        self,
        exchange: str | list[str],
        symbol: str | list[str],
        start_date: date,
        end_date: date,
        levels: int = 5,
        force_regenerate: bool = False,
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
        """
        exchanges = [exchange] if isinstance(exchange, str) else exchange
        symbols = [symbol] if isinstance(symbol, str) else symbol

        dates = list(date_range(start_date, end_date))

        for ex, sym, d in itertools.product(exchanges, symbols, dates):
            logger.info(f"Processing data for {ex}/{sym} on {d.strftime('%Y-%m-%d')}")
            try:
                self._get_full_event_sampled(
                    exchange=ex,
                    symbol=sym,
                    date=d,
                    levels=levels,
                    force_regenerate=force_regenerate,
                )
            except Exception as e:
                logger.error(
                    f"Failed to process data for {ex}/{sym} on {d.strftime('%Y-%m-%d')}: {e}"
                )

    def _get_data_path(self, exchange: str, symbol: str, date: date) -> Path:
        """Get path to raw orderbook data"""
        return (
            self.cache_root
            / "orderbook"
            / exchange
            / symbol
            / f"{date.strftime('%Y-%m-%d')}.parquet"
        )

    def _get_cache_path(self, exchange: str, symbol: str, date: date, levels: int) -> Path:
        """Get path to cached full event-sampled data"""
        return (
            self.cache_root
            / "orderbook_snapshots"
            / exchange
            / symbol
            / f"{date.strftime('%Y-%m-%d')}_L{levels}.npz"
        )

    def _get_ob_dtype(self, levels: int) -> list:
        """Generate dtype for structured numpy array."""
        dtype = [("timestamp", "i8")]
        for i in range(1, levels + 1):
            dtype.extend([(f"ask{i}_price", "f8"), (f"ask{i}_qty", "f8")])
        for i in range(1, levels + 1):
            dtype.extend([(f"bid{i}_price", "f8"), (f"bid{i}_qty", "f8")])
        return dtype

    def _initialize_orderbook(
        self,
        exchange: str,
        symbol: str,
        date: date,
        ob_init_prev_day_rows: int,
    ) -> dict:
        """Initializes the orderbook from the last `ob_init_prev_day_rows` of the previous day's data if available."""
        orderbook = {"bid": SortedDict(lambda k: -k), "ask": SortedDict()}

        prev_day_path = self._get_data_path(exchange, symbol, date - timedelta(days=1))
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
                    orderbook[side].pop(price, None)
                else:
                    orderbook[side][price] = quantity

        return orderbook

    def _generate_full_event_sampled(
        self,
        exchange: str,
        symbol: str,
        date: date,
        levels: int = 5,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
        batch_size: int = 1_000_000,
        ob_init_prev_day_rows: int = 100_000,
        verbose: bool = True,
    ) -> np.ndarray:
        """Generate full event-sampled orderbook data (expensive operation)"""
        logger.info(
            f"Generating full event-sampled data for {exchange}/{symbol}/{date} with {levels} levels"
        )

        df_path = self._get_data_path(exchange, symbol, date)
        if not df_path.exists():
            raise FileNotFoundError(f"Orderbook data not found at {df_path}")

        df = pl.scan_parquet(str(df_path)).select([reference_ts, "price", "quantity", "side"])
        df = df.with_columns(
            [pl.col("price").cast(pl.Float64), pl.col("quantity").cast(pl.Float64)]
        )

        # Initialize orderbook from previous day if available
        ob = self._initialize_orderbook(exchange, symbol, date, ob_init_prev_day_rows)

        # Define dtype for structured array
        dtype = self._get_ob_dtype(levels)

        capacity = 1_000_000
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = np.full(capacity, np.nan, dtype=dtype)
        row_num = 0
        prev_ts = 0

        total_rows = df.select(pl.len()).collect().item()

        with tqdm(total=total_rows, unit="item", unit_scale=True, disable=not verbose) as pbar:
            for batch_df in iter_slices(
                df.select([reference_ts, "price", "quantity", "side"]),
                n_rows=batch_size,
            ):
                for ts, price, quantity, side in batch_df.iter_rows():
                    if quantity == 0:
                        ob[side].pop(price, None)
                    else:
                        ob[side][price] = quantity

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
                    for j, (ask_price, ask_qty) in enumerate(ob["ask"].items()[:levels]):
                        result[row_num][f"ask{j+1}_price"] = ask_price
                        result[row_num][f"ask{j+1}_qty"] = ask_qty

                    # Add bid levels
                    for j, (bid_price, bid_qty) in enumerate(ob["bid"].items()[:levels]):
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
        return final_result

    def _get_full_event_sampled(
        self,
        exchange: str,
        symbol: str,
        date: date,
        levels: int = 5,
        force_regenerate: bool = False,
    ) -> np.ndarray:
        """
        Get full event-sampled data from cache or generate it.
        It will try to use a higher-level cache if available.
        After generating, it will prune lower-level caches.
        """
        cache_dir = self.cache_root / "orderbook_snapshots" / exchange / symbol
        cache_dir.mkdir(parents=True, exist_ok=True)

        requested_cache_path = self._get_cache_path(exchange, symbol, date, levels)

        if not force_regenerate:
            if requested_cache_path.exists():
                logger.info(f"Loading cached data from {requested_cache_path}")
                return np.load(requested_cache_path)["data"]

            # Check for higher-level caches
            existing_caches = list(cache_dir.glob(f"{date.strftime('%Y-%m-%d')}_L*.npz"))
            higher_caches = []
            for path in existing_caches:
                try:
                    level_str = path.stem.split("_L")[-1]
                    existing_level = int(level_str)
                    if existing_level > levels:
                        higher_caches.append((existing_level, path))
                except (IndexError, ValueError):
                    continue

            if higher_caches:
                # Use the lowest of the higher caches
                highest_level, highest_cache_path = min(higher_caches, key=lambda x: x[0])
                logger.info(
                    f"Found higher-level cache at {highest_cache_path} (L{highest_level}), using it to generate L{levels} data."
                )

                full_data = np.load(highest_cache_path)["data"]

                # Define dtype for the requested level
                dtype_req = self._get_ob_dtype(levels)

                # Create a new array with the requested dtype and copy data
                sliced_data = np.empty(full_data.shape, dtype=dtype_req)
                for name in sliced_data.dtype.names:
                    sliced_data[name] = full_data[name]

                return sliced_data

        # Generate data from scratch
        data = self._generate_full_event_sampled(exchange, symbol, date, levels)
        logger.info(f"Saving newly generated data to {requested_cache_path}")
        np.savez_compressed(requested_cache_path, data=data)

        # Prune lower-level caches
        existing_caches = list(cache_dir.glob(f"{date.strftime('%Y-%m-%d')}_L*.npz"))
        for path in existing_caches:
            if path == requested_cache_path:
                continue
            try:
                level_str = path.stem.split("_L")[-1]
                existing_level = int(level_str)
                if existing_level < levels:
                    logger.info(f"Pruning lower-level cache: {path}")
                    path.unlink()
            except (IndexError, ValueError):
                continue

        return data

    def sample_by_events(
        self,
        exchange: str,
        symbol: str,
        date: date,
        sample_rate: int,
        levels: int = 5,
        force_regenerate: bool = False,
    ) -> np.ndarray:
        """
        Sample orderbook by events (keep every nth event)

        Args:
            exchange: Exchange name
            symbol: Symbol name
            date: Date to sample
            sample_rate: Keep one row every sample_rate rows
            levels: Number of orderbook levels
            force_regenerate: Force regeneration of cached data
        """

        full_data = self._get_full_event_sampled(exchange, symbol, date, levels, force_regenerate)

        if sample_rate <= 1:
            return full_data

        sampled_data = full_data[::sample_rate]
        logger.info(f"Sampled {len(sampled_data)} rows from {len(full_data)} original rows")
        return sampled_data

    def sample_by_time(
        self,
        exchange: str,
        symbol: str,
        date: date,
        time_delta_ns: int,
        levels: int = 5,
        force_regenerate: bool = False,
    ) -> np.ndarray:
        """
        Sample orderbook by time interval

        Args:
            exchange: Exchange name
            symbol: Symbol name
            date: Date to sample
            time_delta_ns: Time interval in nanoseconds
            levels: Number of orderbook levels
            force_regenerate: Force regeneration of cached data
        """

        full_data = self._get_full_event_sampled(exchange, symbol, date, levels, force_regenerate)

        if time_delta_ns <= 0 or len(full_data) == 0:
            return full_data

        # Convert to Polars DataFrame
        df = pl.from_numpy(full_data)

        # Calculate adjusted timestamps
        df = df.with_columns(
            (pl.col("timestamp") - pl.col("timestamp") % time_delta_ns).alias("timestamp")
        )

        # Group by adjusted timestamp and take last row from each group
        result_df = df.group_by("timestamp").last().sort("timestamp")

        # Create result array with the new dtype
        result_array = np.empty(len(result_df), dtype=full_data.dtype)

        # Fill the result array
        for col_name in full_data.dtype.names:
            result_array[col_name] = result_df[col_name].to_numpy()

        logger.info(f"Sampled {len(result_array)} rows from {len(full_data)} original rows")
        return result_array

    def clear_cache(self, exchange: str = None, symbol: str = None, date: date = None):
        """Clear cached data for specific parameters"""
        cache_dir = self.cache_root / "orderbook_snapshots"

        if exchange is None:
            # Clear all cache
            for path in cache_dir.rglob("*.npz"):
                path.unlink()
            # Remove empty directories
            for path in sorted(cache_dir.rglob("*"), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            logger.info("Cleared all cached data")
        elif symbol is None:
            # Clear exchange cache
            exchange_dir = cache_dir / exchange
            if exchange_dir.exists():
                for path in exchange_dir.rglob("*.npz"):
                    path.unlink()
                # Remove empty directories
                for path in sorted(exchange_dir.rglob("*"), reverse=True):
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                # Remove exchange directory if empty
                if not any(exchange_dir.iterdir()):
                    exchange_dir.rmdir()
                logger.info(f"Cleared cache for exchange: {exchange}")
        elif date is None:
            # Clear symbol cache
            symbol_dir = cache_dir / exchange / symbol
            if symbol_dir.exists():
                for path in symbol_dir.rglob("*.npz"):
                    path.unlink()
                # Remove symbol directory if empty
                if not any(symbol_dir.iterdir()):
                    symbol_dir.rmdir()
                logger.info(f"Cleared cache for {exchange}/{symbol}")
        else:
            # Clear specific date cache
            pattern = f"{date.strftime('%Y-%m-%d')}_*.npz"
            date_dir = cache_dir / exchange / symbol
            if date_dir.exists():
                for path in date_dir.glob(pattern):
                    path.unlink()
                logger.info(f"Cleared cache for {exchange}/{symbol}/{date}")
