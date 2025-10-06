import polars as pl
import numpy as np
from typing import Iterator
from pathlib import Path
from typing import Callable
from custom_types import DataRequest
from hftbacktest import (
    BUY_EVENT,
    SELL_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    event_dtype,
)
from hftbacktest.data import validate_event_order


def iter_slices(
    df: pl.DataFrame | pl.LazyFrame, n_rows: int = 10_000
) -> Iterator[pl.DataFrame | pl.LazyFrame]:
    """Iterate over DataFrame/LazyFrame in slices."""
    if isinstance(df, pl.DataFrame):
        for offset in range(0, df.height, n_rows):
            yield df.slice(offset, n_rows)
    else:
        row_count = df.select(pl.len()).collect().item()
        for offset in range(0, row_count, n_rows):
            yield df.slice(offset, n_rows).collect()


def cache_func(
    func: Callable[[DataRequest], pl.DataFrame | None],
    cache_root: Path,
    load_cached_data: bool = True,
) -> tuple[Callable[[DataRequest], pl.DataFrame | None], bool]:
    """Decorator to cache function output to a Parquet file.
    If the file exists, load from it instead of calling the function.
    Args:
        func (Callable[[DataRequest], pl.DataFrame]): Function to be decorated.
        cache_root (Path): Root directory for caching.
        load_cached_data (bool, optional): If True, load from cache if available.
    Returns:
        Callable[[DataRequest], pl.DataFrame]: Decorated function with caching.
        bool: True if data was loaded from cache, False if function was called.
    """

    def wrapper(request: DataRequest, *args, **kwargs) -> pl.DataFrame:
        cache_path = cache_root / request.get_path()
        if cache_path.exists():
            if not load_cached_data:
                return None, True
            return pl.read_parquet(cache_path), True
        else:
            result: pl.DataFrame | None = func(request, *args, **kwargs)
            if result is None:
                return None, False
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.write_parquet(cache_path)
            return result, False

    return wrapper


def get_hftbacktest_array(df: pl.DataFrame) -> np.ndarray:
    """
    Convert a Polars DataFrame representing an incremental orderbook event stream
    into a structured NumPy array compatible with hftbacktest.
    """
    arr = np.zeros(len(df), dtype=event_dtype)

    # Use vectorized operations to determine event types
    is_snapshot = (df["event_type"] == "snapshot").to_numpy()
    is_ask = (df["side"] == "ask").to_numpy()

    # Set event type values with numpy's where (no loops)
    arr["ev"] = np.where(
        is_snapshot,
        DEPTH_SNAPSHOT_EVENT,
        DEPTH_EVENT | np.where(is_ask, SELL_EVENT, BUY_EVENT) | EXCH_EVENT | LOCAL_EVENT,
    )

    # Directly set other fields from the dataframe
    arr["exch_ts"] = df["event_time"].to_numpy()
    arr["local_ts"] = df["received_time"].to_numpy()
    arr["px"] = df["price"].cast(pl.Float64).to_numpy()
    arr["qty"] = df["quantity"].cast(pl.Float64).to_numpy()

    validate_event_order(arr)

    return arr
