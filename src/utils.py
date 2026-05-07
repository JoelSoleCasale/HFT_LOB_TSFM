import polars as pl
import numpy as np
from pathlib import Path
from typing import Iterator, Callable, Literal
from datetime import timedelta, date
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
from loguru import logger
from definitions import ROOT_DIR


def setup_logging(log_file: str | Path | None = None, level: str = "INFO") -> None:
    """
    Set up loguru logging configuration.

    Console format uses colorized YYYY-MM-DD HH:mm:ss timestamps.
    File handler rotates at 10 MB and retains logs for 7 days.
    """
    if log_file is None:
        log_file = Path(ROOT_DIR / "logs" / "logfile.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    logger.add(
        log_file,
        level=level,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


def iter_slices(
    df: pl.DataFrame | pl.LazyFrame, n_rows: int = 10_000
) -> Iterator[pl.DataFrame | pl.LazyFrame]:
    """Iterate over DataFrame/LazyFrame in slices."""
    row_count = df.lazy().select(pl.len()).collect().item()
    for offset in range(0, row_count, n_rows):
        yield df.slice(offset, n_rows)


def cache_func(
    func: Callable[[DataRequest], pl.DataFrame | None],
    cache_root: Path,
    load_cached_data: bool = True,
) -> Callable[[DataRequest], tuple[pl.DataFrame | None, bool]]:
    """Wrap *func* with Parquet caching.

    The returned callable returns ``(result, from_cache)`` where ``from_cache``
    is True when data was loaded from an existing file rather than recomputed.
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


def date_range(start_date: date, end_date: date) -> Iterator[date]:
    """Generate dates from start_date to end_date inclusive."""
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def get_ob_path(d: date) -> Path:
    """Return the canonical path for a BTCUSDT L20 orderbook snapshot parquet file."""
    return (
        ROOT_DIR
        / f'data/orderbook_snapshots/binance_futures/BTCUSDT/{d.strftime("%Y-%m-%d")}_L20.parquet'
    )


def get_hftbacktest_array(
    df: pl.DataFrame,
    df_time_unit: Literal["s", "ms", "us", "ns"] = "ms",
    target_time_unit: Literal["s", "ms", "us", "ns"] = "ns",
) -> np.ndarray:
    """
    Convert a Polars DataFrame representing an incremental orderbook event stream
    into a structured NumPy array compatible with hftbacktest.
    """
    arr = np.zeros(len(df), dtype=event_dtype)

    is_snapshot = (df["event_type"] == "snapshot").to_numpy()
    is_ask = (df["side"] == "ask").to_numpy()

    arr["ev"] = np.where(
        is_snapshot,
        DEPTH_SNAPSHOT_EVENT,
        DEPTH_EVENT | np.where(is_ask, SELL_EVENT, BUY_EVENT) | EXCH_EVENT | LOCAL_EVENT,
    )

    time_factors = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
    time_multiplier = time_factors[df_time_unit] // time_factors[target_time_unit]
    arr["exch_ts"] = (df["event_time"].to_numpy() * time_multiplier).astype(np.int64)
    arr["local_ts"] = (df["received_time"].to_numpy() * time_multiplier).astype(np.int64)
    arr["px"] = df["price"].cast(pl.Float64).to_numpy()
    arr["qty"] = df["quantity"].cast(pl.Float64).to_numpy()

    validate_event_order(arr)

    return arr
