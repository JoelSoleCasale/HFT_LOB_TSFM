"""Global pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import date, timedelta
import polars as pl
from core.orderbook import OrderBook, OrderBookSnapshot


@pytest.fixture
def temp_cache_root():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_date():
    """Sample date for testing."""
    return date(2025, 7, 1)


@pytest.fixture
def sample_exchange():
    """Sample exchange for testing."""
    return "binance_futures"


@pytest.fixture
def sample_symbol():
    """Sample symbol for testing."""
    return "BTCUSDT"


@pytest.fixture
def sample_incremental_orderbook_data():
    """Create sample orderbook data for testing."""
    # Create sample orderbook updates
    data = [
        (1000, 50000.0, 1.5, "bid"),  # timestamp, price, quantity, side
        (1001, 50001.0, 2.0, "ask"),
        (1002, 49999.0, 1.0, "bid"),
        (1003, 50002.0, 1.8, "ask"),
        (1004, 50000.0, 0.0, "bid"),  # Remove bid at 50000
        (1005, 50003.0, 2.5, "ask"),
    ]

    df = pl.DataFrame(data, schema=["timestamp", "price", "quantity", "side"])
    return df


@pytest.fixture
def sample_parquet_file(
    temp_cache_root, sample_exchange, sample_symbol, sample_date, sample_incremental_orderbook_data
):
    """Create a sample parquet file for testing."""
    file_path = (
        temp_cache_root
        / "orderbook"
        / sample_exchange
        / sample_symbol
        / f"{sample_date.strftime('%Y-%m-%d')}.parquet"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Add received_time column if not present
    if "received_time" not in sample_incremental_orderbook_data.columns:
        df_with_time = sample_incremental_orderbook_data.with_columns(
            pl.col("timestamp").alias("received_time")
        )
    else:
        df_with_time = sample_incremental_orderbook_data

    df_with_time.write_parquet(file_path)
    return file_path


@pytest.fixture
def sample_previous_day_data():
    """Create sample previous day data for orderbook initialization."""
    data = [
        (50000, 1.5, "bid"),
        (50001, 2.0, "ask"),
        (49999, 1.0, "bid"),
        (50002, 1.8, "ask"),
    ]

    df = pl.DataFrame(data, schema=["price", "quantity", "side"])
    return df


@pytest.fixture
def sample_previous_day_parquet(
    temp_cache_root,
    sample_exchange,
    sample_symbol,
    sample_date,
    sample_previous_day_data,
):
    """Create a sample previous day parquet file."""
    prev_date = sample_date - timedelta(days=1)
    file_path = (
        temp_cache_root
        / "orderbook"
        / sample_exchange
        / sample_symbol
        / f"{prev_date.strftime('%Y-%m-%d')}.parquet"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df_with_time = sample_previous_day_data.with_columns(
        pl.lit(900).alias("timestamp"), pl.lit(900).alias("received_time")
    )

    df_with_time.write_parquet(file_path)
    return file_path


@pytest.fixture
def empty_orderbook():
    """Create an empty orderbook structure."""
    return OrderBookSnapshot


@pytest.fixture
def populated_orderbook():
    """Create a populated orderbook structure."""
    ob = OrderBookSnapshot
    ob["bid"][49999.0] = 1.0
    ob["bid"][50000.0] = 1.5
    ob["ask"][50001.0] = 2.0
    ob["ask"][50002.0] = 1.8
    return ob


@pytest.fixture
def sample_snapshot_df() -> OrderBook:
    """Create sample OBSnapshotDataFrame data for testing."""
    data = {
        "timestamp": [1000, 1001, 1004, 1033],
        "ask1_price": [50001.0, 50002.0, 50003.0, 50004.0],
        "ask1_qty": [2.0, 1.8, 2.5, 3.0],
        "ask2_price": [50002.0, 50003.0, 50004.0, 50005.0],
        "ask2_qty": [1.8, 2.5, 3.0, 2.2],
        "bid1_price": [50000.0, 50001.0, 50002.0, 50003.0],
        "bid1_qty": [1.5, 2.0, 2.5, 1.8],
        "bid2_price": [49999.0, 50000.0, 50001.0, 50002.0],
        "bid2_qty": [1.0, 1.5, 2.0, 2.5],
    }

    return OrderBook(pl.DataFrame(data))
