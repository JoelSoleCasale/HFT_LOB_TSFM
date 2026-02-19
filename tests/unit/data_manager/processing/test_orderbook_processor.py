"""Unit tests for OrderBookProcessor class."""

import pytest
import polars as pl
from datetime import date
from unittest.mock import patch

from data_manager.processing.orderbook_processor import OrderBookProcessor
from data_manager.processing.orderbook_request import OrderBookSnapshotRequest
from core.orderbook import OrderBook, OrderBookSnapshot


class TestOrderBookProcessorInit:
    """Test OrderBookProcessor initialization."""

    def test_init(self, temp_cache_root):
        """Test that initialization works correctly."""
        processor = OrderBookProcessor(temp_cache_root)
        assert processor.cache_root == temp_cache_root


class TestOrderBookProcessorInitializeOrderbook:
    """Test initialize_orderbook method."""

    def test_initialize_orderbook_no_previous_day(self, temp_cache_root):
        """Test initialization when no previous day data exists."""
        processor = OrderBookProcessor(temp_cache_root)

        result = processor.initialize_orderbook(
            exchange="binance_futures",
            symbol="BTCUSDT",
            date=date(2025, 7, 1),
            ob_init_prev_day_rows=100,
        )

        assert isinstance(result, OrderBookSnapshot)
        assert len(result.bid) == 0
        assert len(result.ask) == 0

    def test_initialize_orderbook_with_previous_day(self, temp_cache_root):
        """Test initialization when previous day data exists."""
        processor = OrderBookProcessor(temp_cache_root)

        # Create mock previous day data
        mock_df = pl.DataFrame(
            {
                "price": [50000.0, 50001.0, 49999.0],
                "quantity": [1.5, 2.0, 0.0],  # 0.0 should remove the order
                "side": ["bid", "ask", "bid"],
            }
        )

        # Create the actual file first
        prev_day_path = (
            temp_cache_root / "orderbook" / "binance_futures" / "BTCUSDT" / "2025-06-30.parquet"
        )
        prev_day_path.parent.mkdir(parents=True, exist_ok=True)
        mock_df.write_parquet(prev_day_path)

        result = processor.initialize_orderbook(
            exchange="binance_futures",
            symbol="BTCUSDT",
            date=date(2025, 7, 1),
            ob_init_prev_day_rows=100,
        )

        assert isinstance(result, OrderBookSnapshot)
        # Should have 2 orders (one with quantity 0 should be removed)
        assert len(result.bid) == 1
        assert len(result.ask) == 1
        assert result.bid[50000.0] == 1.5
        assert result.ask[50001.0] == 2.0


class TestOrderBookProcessorGenerateFullEventSampled:
    """Test generate_full_event_sampled method."""

    def test_generate_full_event_sampled_file_not_found(self, temp_cache_root):
        """Test that FileNotFoundError is raised when data file doesn't exist."""
        processor = OrderBookProcessor(temp_cache_root)
        request = OrderBookSnapshotRequest(
            exchange="binance_futures", symbol="BTCUSDT", date=date(2025, 7, 1), levels=5
        )

        with pytest.raises(FileNotFoundError):
            processor.generate_full_event_sampled(request)

    @patch("data_manager.processing.orderbook_processor.iter_slices")
    def test_generate_full_event_sampled_success(self, mock_iter_slices, temp_cache_root):
        """Test successful generation of event-sampled data."""
        processor = OrderBookProcessor(temp_cache_root)

        # Create mock data
        mock_data = pl.DataFrame(
            {
                "received_time": [1000, 1000, 1100, 1100],
                "price": [50000.0, 50001.0, 50002.0, 50003.0],
                "quantity": [1.0, 1.5, 2.0, 2.5],
                "side": ["bid", "ask", "bid", "ask"],
            }
        )

        # Mock iter_slices to return the data in one batch
        mock_iter_slices.return_value = [mock_data]

        request = OrderBookSnapshotRequest(
            exchange="binance_futures", symbol="BTCUSDT", date=date(2025, 7, 1), levels=5
        )

        # Create the data file first
        data_path = (
            temp_cache_root / "orderbook" / "binance_futures" / "BTCUSDT" / "2025-07-01.parquet"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        mock_data.write_parquet(data_path)

        result = processor.generate_full_event_sampled(request)

        assert isinstance(result, OrderBook)
        assert len(result) > 0
