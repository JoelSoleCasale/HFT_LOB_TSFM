"""Unit tests for IncrementalOrderBookSampler class."""

from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock

from data_manager.processing.incremental_orderbook_sampler import IncrementalOrderBookSampler
from data_manager.processing.orderbook_request import OrderBookSnapshotRequest
from data_manager.processing.orderbook_cache_manager import OrderBookCacheManager


class TestIncrementalOrderBookSamplerInit:
    """Test IncrementalOrderBookSampler initialization."""

    def test_init_creates_cache_root(self, temp_cache_root):
        """Test that initialization creates cache root directory."""
        sampler = IncrementalOrderBookSampler(temp_cache_root)
        assert sampler.cache_root.exists()
        assert sampler.cache_root == temp_cache_root
        assert isinstance(sampler.cache_manager, OrderBookCacheManager)


class TestOrderBookSamplingRequest:
    """Test OrderBookSamplingRequest class."""

    def test_get_path(self):
        """Test path generation for sampling request."""
        request = OrderBookSnapshotRequest(
            exchange="binance_futures", symbol="BTCUSDT", date=date(2025, 7, 1), levels=5
        )
        expected_path = Path("orderbook_snapshots/binance_futures/BTCUSDT/2025-07-01_L5.parquet")
        assert request.get_path() == expected_path

    def test_get_path_different_levels(self):
        """Test path generation with different levels."""
        request = OrderBookSnapshotRequest(
            exchange="binance_spot", symbol="ETHUSDT", date=date(2025, 7, 15), levels=10
        )
        expected_path = Path("orderbook_snapshots/binance_spot/ETHUSDT/2025-07-15_L10.parquet")
        assert request.get_path() == expected_path


class TestIncrementalOrderBookSamplerMethods:
    """Test IncrementalOrderBookSampler methods."""

    def test_get_orderbook_single_request(
        self, temp_cache_root, sample_exchange, sample_symbol, sample_date
    ):
        """Test get_orderbook method with single request."""
        sampler = IncrementalOrderBookSampler(temp_cache_root)

        # Mock the cache manager to avoid actual processing
        mock_orderbook = MagicMock()
        sampler.cache_manager.get_cached_orderbook = MagicMock(return_value=mock_orderbook)

        result = sampler.get_orderbook(
            exchange=sample_exchange, symbol=sample_symbol, date=sample_date, levels=5
        )

        assert result == mock_orderbook
        sampler.cache_manager.get_cached_orderbook.assert_called_once()

        # Check that the request was created with correct parameters
        call_args = sampler.cache_manager.get_cached_orderbook.call_args
        request = call_args[0][0]
        assert isinstance(request, OrderBookSnapshotRequest)
        assert request.exchange == sample_exchange
        assert request.symbol == sample_symbol
        assert request.date == sample_date
        assert request.levels == 5

    def test_precompute_full_snapshots_single_exchange_symbol(self, temp_cache_root):
        """Test precompute with single exchange and symbol."""
        sampler = IncrementalOrderBookSampler(temp_cache_root)

        # Mock the cache manager
        sampler.cache_manager.get_cached_orderbook = MagicMock()

        sampler.precompute_full_snapshots(
            exchange="binance_futures",
            symbol="BTCUSDT",
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 3),
            levels=5,
        )

        # Should be called for each date
        assert sampler.cache_manager.get_cached_orderbook.call_count == 3

    def test_precompute_full_snapshots_multiple_exchanges_symbols(self, temp_cache_root):
        """Test precompute with multiple exchanges and symbols."""
        sampler = IncrementalOrderBookSampler(temp_cache_root)

        # Mock the cache manager
        sampler.cache_manager.get_cached_orderbook = MagicMock()

        sampler.precompute_full_snapshots(
            exchange=["binance_futures", "binance_spot"],
            symbol=["BTCUSDT", "ETHUSDT"],
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 2),
            levels=5,
        )

        # Should be called for each combination: 2 exchanges * 2 symbols * 2 dates = 8 calls
        assert sampler.cache_manager.get_cached_orderbook.call_count == 8

    @patch("data_manager.processing.incremental_orderbook_sampler.logger")
    def test_precompute_handles_exceptions(self, mock_logger, temp_cache_root):
        """Test that precompute handles exceptions gracefully."""
        sampler = IncrementalOrderBookSampler(temp_cache_root)

        # Mock the cache manager to raise an exception
        sampler.cache_manager.get_cached_orderbook = MagicMock(side_effect=Exception("Test error"))

        # Should not raise an exception
        sampler.precompute_full_snapshots(
            exchange="binance_futures",
            symbol="BTCUSDT",
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 1),
            levels=5,
        )

        # Should log the error
        mock_logger.error.assert_called_once()
