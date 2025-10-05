"""Unit tests for OrderBookCacheManager class."""

from datetime import date
from unittest.mock import patch, MagicMock

from data_manager.processing.orderbook_cache_manager import OrderBookCacheManager
from data_manager.processing.orderbook_request import OrderBookSnapshotRequest
from core.orderbook import OrderBook


class TestOrderBookCacheManagerInit:
    """Test OrderBookCacheManager initialization."""

    def test_init(self, temp_cache_root):
        """Test that initialization works correctly."""
        cache_manager = OrderBookCacheManager(temp_cache_root)
        assert cache_manager.cache_root == temp_cache_root
        assert cache_manager.processor is not None


class TestOrderBookCacheManagerGetCachedOrderbook:
    """Test get_cached_orderbook method."""

    def test_get_cached_orderbook_existing_cache(self, temp_cache_root):
        """Test loading from existing cache."""
        cache_manager = OrderBookCacheManager(temp_cache_root)

        # Create a mock cached file
        cache_path = (
            temp_cache_root
            / "orderbook_snapshots"
            / "binance_futures"
            / "BTCUSDT"
            / "2025-07-01_L5.parquet"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a simple parquet file
        import polars as pl

        mock_data = pl.DataFrame(
            {
                "timestamp": [1000, 1100],
                "ask1_price": [50001.0, 50002.0],
                "ask1_qty": [1.0, 1.5],
                "bid1_price": [50000.0, 50001.0],
                "bid1_qty": [1.0, 1.5],
            }
        )
        mock_data.write_parquet(cache_path)

        request = OrderBookSnapshotRequest(
            exchange="binance_futures", symbol="BTCUSDT", date=date(2025, 7, 1), levels=5
        )

        result = cache_manager.get_cached_orderbook(request, force_regenerate=False)

        assert isinstance(result, OrderBook)
        assert len(result) == 2

    def test_get_cached_orderbook_force_regenerate(self, temp_cache_root):
        """Test force regeneration of cache."""
        cache_manager = OrderBookCacheManager(temp_cache_root)

        # Mock the processor
        mock_orderbook = MagicMock()
        cache_manager.processor.generate_full_event_sampled = MagicMock(
            return_value=mock_orderbook
        )

        request = OrderBookSnapshotRequest(
            exchange="binance_futures", symbol="BTCUSDT", date=date(2025, 7, 1), levels=5
        )

        result = cache_manager.get_cached_orderbook(request, force_regenerate=True)

        assert result == mock_orderbook
        cache_manager.processor.generate_full_event_sampled.assert_called_once_with(request)

    def test_get_cached_orderbook_higher_level_cache(self, temp_cache_root):
        """Test using higher-level cache when available."""
        cache_manager = OrderBookCacheManager(temp_cache_root)

        # Create a higher-level cache file
        cache_dir = temp_cache_root / "orderbook_snapshots" / "binance_futures" / "BTCUSDT"
        cache_dir.mkdir(parents=True, exist_ok=True)

        higher_cache_path = cache_dir / "2025-07-01_L10.parquet"
        import polars as pl

        mock_data = pl.DataFrame(
            {
                "timestamp": [1000, 1100],
                "ask1_price": [50001.0, 50002.0],
                "ask1_qty": [1.0, 1.5],
                "bid1_price": [50000.0, 50001.0],
                "bid1_qty": [1.0, 1.5],
            }
        )
        mock_data.write_parquet(higher_cache_path)

        request = OrderBookSnapshotRequest(
            exchange="binance_futures", symbol="BTCUSDT", date=date(2025, 7, 1), levels=5
        )

        # Mock the select_levels method
        mock_orderbook = MagicMock()
        mock_orderbook.select_levels.return_value = mock_orderbook

        with patch(
            "data_manager.processing.orderbook_cache_manager.OrderBook.from_parquet",
            return_value=mock_orderbook,
        ):
            result = cache_manager.get_cached_orderbook(request, force_regenerate=False)

        assert result == mock_orderbook
        mock_orderbook.select_levels.assert_called_once_with(5)

    def test_prune_lower_level_caches(self, temp_cache_root):
        """Test that lower-level caches are pruned."""
        cache_manager = OrderBookCacheManager(temp_cache_root)

        # Create cache directory with multiple level files
        cache_dir = temp_cache_root / "orderbook_snapshots" / "binance_futures" / "BTCUSDT"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create files for different levels
        l3_file = cache_dir / "2025-07-01_L3.parquet"
        l5_file = cache_dir / "2025-07-01_L5.parquet"
        l10_file = cache_dir / "2025-07-01_L10.parquet"

        # Create empty files
        l3_file.touch()
        l5_file.touch()
        l10_file.touch()

        request = OrderBookSnapshotRequest(
            exchange="binance_futures", symbol="BTCUSDT", date=date(2025, 7, 1), levels=5
        )

        # Mock the processor to return a mock orderbook
        mock_orderbook = MagicMock()
        cache_manager.processor.generate_full_event_sampled = MagicMock(
            return_value=mock_orderbook
        )

        _ = cache_manager.get_cached_orderbook(request, force_regenerate=True)

        # L3 file should be deleted, L5 and L10 should remain
        assert not l3_file.exists()
        assert l5_file.exists()
        assert l10_file.exists()
