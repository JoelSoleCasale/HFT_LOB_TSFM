"""Unit tests for IncrementalOBSampler class."""

import pytest
import numpy as np
from pathlib import Path
from datetime import timedelta
from unittest.mock import patch, MagicMock

from data_manager.processing.incremental_OB_sampler import IncrementalOBSampler


class TestIncrementalOBSamplerInit:
    """Test IncrementalOBSampler initialization."""

    def test_init_creates_cache_root(self, temp_cache_root):
        """Test that initialization creates cache root directory."""
        sampler = IncrementalOBSampler(temp_cache_root)
        assert sampler.cache_root.exists()
        assert sampler.cache_root == temp_cache_root


class TestIncrementalOBSamplerPathMethods:
    """Test path-related methods."""

    def test_get_data_path(
        self, temp_cache_root, sample_exchange, sample_symbol, sample_date
    ):
        """Test _get_data_path method."""
        sampler = IncrementalOBSampler(temp_cache_root)
        expected_path = (
            temp_cache_root
            / "orderbook"
            / sample_exchange
            / sample_symbol
            / "2025-07-01.parquet"
        )

        result = sampler._get_data_path(sample_exchange, sample_symbol, sample_date)
        assert result == expected_path

    def test_get_cache_path(
        self, temp_cache_root, sample_exchange, sample_symbol, sample_date
    ):
        """Test _get_cache_path method."""
        sampler = IncrementalOBSampler(temp_cache_root)
        expected_path = (
            temp_cache_root
            / "orderbook_snapshots"
            / sample_exchange
            / sample_symbol
            / "2025-07-01_L5.npz"
        )

        result = sampler._get_cache_path(
            sample_exchange, sample_symbol, sample_date, levels=5
        )
        assert result == expected_path


class TestIncrementalOBSamplerDtypeMethods:
    """Test dtype-related methods."""

    def test_get_ob_dtype_levels_5(self, temp_cache_root):
        """Test _get_ob_dtype with 5 levels."""
        sampler = IncrementalOBSampler(temp_cache_root)
        dtype = sampler._get_ob_dtype(levels=5)

        expected_fields = ["timestamp"]
        for i in range(1, 6):
            expected_fields.extend([f"ask{i}_price", f"ask{i}_qty"])
        for i in range(1, 6):
            expected_fields.extend([f"bid{i}_price", f"bid{i}_qty"])

        assert len(dtype) == len(expected_fields)
        for field_name in expected_fields:
            assert any(field[0] == field_name for field in dtype)

    def test_get_ob_dtype_levels_10(self, temp_cache_root):
        """Test _get_ob_dtype with 10 levels."""
        sampler = IncrementalOBSampler(temp_cache_root)
        dtype = sampler._get_ob_dtype(levels=10)

        assert len(dtype) == 1 + 10 * 4  # timestamp + (ask+bid) * (price+qty) * levels
        assert any(field[0] == "ask10_price" for field in dtype)
        assert any(field[0] == "bid10_price" for field in dtype)


class TestIncrementalOBSamplerOrderbookInitialization:
    """Test orderbook initialization methods."""

    @patch("src.data_manager.processing.incremental_OB_sampler.pl.scan_parquet")
    def test_initialize_orderbook_no_prev_day(
        self, mock_scan, temp_cache_root, sample_exchange, sample_symbol, sample_date
    ):
        """Test orderbook initialization when no previous day data exists."""
        sampler = IncrementalOBSampler(temp_cache_root)

        # Mock path to not exist
        with patch.object(sampler, "_get_data_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/path")

            ob = sampler._initialize_orderbook(
                sample_exchange, sample_symbol, sample_date, 100_000
            )

            assert "bid" in ob
            assert "ask" in ob
            assert len(ob["bid"]) == 0
            assert len(ob["ask"]) == 0
            mock_scan.assert_not_called()

    @patch("src.data_manager.processing.incremental_OB_sampler.pl.scan_parquet")
    def test_initialize_orderbook_with_prev_day(
        self, mock_scan, temp_cache_root, sample_exchange, sample_symbol, sample_date
    ):
        """Test orderbook initialization with previous day data."""
        sampler = IncrementalOBSampler(temp_cache_root)

        # Create mock data with proper chain
        mock_df = MagicMock()
        mock_select = MagicMock()
        mock_with_columns = MagicMock()
        mock_tail = MagicMock()
        mock_collect = MagicMock()

        mock_df.select.return_value = mock_select
        mock_select.with_columns.return_value = mock_with_columns
        mock_with_columns.tail.return_value = mock_tail
        mock_tail.collect.return_value = mock_collect
        mock_collect.iter_rows.return_value = [
            (50000.0, 1.5, "bid"),
            (50001.0, 2.0, "ask"),
            (49999.0, 0.0, "bid"),  # Remove this price level
        ]
        mock_scan.return_value = mock_df

        with patch.object(sampler, "_get_data_path") as mock_path:
            # Create a mock path that exists
            mock_existing_path = MagicMock()
            mock_existing_path.exists.return_value = True
            mock_path.return_value = mock_existing_path

            ob = sampler._initialize_orderbook(
                sample_exchange, sample_symbol, sample_date, 100_000
            )

            assert "bid" in ob
            assert "ask" in ob
            # Should have one bid (50000.0) and one ask (50001.0), but not 49999.0 (quantity=0)
            assert len(ob["bid"]) == 1
            assert len(ob["ask"]) == 1


class TestIncrementalOBSamplerPrecompute:
    """Test precompute_full_snapshots method."""

    @patch.object(IncrementalOBSampler, "_get_full_event_sampled")
    def test_precompute_single_exchange_symbol(
        self,
        mock_get_full,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
    ):
        """Test precompute with single exchange and symbol."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_get_full.return_value = np.array([])

        end_date = sample_date + timedelta(days=2)
        sampler.precompute_full_snapshots(
            exchange=sample_exchange,
            symbol=sample_symbol,
            start_date=sample_date,
            end_date=end_date,
            levels=5,
        )

        # Should be called 3 times (3 days)
        assert mock_get_full.call_count == 3

    @patch.object(IncrementalOBSampler, "_get_full_event_sampled")
    def test_precompute_multiple_exchanges_symbols(
        self, mock_get_full, temp_cache_root, sample_date
    ):
        """Test precompute with multiple exchanges and symbols."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_get_full.return_value = np.array([])

        exchanges = ["binance_futures", "binance_spot"]
        symbols = ["BTCUSDT", "ETHUSDT"]
        end_date = sample_date + timedelta(days=1)

        sampler.precompute_full_snapshots(
            exchange=exchanges,
            symbol=symbols,
            start_date=sample_date,
            end_date=end_date,
            levels=5,
        )

        # Should be called 2 exchanges * 2 symbols * 2 days = 8 times
        assert mock_get_full.call_count == 8

    @patch.object(IncrementalOBSampler, "_get_full_event_sampled")
    def test_precompute_handles_exceptions(
        self,
        mock_get_full,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
    ):
        """Test precompute handles exceptions gracefully."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_get_full.side_effect = [Exception("Test error"), np.array([])]

        end_date = sample_date + timedelta(days=1)
        # Should not raise exception even if one day fails
        sampler.precompute_full_snapshots(
            exchange=sample_exchange,
            symbol=sample_symbol,
            start_date=sample_date,
            end_date=end_date,
            levels=5,
        )

        assert mock_get_full.call_count == 2


class TestIncrementalOBSamplerEventSampling:
    """Test event sampling methods."""

    @patch.object(IncrementalOBSampler, "_get_full_event_sampled")
    def test_sample_by_events_no_sampling(
        self,
        mock_get_full,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
    ):
        """Test sample_by_events with sample_rate <= 1."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_data = np.array(
            [(1000, 50000.0, 1.5, 50001.0, 2.0)],
            dtype=[
                ("timestamp", "i8"),
                ("bid1_price", "f8"),
                ("bid1_qty", "f8"),
                ("ask1_price", "f8"),
                ("ask1_qty", "f8"),
            ],
        )
        mock_get_full.return_value = mock_data

        result = sampler.sample_by_events(
            sample_exchange, sample_symbol, sample_date, sample_rate=1
        )

        assert np.array_equal(result, mock_data)

    @patch.object(IncrementalOBSampler, "_get_full_event_sampled")
    def test_sample_by_events_with_sampling(
        self,
        mock_get_full,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
    ):
        """Test sample_by_events with sample_rate > 1."""
        sampler = IncrementalOBSampler(temp_cache_root)
        # Create mock data with 6 rows
        mock_data = np.array(
            [
                (1000, 50000.0, 1.5, 50001.0, 2.0),
                (1001, 49999.0, 1.0, 50002.0, 1.8),
                (1002, 50001.0, 2.0, 50003.0, 2.5),
                (1003, 49998.0, 0.8, 50004.0, 3.0),
                (1004, 50002.0, 1.8, 50005.0, 1.5),
                (1005, 49997.0, 0.6, 50006.0, 2.2),
            ],
            dtype=[
                ("timestamp", "i8"),
                ("bid1_price", "f8"),
                ("bid1_qty", "f8"),
                ("ask1_price", "f8"),
                ("ask1_qty", "f8"),
            ],
        )
        mock_get_full.return_value = mock_data

        result = sampler.sample_by_events(
            sample_exchange, sample_symbol, sample_date, sample_rate=2
        )

        # Should keep every 2nd row (0, 2, 4) -> 3 rows
        assert len(result) == 3
        assert result[0]["timestamp"] == 1000
        assert result[1]["timestamp"] == 1002
        assert result[2]["timestamp"] == 1004


class TestIncrementalOBSamplerTimeSampling:
    """Test time-based sampling methods."""

    @patch.object(IncrementalOBSampler, "_get_full_event_sampled")
    def test_sample_by_time_no_sampling(
        self,
        mock_get_full,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
    ):
        """Test sample_by_time with time_delta_ns <= 0."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_data = np.array(
            [(1000, 50000.0, 1.5, 50001.0, 2.0)],
            dtype=[
                ("timestamp", "i8"),
                ("bid1_price", "f8"),
                ("bid1_qty", "f8"),
                ("ask1_price", "f8"),
                ("ask1_qty", "f8"),
            ],
        )
        mock_get_full.return_value = mock_data

        result = sampler.sample_by_time(
            sample_exchange, sample_symbol, sample_date, time_delta_ns=0
        )

        assert np.array_equal(result, mock_data)

    @patch.object(IncrementalOBSampler, "_get_full_event_sampled")
    def test_sample_by_time_empty_data(
        self,
        mock_get_full,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
    ):
        """Test sample_by_time with empty data."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_data = np.array([], dtype=[("timestamp", "i8")])
        mock_get_full.return_value = mock_data

        result = sampler.sample_by_time(
            sample_exchange, sample_symbol, sample_date, time_delta_ns=1000
        )

        assert len(result) == 0

    @patch.object(IncrementalOBSampler, "_get_full_event_sampled")
    def test_sample_by_time_with_sampling(
        self,
        mock_get_full,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
    ):
        """Test sample_by_time with time-based sampling."""
        sampler = IncrementalOBSampler(temp_cache_root)
        # Create mock data with timestamps every 500ns, sample every 1000ns
        mock_data = np.array(
            [
                (1000, 50000.0, 1.5, 50001.0, 2.0),
                (1500, 49999.0, 1.0, 50002.0, 1.8),
                (2000, 50001.0, 2.0, 50003.0, 2.5),
                (2500, 49998.0, 0.8, 50004.0, 3.0),
                (3000, 50002.0, 1.8, 50005.0, 1.5),
            ],
            dtype=[
                ("timestamp", "i8"),
                ("bid1_price", "f8"),
                ("bid1_qty", "f8"),
                ("ask1_price", "f8"),
                ("ask1_qty", "f8"),
            ],
        )
        mock_get_full.return_value = mock_data

        result = sampler.sample_by_time(
            sample_exchange, sample_symbol, sample_date, time_delta_ns=1000
        )

        # Should sample at 1000, 2000, 3000 -> 3 rows
        assert len(result) == 3
        assert result[0]["timestamp"] == 1000
        assert result[1]["timestamp"] == 2000
        assert result[2]["timestamp"] == 3000


class TestIncrementalOBSamplerCacheManagement:
    """Test cache management methods."""

    def test_clear_cache_all(self, temp_cache_root):
        """Test clearing all cache."""
        sampler = IncrementalOBSampler(temp_cache_root)

        # Create some cache files
        cache_dir = temp_cache_root / "orderbook_snapshots"
        cache_dir.mkdir(parents=True)
        test_file = cache_dir / "test.npz"
        test_file.touch()

        sampler.clear_cache()

        # Cache directory should be empty
        assert not any(cache_dir.iterdir())

    def test_clear_cache_exchange(self, temp_cache_root):
        """Test clearing cache for specific exchange."""
        sampler = IncrementalOBSampler(temp_cache_root)

        # Create cache structure
        cache_dir = temp_cache_root / "orderbook_snapshots"
        exchange_dir = cache_dir / "binance_futures"
        exchange_dir.mkdir(parents=True)
        test_file = exchange_dir / "test.npz"
        test_file.touch()

        # Create another exchange
        other_exchange_dir = cache_dir / "binance_spot"
        other_exchange_dir.mkdir()
        other_test_file = other_exchange_dir / "test.npz"
        other_test_file.touch()

        sampler.clear_cache(exchange="binance_futures")

        # Only binance_futures should be cleared
        assert not exchange_dir.exists()
        assert other_exchange_dir.exists()
        assert other_test_file.exists()

    def test_clear_cache_symbol(self, temp_cache_root):
        """Test clearing cache for specific symbol."""
        sampler = IncrementalOBSampler(temp_cache_root)

        # Create cache structure
        cache_dir = temp_cache_root / "orderbook_snapshots"
        symbol_dir = cache_dir / "binance_futures" / "BTCUSDT"
        symbol_dir.mkdir(parents=True)
        test_file = symbol_dir / "test.npz"
        test_file.touch()

        # Create another symbol
        other_symbol_dir = cache_dir / "binance_futures" / "ETHUSDT"
        other_symbol_dir.mkdir(parents=True)
        other_test_file = other_symbol_dir / "test.npz"
        other_test_file.touch()

        sampler.clear_cache(exchange="binance_futures", symbol="BTCUSDT")

        # Only BTCUSDT should be cleared
        assert not symbol_dir.exists()
        assert other_symbol_dir.exists()
        assert other_test_file.exists()

    def test_clear_cache_date(self, temp_cache_root, sample_date):
        """Test clearing cache for specific date."""
        sampler = IncrementalOBSampler(temp_cache_root)

        # Create cache structure
        cache_dir = temp_cache_root / "orderbook_snapshots"
        date_dir = cache_dir / "binance_futures" / "BTCUSDT"
        date_dir.mkdir(parents=True)
        test_file = date_dir / "2025-07-01_L5.npz"
        test_file.touch()

        # Create another date
        other_test_file = date_dir / "2025-07-02_L5.npz"
        other_test_file.touch()

        sampler.clear_cache(
            exchange="binance_futures", symbol="BTCUSDT", date=sample_date
        )

        # Only 2025-07-01 files should be cleared
        assert not test_file.exists()
        assert other_test_file.exists()


class TestIncrementalOBSamplerGetFullEventSampled:
    """Test _get_full_event_sampled method."""

    @patch("numpy.load")
    def test_get_full_event_sampled_from_cache(
        self,
        mock_load,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
        sample_npz_data,
    ):
        """Test loading data from existing cache."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_load.return_value = {"data": sample_npz_data}

        # Create cache file
        cache_path = sampler._get_cache_path(
            sample_exchange, sample_symbol, sample_date, levels=5
        )
        cache_path.parent.mkdir(parents=True)
        cache_path.touch()

        result = sampler._get_full_event_sampled(
            sample_exchange, sample_symbol, sample_date, levels=5
        )

        assert np.array_equal(result, sample_npz_data)
        mock_load.assert_called_once_with(cache_path)

    @patch.object(IncrementalOBSampler, "_generate_full_event_sampled")
    @patch("numpy.savez_compressed")
    def test_get_full_event_sampled_generate_new(
        self,
        mock_save,
        mock_generate,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
        sample_npz_data,
    ):
        """Test generating new data when cache doesn't exist."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_generate.return_value = sample_npz_data

        # Don't create cache file
        result = sampler._get_full_event_sampled(
            sample_exchange, sample_symbol, sample_date, levels=5, force_regenerate=True
        )

        assert np.array_equal(result, sample_npz_data)
        mock_generate.assert_called_once_with(
            sample_exchange, sample_symbol, sample_date, 5
        )
        mock_save.assert_called_once()

    @patch("numpy.load")
    @patch.object(IncrementalOBSampler, "_get_ob_dtype")
    def test_get_full_event_sampled_from_higher_cache(
        self,
        mock_dtype,
        mock_load,
        temp_cache_root,
        sample_exchange,
        sample_symbol,
        sample_date,
        sample_npz_data,
    ):
        """Test using higher-level cache to generate lower-level data."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_load.return_value = {"data": sample_npz_data}

        # Create higher-level cache file (L10)
        cache_path = sampler._get_cache_path(
            sample_exchange, sample_symbol, sample_date, levels=10
        )
        cache_path.parent.mkdir(parents=True)
        cache_path.touch()

        # Mock dtype for lower level
        lower_dtype = [
            ("timestamp", "i8"),
            ("ask1_price", "f8"),
            ("ask1_qty", "f8"),
            ("bid1_price", "f8"),
            ("bid1_qty", "f8"),
        ]
        mock_dtype.return_value = lower_dtype

        result = sampler._get_full_event_sampled(
            sample_exchange, sample_symbol, sample_date, levels=5
        )

        assert result is not None
        mock_load.assert_called_once_with(cache_path)
        mock_dtype.assert_called_once_with(5)


@pytest.mark.unit
class TestIncrementalOBSamplerEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_levels(self, temp_cache_root):
        """Test with invalid levels parameter."""
        sampler = IncrementalOBSampler(temp_cache_root)

        # Test with 0 levels
        dtype = sampler._get_ob_dtype(levels=0)
        assert len(dtype) == 1  # Only timestamp
        assert dtype[0] == ("timestamp", "i8")

    def test_negative_levels(self, temp_cache_root):
        """Test with negative levels parameter."""
        sampler = IncrementalOBSampler(temp_cache_root)

        # Should handle negative levels gracefully
        dtype = sampler._get_ob_dtype(levels=-1)
        assert len(dtype) == 1  # Only timestamp

    @patch.object(IncrementalOBSampler, "_get_data_path")
    def test_missing_data_file(
        self, mock_path, temp_cache_root, sample_exchange, sample_symbol, sample_date
    ):
        """Test behavior when data file is missing."""
        sampler = IncrementalOBSampler(temp_cache_root)
        mock_path.return_value = Path("/nonexistent/file.parquet")

        with pytest.raises(FileNotFoundError):
            sampler._generate_full_event_sampled(
                sample_exchange, sample_symbol, sample_date
            )
