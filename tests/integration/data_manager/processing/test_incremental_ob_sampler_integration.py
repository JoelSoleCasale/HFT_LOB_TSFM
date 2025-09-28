"""Integration tests for IncrementalOBSampler class."""

import pytest
import numpy as np
import polars as pl
from datetime import date, timedelta

from data_manager.processing.incremental_OB_sampler import IncrementalOBSampler


@pytest.mark.integration
class TestIncrementalOBSamplerIntegration:
    """Integration tests for IncrementalOBSampler."""

    def create_realistic_orderbook_data(
        self, start_timestamp: int = 1000, num_rows: int = 100, unique_prices: int = 50
    ):
        """Create realistic orderbook data for testing."""
        data = []
        base_price = 50000.0

        # Generate data with random number of rows per timestamp
        current_timestamp = start_timestamp
        rows_generated = 0

        while rows_generated < num_rows:
            # Decide how many rows will have this timestamp (0 to 10)
            rows_for_timestamp = min(
                np.random.randint(0, 11), num_rows - rows_generated
            )

            for j in range(rows_for_timestamp):
                # Simulate realistic price movements
                price_change = np.random.choice(np.linspace(-5, 5, unique_prices))
                price = base_price + price_change

                # Simulate orderbook updates
                side = "bid" if rows_generated % 2 == 0 else "ask"
                quantity = np.random.uniform(0.1, 5.0)

                # Occasionally remove orders (quantity = 0)
                if rows_generated % 20 == 0:
                    quantity = 0.0

                data.append((current_timestamp, price, quantity, side))
                rows_generated += 1

            # Move to next timestamp
            current_timestamp += 100  # 100ns intervals

        df = pl.DataFrame(
            data, schema=["timestamp", "price", "quantity", "side"], orient="row"
        )
        df_with_time = df.with_columns(pl.col("timestamp").alias("received_time"))
        return df_with_time

    def test_caching_workflow(self, temp_cache_root):
        """Test caching functionality with real data."""
        # Setup
        sampler = IncrementalOBSampler(temp_cache_root)
        exchange = "test_exchange"
        symbol = "TESTUSDT"
        test_date = date(2025, 7, 2)

        # Create orderbook data
        orderbook_data = self.create_realistic_orderbook_data(num_rows=500)

        # Save data to parquet file
        data_path = sampler._get_data_path(exchange, symbol, test_date)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        orderbook_data.write_parquet(data_path)

        # First call - should generate and cache data
        result1 = sampler._get_full_event_sampled(
            exchange, symbol, test_date, levels=5, force_regenerate=False
        )

        # Verify cache file was created
        cache_path = sampler._get_cache_path(exchange, symbol, test_date, levels=5)
        assert cache_path.exists()

        # Second call - should load from cache
        result2 = sampler._get_full_event_sampled(
            exchange, symbol, test_date, levels=5, force_regenerate=False
        )

        # Compare all fields, allowing for small floating point differences
        for name in result1.dtype.names:
            arr1 = result1[name]
            arr2 = result2[name]
            if np.issubdtype(arr1.dtype, np.floating):
                # Use nan_to_num to handle NaNs, then compare with tolerance
                assert np.allclose(
                    np.nan_to_num(arr1, nan=-1),
                    np.nan_to_num(arr2, nan=-1),
                    atol=1e-7,
                    equal_nan=True,
                )
            else:
                assert np.array_equal(arr1, arr2)

    def test_higher_level_cache_usage(self, temp_cache_root):
        """Test using higher-level cache to generate lower-level data."""
        # Setup
        sampler = IncrementalOBSampler(temp_cache_root)
        exchange = "test_exchange"
        symbol = "TESTUSDT"
        test_date = date(2025, 7, 3)

        # Create orderbook data
        orderbook_data = self.create_realistic_orderbook_data(num_rows=500)

        # Save data to parquet file
        data_path = sampler._get_data_path(exchange, symbol, test_date)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        orderbook_data.write_parquet(data_path)

        # Generate L10 data first
        result_l10 = sampler._get_full_event_sampled(
            exchange, symbol, test_date, levels=10, force_regenerate=True
        )

        # Now request L5 data - should use L10 cache
        result_l5 = sampler._get_full_event_sampled(
            exchange, symbol, test_date, levels=5, force_regenerate=False
        )

        # Verify results
        assert len(result_l5) == len(result_l10)
        assert result_l5.dtype.names[0] == "timestamp"

        # L5 should have fewer fields than L10
        assert len(result_l5.dtype.names) < len(result_l10.dtype.names)

        # L5 should not have ask6_price, bid6_price, etc.
        assert "ask6_price" not in result_l5.dtype.names
        assert "bid6_price" not in result_l5.dtype.names

        # But L10 should have these fields
        assert "ask6_price" in result_l10.dtype.names
        assert "bid6_price" in result_l10.dtype.names

    def test_previous_day_initialization(self, temp_cache_root):
        """Test orderbook initialization from previous day data."""
        # Setup
        sampler = IncrementalOBSampler(temp_cache_root)
        exchange = "test_exchange"
        symbol = "TESTUSDT"
        test_date = date(2025, 7, 4)
        prev_date = test_date - timedelta(days=1)

        # Create previous day data
        prev_day_data = self.create_realistic_orderbook_data(
            start_timestamp=900, num_rows=200
        )

        # Save previous day data
        prev_data_path = sampler._get_data_path(exchange, symbol, prev_date)
        prev_data_path.parent.mkdir(parents=True, exist_ok=True)
        prev_day_data.write_parquet(prev_data_path)

        # Create current day data
        current_day_data = self.create_realistic_orderbook_data(
            start_timestamp=1000, num_rows=100
        )

        # Save current day data
        current_data_path = sampler._get_data_path(exchange, symbol, test_date)
        current_data_path.parent.mkdir(parents=True, exist_ok=True)
        current_day_data.write_parquet(current_data_path)

        # Generate data with previous day initialization
        result = sampler._generate_full_event_sampled(
            exchange=exchange,
            symbol=symbol,
            date=test_date,
            levels=5,
            ob_init_prev_day_rows=50,
            verbose=False,
        )

        # Verify results
        assert len(result) > 0
        assert result["timestamp"][0] >= 1000  # Should start from current day

        # The orderbook should be initialized from previous day data
        # This is harder to verify directly, but we can check that the process completes

    def test_sampling_methods_integration(self, temp_cache_root):
        """Test integration of sampling methods."""
        # Setup
        sampler = IncrementalOBSampler(temp_cache_root)
        exchange = "test_exchange"
        symbol = "TESTUSDT"
        test_date = date(2025, 7, 5)

        # Create orderbook data with known timestamps
        timestamps = np.arange(1000, 11000, 100)  # 100ns intervals, 100 points
        data = []

        for i, ts in enumerate(timestamps):
            price = 50000.0 + i * 0.1
            side = "bid" if i % 2 == 0 else "ask"
            quantity = 1.0 + (i % 10) * 0.1

            data.append((ts, price, quantity, side))

        df = pl.DataFrame(
            data, schema=["timestamp", "price", "quantity", "side"], orient="row"
        )
        df_with_time = df.with_columns(pl.col("timestamp").alias("received_time"))

        # Save data
        data_path = sampler._get_data_path(exchange, symbol, test_date)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_time.write_parquet(data_path)

        # Full event sampled data
        full_data = sampler._get_full_event_sampled(
            exchange, symbol, test_date, levels=5
        )

        # Test event-based sampling
        event_sampled = sampler.sample_by_events(
            exchange, symbol, test_date, sample_rate=5, levels=5
        )

        # Should have approximately 1/5 of the original data
        assert len(full_data) / 5 - 1 <= len(event_sampled) <= len(full_data) / 5 + 1

        # Test time-based sampling
        time_sampled = sampler.sample_by_time(
            exchange, symbol, test_date, time_delta_ns=500, levels=5
        )

        # Should have data sampled every 500ns
        assert len(time_sampled) > 0
        timestamps_sampled = time_sampled["timestamp"]

        # Verify sampling intervals (allowing for some tolerance)
        intervals = np.diff(timestamps_sampled)
        assert np.all((intervals >= 500) & (intervals % 500 == 0))

    def test_precompute_workflow(self, temp_cache_root):
        """Test precompute workflow with multiple dates."""
        # Setup
        sampler = IncrementalOBSampler(temp_cache_root)
        exchange = "test_exchange"
        symbol = "TESTUSDT"
        start_date = date(2025, 7, 6)
        end_date = date(2025, 7, 8)

        # Create data for multiple days
        dates = [start_date, start_date + timedelta(days=1)]

        for d in dates:
            orderbook_data = self.create_realistic_orderbook_data(
                start_timestamp=1000, num_rows=200
            )

            data_path = sampler._get_data_path(exchange, symbol, d)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            orderbook_data.write_parquet(data_path)

        # Run precompute
        sampler.precompute_full_snapshots(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            levels=5,
        )

        # Verify cache files were created for both days
        for d in dates:
            cache_path = sampler._get_cache_path(exchange, symbol, d, levels=5)
            assert cache_path.exists()

            # Verify we can load the cached data
            data = sampler._get_full_event_sampled(
                exchange, symbol, d, levels=5, force_regenerate=False
            )
            assert len(data) > 0

    def test_cache_pruning(self, temp_cache_root):
        """Test cache pruning functionality."""
        # Setup
        sampler = IncrementalOBSampler(temp_cache_root)
        exchange = "test_exchange"
        symbol = "TESTUSDT"
        test_date = date(2025, 7, 9)

        # Create orderbook data
        orderbook_data = self.create_realistic_orderbook_data(num_rows=300)

        # Save data
        data_path = sampler._get_data_path(exchange, symbol, test_date)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        orderbook_data.write_parquet(data_path)

        # Generate L3 data first
        sampler._get_full_event_sampled(
            exchange, symbol, test_date, levels=3, force_regenerate=True
        )

        # Verify L3 cache exists
        l3_cache = sampler._get_cache_path(exchange, symbol, test_date, levels=3)
        assert l3_cache.exists()

        # Generate L10 data - should prune L3 cache
        sampler._get_full_event_sampled(
            exchange, symbol, test_date, levels=10, force_regenerate=True
        )

        # L3 cache should be pruned
        assert not l3_cache.exists()

        # L10 cache should exist
        l10_cache = sampler._get_cache_path(exchange, symbol, test_date, levels=10)
        assert l10_cache.exists()

    @pytest.mark.slow
    def test_large_dataset_processing(self, temp_cache_root):
        """Test processing of larger datasets."""
        # Setup
        sampler = IncrementalOBSampler(temp_cache_root)
        exchange = "test_exchange"
        symbol = "TESTUSDT"
        test_date = date(2025, 7, 10)

        # Create larger dataset
        orderbook_data = self.create_realistic_orderbook_data(
            start_timestamp=1000, num_rows=10000, unique_prices=10
        )

        # Save data
        data_path = sampler._get_data_path(exchange, symbol, test_date)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        orderbook_data.write_parquet(data_path)

        # Process with smaller batch size to test batching
        result = sampler._generate_full_event_sampled(
            exchange=exchange,
            symbol=symbol,
            date=test_date,
            levels=10,
            batch_size=1000,
            verbose=False,
        )
        # Verify results
        assert len(result) > 0
        assert result["timestamp"][-1] >= orderbook_data["timestamp"][-1]

        # Verify no memory issues (array should be properly sized)
        assert len(result) < 10000  # Should be less than input due to deduplication
