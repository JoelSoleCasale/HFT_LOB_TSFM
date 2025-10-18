"""Integration tests for OrderBook classes.

These tests focus on testing the interaction between different components,
real-world usage scenarios, and end-to-end workflows.
"""

import pytest
import polars as pl
import tempfile
import numpy as np
from pathlib import Path

from src.core.orderbook import OrderBook, OrderBookData, OrderBookSnapshot


class TestOrderBookIntegration:
    """Integration tests for OrderBook ecosystem."""

    @pytest.fixture
    def realistic_orderbook_data(self):
        """Create realistic orderbook data with proper price/quantity relationships."""
        timestamps = list(range(1000, 1100, 5))  # 20 snapshots, 5ms apart
        data = []

        base_price = 50000.0
        for i, ts in enumerate(timestamps):
            # Create realistic bid/ask spread
            spread = 1.0 + (i * 0.1)  # Increasing spread over time
            mid_price = base_price + (i * 0.5)  # Slight price drift

            row = {
                "timestamp": ts,
                "ask1_price": mid_price + (spread / 2),
                "ask1_qty": 1.5 + (i % 3) * 0.5,
                "ask2_price": mid_price + (spread / 2) + 0.5,
                "ask2_qty": 2.0 + (i % 4) * 0.3,
                "bid1_price": mid_price - (spread / 2),
                "bid1_qty": 1.8 + (i % 5) * 0.4,
                "bid2_price": mid_price - (spread / 2) - 0.5,
                "bid2_qty": 2.2 + (i % 3) * 0.6,
            }
            data.append(row)

        return pl.DataFrame(data)

    @pytest.fixture
    def high_frequency_data(self):
        """Create high-frequency orderbook data for performance testing."""
        # Generate 10,000 snapshots (microsecond timestamps)
        n_snapshots = 10000
        timestamps = list(range(1000000, 1000000 + n_snapshots))

        data = {
            "timestamp": timestamps,
            "ask1_price": [50000.0 + np.random.normal(0, 1) for _ in range(n_snapshots)],
            "ask1_qty": [1.0 + np.random.exponential(0.5) for _ in range(n_snapshots)],
            "bid1_price": [49999.0 + np.random.normal(0, 1) for _ in range(n_snapshots)],
            "bid1_qty": [1.0 + np.random.exponential(0.5) for _ in range(n_snapshots)],
        }

        return pl.DataFrame(data)

    def test_orderbook_snapshot_to_dataframe_workflow(self):
        """Test converting OrderBookSnapshot updates to DataFrame format."""
        # Create an orderbook snapshot and simulate live updates
        ob_snapshot = OrderBookSnapshot()

        # Simulate incoming market data updates
        updates = [
            ("bid", 49999.0, 1.5),
            ("bid", 49998.0, 2.0),
            ("ask", 50001.0, 1.8),
            ("ask", 50002.0, 2.2),
            ("bid", 49999.0, 2.0),  # Update existing price
            ("ask", 50001.0, 0.0),  # Cancel order
        ]

        # Apply updates
        for side, price, qty in updates:
            ob_snapshot.update(side, price, qty)

        # Verify final state
        assert len(ob_snapshot.bid) == 2
        assert len(ob_snapshot.ask) == 1
        assert ob_snapshot.bid[49999.0] == 2.0
        assert ob_snapshot.bid[49998.0] == 2.0
        assert ob_snapshot.ask[50002.0] == 2.2
        assert 50001.0 not in ob_snapshot.ask

        # Test that prices are properly sorted
        bid_prices = list(ob_snapshot.bid.keys())
        ask_prices = list(ob_snapshot.ask.keys())
        assert bid_prices == sorted(bid_prices, reverse=True)  # Descending
        assert ask_prices == sorted(ask_prices)  # Ascending

    def test_end_to_end_parquet_workflow(self, realistic_orderbook_data):
        """Test complete workflow: create OrderBook -> save to Parquet -> load -> analyze."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = Path(tmp_dir) / "test_orderbook.parquet"

            # Create OrderBook and save to Parquet
            original_ob = OrderBook(data=realistic_orderbook_data)
            original_ob.to_parquet(parquet_path)

            # Load from Parquet
            loaded_ob = OrderBook.from_parquet(parquet_path)

            # Verify data integrity
            assert len(loaded_ob) == len(original_ob)
            assert loaded_ob.levels == original_ob.levels
            assert loaded_ob.df.equals(original_ob.df)

            # Perform analysis on loaded data
            df = loaded_ob.df

            # Calculate spreads
            spreads = df.select((pl.col("ask1_price") - pl.col("bid1_price")).alias("spread"))[
                "spread"
            ]

            # Verify spreads are positive (basic market integrity check)
            assert spreads.min() > 0
            assert spreads.mean() > 0

            # Test sampling on loaded data
            sampled = loaded_ob.sample_by_events(2)
            assert len(sampled) == len(loaded_ob) // 2

    def test_multi_level_orderbook_operations(self, realistic_orderbook_data):
        """Test operations on multi-level orderbook data."""
        ob = OrderBook(data=realistic_orderbook_data)

        # Test level selection preserves data integrity
        single_level = ob.select_levels(1)
        assert single_level.levels == 1
        assert len(single_level) == len(ob)

        # Verify that level 1 data is subset of original
        l1_cols = ["timestamp", "ask1_price", "ask1_qty", "bid1_price", "bid1_qty"]
        assert set(single_level.df.columns) == set(l1_cols)

        # Test that selecting more levels than available returns same object
        same_ob = ob.select_levels(10)
        assert same_ob is ob

        # Test level selection with different levels
        df_l1 = single_level.df
        df_l2 = ob.df

        # Level 1 prices should match between original and reduced
        assert df_l1.select("ask1_price").equals(df_l2.select("ask1_price"))
        assert df_l1.select("bid1_price").equals(df_l2.select("bid1_price"))

    def test_time_based_sampling_with_realistic_data(self, realistic_orderbook_data):
        """Test time-based sampling with realistic market data scenarios."""
        ob = OrderBook(data=realistic_orderbook_data)

        # Test sampling at different intervals
        sample_10ms = ob.sample_by_time(10)  # 10ms intervals
        sample_20ms = ob.sample_by_time(20)  # 20ms intervals

        # Verify sampling reduces data size appropriately
        assert len(sample_10ms) <= len(ob)
        assert len(sample_20ms) <= len(sample_10ms)

        # Test with interpolation
        sample_interpolated = ob.sample_by_time(15, interpolate=True)

        # With interpolation, we should have regular time intervals
        if len(sample_interpolated) > 1:
            timestamps = sample_interpolated.df["timestamp"]
            time_diffs = timestamps.diff().drop_nulls()
            # Most time differences should be 15ms (allowing for start/end edge cases)
            assert time_diffs.mode().item() == 15

    def test_large_dataset_performance(self, high_frequency_data):
        """Test performance with large datasets (integration with memory management)."""
        # Create OrderBook with large dataset
        ob = OrderBook(data=high_frequency_data)

        # Test that operations complete in reasonable time and don't consume excessive memory
        assert len(ob) == 10000

        # Test sampling on large dataset
        sampled = ob.sample_by_events(100)  # Every 100th row
        assert len(sampled) == 100

        # Test time-based sampling
        time_sampled = ob.sample_by_time(1000)  # Every 1000 microseconds
        assert len(time_sampled) <= len(ob)

        # Test level selection
        reduced = ob.select_levels(1)
        assert reduced.levels == 1
        assert len(reduced) == len(ob)

    def test_data_validation_integration(self):
        """Test data validation across different creation methods."""
        # Test creating with valid data
        valid_data = {
            "timestamp": [1000, 1001, 1002],
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [1.0, 1.5, 2.0],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
        }
        df = pl.DataFrame(valid_data)

        # This should work fine
        ob1 = OrderBook(data=df)
        ob2 = OrderBook(data=OrderBookData(data=df))

        assert len(ob1) == len(ob2) == 3
        assert ob1.levels == ob2.levels == 1

        # Test with invalid data should raise appropriate errors
        invalid_data = {
            "timestamp": [1002, 1001, 1000],  # Unsorted
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [1.0, 1.5, 2.0],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
        }
        invalid_df = pl.DataFrame(invalid_data)

        with pytest.raises(ValueError, match="timestamp.*not sorted"):
            OrderBook(data=invalid_df)

    def test_orderbook_data_immutability_integration(self, realistic_orderbook_data):
        """Test that OrderBookData maintains immutability across operations."""
        ob_data = OrderBookData(data=realistic_orderbook_data)
        ob = OrderBook(data=ob_data)

        # Get DataFrame references
        df1 = ob_data.df
        df2 = ob.df
        df3 = ob_data.df

        # They should be equal but different objects
        assert df1.equals(df2)
        assert df1.equals(df3)
        assert df1 is not df2  # Different objects due to cloning
        assert df1 is not df3  # Different objects due to cloning

        # Modifying one shouldn't affect others (test immutability)
        # Note: Since these are clones, modifications wouldn't affect the original anyway,
        # but this tests the cloning behavior

        # Test operations return new instances
        sampled_ob = ob.sample_by_events(2)
        assert sampled_ob is not ob
        assert len(sampled_ob) != len(ob)

        # Original should be unchanged
        assert len(ob) == len(realistic_orderbook_data)

    def test_cross_level_data_consistency(self):
        """Test data consistency across different level configurations."""
        # Create multi-level data
        data = {
            "timestamp": [1000, 1001, 1002],
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [1.0, 1.5, 2.0],
            "ask2_price": [50002.0, 50003.0, 50004.0],
            "ask2_qty": [0.8, 1.2, 1.8],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
            "bid2_price": [49999.0, 50000.0, 50001.0],
            "bid2_qty": [1.2, 1.8, 2.2],
        }
        df = pl.DataFrame(data)

        ob_2level = OrderBook(data=df)
        ob_1level = ob_2level.select_levels(1)

        # Level 1 data should be consistent
        l2_level1_ask = ob_2level.df.select("ask1_price", "ask1_qty")
        l1_ask = ob_1level.df.select("ask1_price", "ask1_qty")

        assert l2_level1_ask.equals(l1_ask)

        # Best bid/ask should be level 1 in both cases
        assert ob_2level.df["ask1_price"].equals(ob_1level.df["ask1_price"])
        assert ob_2level.df["bid1_price"].equals(ob_1level.df["bid1_price"])

    def test_select_levels_removes_duplicates_when_deeper_levels_change(self):
        """Test that select_levels removes duplicated rows when only deeper levels change.

        When an orderbook update only affects deeper levels (e.g., level 3) and we
        select fewer levels (e.g., level 1-2), the resulting orderbook should remove
        the duplicated row since there was no actual update at the selected levels.
        """
        # Create 3-level orderbook with 3 snapshots
        # Snapshot 1 (ts=1000): Initial state
        # Snapshot 2 (ts=1001): Only level 3 changes
        # Snapshot 3 (ts=1002): Level 1 and 2 change
        data = {
            "timestamp": [1000, 1001, 1002],
            # Level 1 - same for first two snapshots
            "ask1_price": [50001.0, 50001.0, 50002.0],
            "ask1_qty": [1.0, 1.0, 1.5],
            "bid1_price": [50000.0, 50000.0, 50001.0],
            "bid1_qty": [1.5, 1.5, 2.0],
            # Level 2 - same for first two snapshots
            "ask2_price": [50002.0, 50002.0, 50003.0],
            "ask2_qty": [0.8, 0.8, 1.2],
            "bid2_price": [49999.0, 49999.0, 50000.0],
            "bid2_qty": [1.2, 1.2, 1.8],
            # Level 3 - CHANGES between first and second snapshot
            "ask3_price": [50003.0, 50003.5, 50004.0],
            "ask3_qty": [0.5, 0.9, 1.1],
            "bid3_price": [49998.0, 49997.5, 49999.0],
            "bid3_qty": [0.7, 1.3, 1.5],
        }
        df = pl.DataFrame(data)

        ob_3level = OrderBook(data=df)

        # Verify we start with 3 snapshots at 3 levels
        assert len(ob_3level) == 3
        assert ob_3level.levels == 3

        # Select only 2 levels
        ob_2level = ob_3level.select_levels(2)

        # After selecting 2 levels, the second snapshot should be removed
        # because levels 1-2 are identical between timestamp 1000 and 1001
        assert len(ob_2level) == 2
        assert ob_2level.levels == 2

        # Verify that the remaining timestamps are 1000 and 1002
        timestamps = ob_2level.df["timestamp"].to_list()
        assert timestamps == [1000, 1002]

        # Select only 1 level
        ob_1level = ob_3level.select_levels(1)

        # After selecting 1 level, the second snapshot should also be removed
        # (since level 1 is identical between timestamp 1000 and 1001)
        assert len(ob_1level) == 2
        assert ob_1level.levels == 1

        # Verify that the remaining timestamps are still 1000 and 1002
        timestamps = ob_1level.df["timestamp"].to_list()
        assert timestamps == [1000, 1002]

        # Verify data integrity - level 1 values at ts=1000 should match original
        original_level1_1000 = ob_3level.df.filter(pl.col("timestamp") == 1000).select(
            ["ask1_price", "ask1_qty", "bid1_price", "bid1_qty"]
        )
        reduced_level1_1000 = ob_1level.df.filter(pl.col("timestamp") == 1000).select(
            ["ask1_price", "ask1_qty", "bid1_price", "bid1_qty"]
        )
        assert original_level1_1000.equals(reduced_level1_1000)

    def test_select_levels_preserves_all_rows_when_selected_levels_always_change(self):
        """Test that select_levels keeps all rows when updates occur at selected levels.

        This is a complementary test to verify that rows are NOT removed when
        the selected levels actually have changes.
        """
        # Create 2-level orderbook where level 1 changes at every snapshot
        data = {
            "timestamp": [1000, 1001, 1002],
            # Level 1 - changes at every snapshot
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [1.0, 1.5, 2.0],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
            # Level 2 - also changes at every snapshot
            "ask2_price": [50002.0, 50003.0, 50004.0],
            "ask2_qty": [0.8, 1.2, 1.8],
            "bid2_price": [49999.0, 50000.0, 50001.0],
            "bid2_qty": [1.2, 1.8, 2.2],
        }
        df = pl.DataFrame(data)

        ob_2level = OrderBook(data=df)
        ob_1level = ob_2level.select_levels(1)

        # All snapshots should be preserved since level 1 changes at each timestamp
        assert len(ob_1level) == 3
        assert len(ob_2level) == 3

        # Timestamps should be unchanged
        assert ob_1level.df["timestamp"].to_list() == [1000, 1001, 1002]

    def test_realistic_market_scenario_simulation(self):
        """Simulate a realistic market scenario with multiple operations."""
        # Start with empty orderbook
        ob_snapshot = OrderBookSnapshot()

        # Simulate market opening with initial orders
        initial_orders = [
            ("bid", 49995.0, 5.0),
            ("bid", 49994.0, 3.0),
            ("bid", 49993.0, 2.0),
            ("ask", 50005.0, 4.0),
            ("ask", 50006.0, 3.5),
            ("ask", 50007.0, 2.5),
        ]

        for side, price, qty in initial_orders:
            ob_snapshot.update(side, price, qty)

        # Verify initial state
        assert len(ob_snapshot.bid) == 3
        assert len(ob_snapshot.ask) == 3

        # Simulate market activity: orders, modifications, cancellations
        market_activity = [
            ("bid", 49996.0, 2.0),  # New bid order
            ("ask", 50004.0, 3.0),  # New ask order (better price)
            ("bid", 49995.0, 0.0),  # Cancel best bid
            ("ask", 50005.0, 1.0),  # Reduce ask quantity
            ("bid", 49997.0, 4.0),  # Aggressive bid
            ("ask", 50003.0, 2.0),  # Very aggressive ask
        ]

        for side, price, qty in market_activity:
            ob_snapshot.update(side, price, qty)

        # Verify market state after activity
        # Best bid should be 49997.0, best ask should be 50003.0
        best_bid = max(ob_snapshot.bid.keys())
        best_ask = min(ob_snapshot.ask.keys())

        assert best_bid == 49997.0
        assert best_ask == 50003.0

        # Spread should be reasonable
        spread = best_ask - best_bid
        assert spread > 0
        assert spread < 100  # Sanity check

        # Create DataFrame from final snapshot state
        # This would be useful for converting live snapshots to DataFrame format
        snapshot_data = []
        timestamp = 1000

        # Extract top 2 levels for DataFrame
        bid_prices = list(ob_snapshot.bid.keys())[:2]
        ask_prices = list(ob_snapshot.ask.keys())[:2]

        if len(bid_prices) >= 2 and len(ask_prices) >= 2:
            row = {
                "timestamp": timestamp,
                "ask1_price": ask_prices[0],
                "ask1_qty": ob_snapshot.ask[ask_prices[0]],
                "ask2_price": ask_prices[1],
                "ask2_qty": ob_snapshot.ask[ask_prices[1]],
                "bid1_price": bid_prices[0],
                "bid1_qty": ob_snapshot.bid[bid_prices[0]],
                "bid2_price": bid_prices[1],
                "bid2_qty": ob_snapshot.bid[bid_prices[1]],
            }
            snapshot_data.append(row)

            # Create OrderBook from snapshot
            df = pl.DataFrame(snapshot_data)
            ob = OrderBook(data=df)

            assert ob.levels == 2
            assert len(ob) == 1


class TestOrderBookErrorHandlingIntegration:
    """Integration tests for error handling across components."""

    def test_invalid_data_propagation(self):
        """Test that invalid data errors propagate correctly through the system."""
        # Test various invalid data scenarios

        # Missing columns
        invalid_df1 = pl.DataFrame(
            {
                "timestamp": [1000, 1001],
                "ask1_price": [50001.0, 50002.0],
                # Missing other required columns
            }
        )

        with pytest.raises(ValueError):
            OrderBook(data=invalid_df1)

        # Wrong data types
        invalid_df2 = pl.DataFrame(
            {
                "timestamp": ["1000", "1001"],  # Should be int
                "ask1_price": [50001.0, 50002.0],
                "ask1_qty": [1.0, 1.5],
                "bid1_price": [50000.0, 50001.0],
                "bid1_qty": [1.5, 2.0],
            }
        )

        with pytest.raises(ValueError):
            OrderBook(data=invalid_df2)

    def test_file_io_error_handling(self):
        """Test error handling in file I/O operations."""
        ob = OrderBook(levels=1)

        # Test writing to invalid path
        with pytest.raises((PermissionError, FileNotFoundError, OSError)):
            ob.to_parquet("/invalid/path/file.parquet")

        # Test reading from nonexistent file
        with pytest.raises((FileNotFoundError, Exception)):
            OrderBook.from_parquet("nonexistent_file.parquet")
