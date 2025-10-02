"""Unit tests for OrderBook and OrderBookData classes."""

import pytest
import polars as pl
import tempfile
from pathlib import Path

from src.core.orderbook import OrderBook, OrderBookData, OrderBookSnapshot


class TestOrderBookSnapshot:
    """Tests for OrderBookSnapshot class."""

    def test_init(self):
        """Test OrderBookSnapshot initialization."""
        ob_snapshot = OrderBookSnapshot()
        assert hasattr(ob_snapshot, "bid")
        assert hasattr(ob_snapshot, "ask")

    def test_getitem_bid(self):
        """Test accessing bid side via indexing."""
        ob_snapshot = OrderBookSnapshot()
        bid_dict = ob_snapshot["bid"]
        assert bid_dict is not None

    def test_getitem_ask(self):
        """Test accessing ask side via indexing."""
        ob_snapshot = OrderBookSnapshot()
        ask_dict = ob_snapshot["ask"]
        assert ask_dict is not None

    def test_getitem_invalid_key(self):
        """Test accessing invalid key raises KeyError."""
        ob_snapshot = OrderBookSnapshot()
        with pytest.raises(KeyError):
            _ = ob_snapshot["invalid"]

    def test_bid_sorting_order(self):
        """Test that bid side is sorted in descending order (highest first)."""
        ob_snapshot = OrderBookSnapshot()
        ob_snapshot.bid[100.0] = 1.0
        ob_snapshot.bid[101.0] = 2.0
        ob_snapshot.bid[99.0] = 3.0

        # Bids should be sorted in descending order
        prices = list(ob_snapshot.bid.keys())
        assert prices == [101.0, 100.0, 99.0]

    def test_ask_sorting_order(self):
        """Test that ask side is sorted in ascending order (lowest first)."""
        ob_snapshot = OrderBookSnapshot()
        ob_snapshot.ask[101.0] = 1.0
        ob_snapshot.ask[100.0] = 2.0
        ob_snapshot.ask[102.0] = 3.0

        # Asks should be sorted in ascending order
        prices = list(ob_snapshot.ask.keys())
        assert prices == [100.0, 101.0, 102.0]

    def test_update_add_bid_order(self):
        """Test adding a new bid order via update method."""
        ob_snapshot = OrderBookSnapshot()

        # Add a new bid order
        ob_snapshot.update("bid", 50000.0, 1.5)

        assert 50000.0 in ob_snapshot.bid
        assert ob_snapshot.bid[50000.0] == 1.5

    def test_update_add_ask_order(self):
        """Test adding a new ask order via update method."""
        ob_snapshot = OrderBookSnapshot()

        # Add a new ask order
        ob_snapshot.update("ask", 50001.0, 2.0)

        assert 50001.0 in ob_snapshot.ask
        assert ob_snapshot.ask[50001.0] == 2.0

    def test_update_modify_existing_order(self):
        """Test modifying an existing order via update method."""
        ob_snapshot = OrderBookSnapshot()

        # Add initial order
        ob_snapshot.update("bid", 50000.0, 1.5)
        assert ob_snapshot.bid[50000.0] == 1.5

        # Modify the order
        ob_snapshot.update("bid", 50000.0, 2.5)
        assert ob_snapshot.bid[50000.0] == 2.5

    def test_update_cancel_order_with_zero_quantity(self):
        """Test canceling an order by setting quantity to zero."""
        ob_snapshot = OrderBookSnapshot()

        # Add initial orders
        ob_snapshot.update("bid", 50000.0, 1.5)
        ob_snapshot.update("ask", 50001.0, 2.0)

        # Verify orders exist
        assert 50000.0 in ob_snapshot.bid
        assert 50001.0 in ob_snapshot.ask

        # Cancel bid order
        ob_snapshot.update("bid", 50000.0, 0.0)
        assert 50000.0 not in ob_snapshot.bid

        # Cancel ask order
        ob_snapshot.update("ask", 50001.0, 0.0)
        assert 50001.0 not in ob_snapshot.ask

    def test_update_cancel_nonexistent_order(self):
        """Test canceling a nonexistent order (should not raise error)."""
        ob_snapshot = OrderBookSnapshot()

        # Try to cancel orders that don't exist - should not raise error
        ob_snapshot.update("bid", 50000.0, 0.0)
        ob_snapshot.update("ask", 50001.0, 0.0)

        # Orderbook should still be empty
        assert len(ob_snapshot.bid) == 0
        assert len(ob_snapshot.ask) == 0

    def test_update_multiple_orders_maintains_sorting(self):
        """Test that multiple updates maintain proper sorting order."""
        ob_snapshot = OrderBookSnapshot()

        # Add multiple bid orders (should be sorted descending)
        ob_snapshot.update("bid", 50000.0, 1.0)
        ob_snapshot.update("bid", 50002.0, 1.5)
        ob_snapshot.update("bid", 50001.0, 2.0)

        bid_prices = list(ob_snapshot.bid.keys())
        assert bid_prices == [50002.0, 50001.0, 50000.0]  # Descending

        # Add multiple ask orders (should be sorted ascending)
        ob_snapshot.update("ask", 50005.0, 1.0)
        ob_snapshot.update("ask", 50003.0, 1.5)
        ob_snapshot.update("ask", 50004.0, 2.0)

        ask_prices = list(ob_snapshot.ask.keys())
        assert ask_prices == [50003.0, 50004.0, 50005.0]  # Ascending

    def test_update_with_invalid_side_raises_error(self):
        """Test that updating with invalid side raises KeyError."""
        ob_snapshot = OrderBookSnapshot()

        with pytest.raises(KeyError):
            ob_snapshot.update("invalid_side", 50000.0, 1.0)


class TestOrderBookData:
    """Tests for OrderBookData class."""

    @pytest.fixture
    def sample_orderbook_data(self):
        """Create sample orderbook DataFrame for testing."""
        data = {
            "timestamp": [1000, 1001, 1002],
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [2.0, 1.8, 2.5],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
        }
        return pl.DataFrame(data)

    @pytest.fixture
    def sample_multilevel_data(self):
        """Create sample multi-level orderbook DataFrame."""
        data = {
            "timestamp": [1000, 1001, 1002],
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [2.0, 1.8, 2.5],
            "ask2_price": [50002.0, 50003.0, 50004.0],
            "ask2_qty": [1.8, 2.5, 3.0],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
            "bid2_price": [49999.0, 50000.0, 50001.0],
            "bid2_qty": [1.0, 1.5, 2.0],
        }
        return pl.DataFrame(data)

    def test_init_with_data(self, sample_orderbook_data):
        """Test OrderBookData initialization with DataFrame."""
        ob_data = OrderBookData(data=sample_orderbook_data)
        assert ob_data.levels == 1
        assert ob_data.df.height == 3
        # Check if timestamp column is sorted (data is already provided sorted)
        assert ob_data.df["timestamp"].is_sorted()

    def test_init_with_levels(self):
        """Test OrderBookData initialization with levels only."""
        ob_data = OrderBookData(levels=5)
        assert ob_data.levels == 5
        assert ob_data.df.height == 0
        expected_cols = len(OrderBookData.get_orderbook_columns(5))
        assert len(ob_data.df.columns) == expected_cols

    def test_init_no_params_raises_error(self):
        """Test that initialization without data or levels raises ValueError."""
        with pytest.raises(ValueError, match="Either data or levels must be provided"):
            OrderBookData()

    def test_df_property_returns_clone(self, sample_orderbook_data):
        """Test that df property returns a clone, not the original."""
        ob_data = OrderBookData(data=sample_orderbook_data)
        df1 = ob_data.df
        df2 = ob_data.df

        # Should be different objects but same content
        assert df1 is not df2
        assert df1.equals(df2)

    def test_levels_property(self, sample_multilevel_data):
        """Test levels property returns correct number."""
        ob_data = OrderBookData(data=sample_multilevel_data)
        assert ob_data.levels == 2

    def test_get_orderbook_schema(self):
        """Test schema generation for different levels."""
        schema_1 = OrderBookData.get_orderbook_schema(1)
        expected_1 = [
            ("timestamp", pl.Int64),
            ("ask1_price", pl.Float64),
            ("bid1_price", pl.Float64),
            ("ask1_qty", pl.Float64),
            ("bid1_qty", pl.Float64),
        ]
        assert schema_1 == expected_1

        schema_2 = OrderBookData.get_orderbook_schema(2)
        assert len(schema_2) == 9  # timestamp + 2 * 2 * 2 levels

    def test_get_orderbook_columns(self):
        """Test column name generation."""
        cols_1 = OrderBookData.get_orderbook_columns(1)
        expected_1 = ["timestamp", "ask1_price", "bid1_price", "ask1_qty", "bid1_qty"]
        assert cols_1 == expected_1

        cols_3 = OrderBookData.get_orderbook_columns(3)
        assert len(cols_3) == 13  # timestamp + 3 * 2 * 2 levels

    def test_infer_levels(self, sample_multilevel_data):
        """Test level inference from DataFrame."""
        levels = OrderBookData._infer_levels(sample_multilevel_data)
        assert levels == 2

    def test_infer_levels_no_price_columns(self):
        """Test level inference fails with no price columns."""
        df = pl.DataFrame({"timestamp": [1000], "qty": [1.0]})
        with pytest.raises(ValueError, match="No price columns found"):
            OrderBookData._infer_levels(df)

    def test_validate_structure_valid(self, sample_orderbook_data):
        """Test structure validation with valid DataFrame."""
        # Should not raise any exception
        OrderBookData._validate_structure(sample_orderbook_data)

    def test_validate_structure_wrong_columns(self):
        """Test structure validation fails with wrong columns."""
        df = pl.DataFrame(
            {
                "timestamp": [1000],
                "wrong_col": [1.0],
                "ask1_price": [50000.0],
                "ask1_qty": [1.0],
                "bid1_price": [49999.0],
                "bid1_qty": [1.0],
            }
        )
        with pytest.raises(
            ValueError, match="DataFrame does not have correct orderbook column names"
        ):
            OrderBookData._validate_structure(df)

    def test_validate_structure_wrong_dtype(self):
        """Test structure validation fails with wrong data types."""
        df = pl.DataFrame(
            {
                "timestamp": ["1000"],  # Wrong type - should be Int64
                "ask1_price": [50000.0],
                "ask1_qty": [1.0],
                "bid1_price": [49999.0],
                "bid1_qty": [1.0],
            }
        )
        with pytest.raises(ValueError, match="Column 'timestamp' has incorrect dtype"):
            OrderBookData._validate_structure(df)

    def test_validate_structure_unsorted_timestamp(self):
        """Test structure validation fails with unsorted timestamps."""
        df = pl.DataFrame(
            {
                "timestamp": [1002, 1001, 1000],  # Unsorted
                "ask1_price": [50000.0, 50001.0, 50002.0],
                "ask1_qty": [1.0, 1.0, 1.0],
                "bid1_price": [49999.0, 50000.0, 50001.0],
                "bid1_qty": [1.0, 1.0, 1.0],
            }
        )
        with pytest.raises(ValueError, match="DataFrame 'timestamp' column is not sorted"):
            OrderBookData._validate_structure(df)


class TestOrderBook:
    """Tests for OrderBook class."""

    @pytest.fixture
    def sample_orderbook_df(self):
        """Create sample orderbook DataFrame."""
        data = {
            "timestamp": [1000, 1001, 1002, 1003],
            "ask1_price": [50001.0, 50002.0, 50003.0, 50004.0],
            "ask1_qty": [2.0, 1.8, 2.5, 3.0],
            "bid1_price": [50000.0, 50001.0, 50002.0, 50003.0],
            "bid1_qty": [1.5, 2.0, 2.5, 1.8],
        }
        return pl.DataFrame(data)

    @pytest.fixture
    def sample_multilevel_orderbook_df(self):
        """Create sample multi-level orderbook DataFrame."""
        data = {
            "timestamp": [1000, 1001, 1002],
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [2.0, 1.8, 2.5],
            "ask2_price": [50002.0, 50003.0, 50004.0],
            "ask2_qty": [1.8, 2.5, 3.0],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
            "bid2_price": [49999.0, 50000.0, 50001.0],
            "bid2_qty": [1.0, 1.5, 2.0],
        }
        return pl.DataFrame(data)

    def test_init_with_dataframe(self, sample_orderbook_df):
        """Test OrderBook initialization with DataFrame."""
        ob = OrderBook(data=sample_orderbook_df)
        assert ob.levels == 1
        assert len(ob) == 4

    def test_init_with_orderbook_data(self, sample_orderbook_df):
        """Test OrderBook initialization with OrderBookData."""
        ob_data = OrderBookData(data=sample_orderbook_df)
        ob = OrderBook(data=ob_data)
        assert ob.levels == 1
        assert len(ob) == 4

    def test_init_with_levels_only(self):
        """Test OrderBook initialization with levels only."""
        ob = OrderBook(levels=3)
        assert ob.levels == 3
        assert len(ob) == 0

    def test_len(self, sample_orderbook_df):
        """Test __len__ method."""
        ob = OrderBook(data=sample_orderbook_df)
        assert len(ob) == 4

    def test_repr(self, sample_orderbook_df):
        """Test __repr__ method."""
        ob = OrderBook(data=sample_orderbook_df)
        repr_str = repr(ob)
        assert "OrderBook" in repr_str
        assert "levels=1" in repr_str
        assert "snapshots=4" in repr_str

    def test_levels_property(self, sample_multilevel_orderbook_df):
        """Test levels property."""
        ob = OrderBook(data=sample_multilevel_orderbook_df)
        assert ob.levels == 2

    def test_df_property(self, sample_orderbook_df):
        """Test df property returns the underlying DataFrame."""
        ob = OrderBook(data=sample_orderbook_df)
        df = ob.df
        assert df.height == 4
        assert "timestamp" in df.columns

    def test_from_parquet(self):
        """Test loading OrderBook from Parquet file."""
        # Create a temporary parquet file
        data = {
            "timestamp": [1000, 1001],
            "ask1_price": [50001.0, 50002.0],
            "ask1_qty": [2.0, 1.8],
            "bid1_price": [50000.0, 50001.0],
            "bid1_qty": [1.5, 2.0],
        }
        df = pl.DataFrame(data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test.parquet"
            df.write_parquet(tmp_path)

            ob = OrderBook.from_parquet(tmp_path)
            assert len(ob) == 2
            assert ob.levels == 1

    def test_to_parquet(self, sample_orderbook_df):
        """Test saving OrderBook to Parquet file."""
        ob = OrderBook(data=sample_orderbook_df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test.parquet"

            ob.to_parquet(tmp_path)

            # Verify file was created and can be read back
            loaded_df = pl.read_parquet(tmp_path)
            assert loaded_df.height == 4
            assert loaded_df.equals(sample_orderbook_df)

    def test_select_levels_reduce(self, sample_multilevel_orderbook_df):
        """Test selecting fewer levels from orderbook."""
        ob = OrderBook(data=sample_multilevel_orderbook_df)
        ob_reduced = ob.select_levels(1)

        assert ob_reduced.levels == 1
        assert len(ob_reduced) == 3

        # Should only have level 1 columns
        expected_cols = ["timestamp", "ask1_price", "bid1_price", "ask1_qty", "bid1_qty"]
        assert set(ob_reduced.df.columns) == set(expected_cols)

    def test_select_levels_same_or_more(self, sample_multilevel_orderbook_df):
        """Test selecting same or more levels returns same orderbook."""
        ob = OrderBook(data=sample_multilevel_orderbook_df)

        # Same levels
        ob_same = ob.select_levels(2)
        assert ob_same is ob

        # More levels
        ob_more = ob.select_levels(5)
        assert ob_more is ob

    def test_sample_by_events(self, sample_orderbook_df):
        """Test sampling by events (every nth row)."""
        ob = OrderBook(data=sample_orderbook_df)
        ob_sampled = ob.sample_by_events(2)

        assert len(ob_sampled) == 2
        # Should have timestamps 1000 and 1002 (every 2nd row)
        timestamps = ob_sampled.df["timestamp"].to_list()
        assert timestamps == [1000, 1002]

    def test_sample_by_time_no_interpolation(self):
        """Test sampling by time without interpolation."""
        data = {
            "timestamp": [1000, 1050, 1100, 1150, 1200],
            "ask1_price": [50001.0, 50002.0, 50003.0, 50004.0, 50005.0],
            "ask1_qty": [2.0, 1.8, 2.5, 3.0, 2.2],
            "bid1_price": [50000.0, 50001.0, 50002.0, 50003.0, 50004.0],
            "bid1_qty": [1.5, 2.0, 2.5, 1.8, 1.9],
        }
        ob = OrderBook(data=pl.DataFrame(data))

        # Sample every 100 time units
        ob_sampled = ob.sample_by_time(100, interpolate=False)

        # Should group timestamps and keep last value in each group
        timestamps = ob_sampled.df["timestamp"].to_list()
        expected_timestamps = [1000, 1100, 1200]
        assert timestamps == expected_timestamps

    def test_sample_by_time_with_interpolation(self):
        """Test sampling by time with interpolation."""
        data = {
            "timestamp": [1000, 1150, 1300],
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [2.0, 1.8, 2.5],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
        }
        ob = OrderBook(data=pl.DataFrame(data))

        # Sample every 100 time units with interpolation
        ob_sampled = ob.sample_by_time(100, interpolate=True)

        # Should have evenly spaced timestamps
        timestamps = ob_sampled.df["timestamp"].to_list()
        expected_timestamps = [1000, 1100, 1200, 1300]
        assert timestamps == expected_timestamps

        # Forward-filled values should be present
        assert ob_sampled.df.height == 4

    def test_sample_by_time_single_row_no_interpolation(self):
        """Test sampling by time with single row doesn't interpolate."""
        data = {
            "timestamp": [1000],
            "ask1_price": [50001.0],
            "ask1_qty": [2.0],
            "bid1_price": [50000.0],
            "bid1_qty": [1.5],
        }
        ob = OrderBook(data=pl.DataFrame(data))

        # Sample with interpolation, but only one row
        ob_sampled = ob.sample_by_time(100, interpolate=True)

        # Should still have only one row
        assert len(ob_sampled) == 1

    def test_sample_by_time_empty_dataframe(self):
        """Test sampling by time with empty DataFrame."""
        ob = OrderBook(levels=1)
        ob_sampled = ob.sample_by_time(100)

        assert len(ob_sampled) == 0
        assert ob_sampled.levels == 1

    def test_orderbook_immutability(self, sample_orderbook_df):
        """Test that OrderBook operations return new instances."""
        ob1 = OrderBook(data=sample_orderbook_df)
        ob2 = ob1.sample_by_events(2)
        ob3 = ob1.select_levels(1)

        # Original should be unchanged
        assert len(ob1) == 4
        assert len(ob2) == 2

        # ob3 should be the same instance since levels match
        assert ob3 is ob1
