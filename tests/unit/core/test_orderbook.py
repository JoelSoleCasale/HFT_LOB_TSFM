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

    @pytest.mark.parametrize("lazy", [False, True])
    def test_init_with_data(self, sample_orderbook_data, lazy):
        """Test OrderBookData initialization with DataFrame and LazyFrame."""
        data = sample_orderbook_data.lazy() if lazy else sample_orderbook_data
        ob_data = OrderBookData(data=data)
        assert ob_data.levels == 1
        assert ob_data.is_lazy == lazy

        if lazy:
            # For LazyFrame, collect to check height
            assert ob_data.df.collect().height == 3
        else:
            assert ob_data.df.height == 3
            # Check if timestamp column is sorted (data is already provided sorted)
            assert ob_data.df["timestamp"].is_sorted()

    def test_init_with_levels(self):
        """Test OrderBookData initialization with levels only."""
        ob_data = OrderBookData(levels=5)
        assert ob_data.levels == 5
        assert not ob_data.is_lazy
        assert ob_data.df.height == 0
        expected_cols = len(OrderBookData.get_orderbook_columns(5))
        assert len(ob_data.df.collect_schema().names()) == expected_cols

    def test_init_no_params_raises_error(self):
        """Test that initialization without data or levels raises ValueError."""
        with pytest.raises(ValueError, match="Either data or levels must be provided"):
            OrderBookData()

    def test_df_property_returns_clone(self, sample_orderbook_data):
        """Test that df property returns a clone for eager mode."""
        ob_data = OrderBookData(data=sample_orderbook_data)
        df1 = ob_data.df
        df2 = ob_data.df

        # Should be different objects but same content (eager mode)
        assert df1 is not df2
        assert df1.equals(df2)

    def test_df_property_lazy_mode(self, sample_orderbook_data):
        """Test that df property returns LazyFrame for lazy mode."""
        ob_data = OrderBookData(data=sample_orderbook_data.lazy())
        df1 = ob_data.df
        df2 = ob_data.df

        # For lazy mode, returns the same LazyFrame reference
        assert isinstance(df1, pl.LazyFrame)
        assert isinstance(df2, pl.LazyFrame)

    def test_levels_property(self, sample_multilevel_data):
        """Test levels property returns correct number."""
        ob_data = OrderBookData(data=sample_multilevel_data)
        assert ob_data.levels == 2

    @pytest.mark.parametrize("lazy", [False, True])
    def test_infer_levels(self, sample_multilevel_data, lazy):
        """Test level inference from DataFrame and LazyFrame."""
        data = sample_multilevel_data.lazy() if lazy else sample_multilevel_data
        levels = OrderBookData._infer_levels(data)
        assert levels == 2

    def test_get_orderbook_schema(self):
        """Test schema generation for different levels."""
        schema_1 = OrderBookData.get_orderbook_schema(1)
        expected_1 = [
            ("timestamp", pl.Int64),
            ("ask1_price", pl.Float64),
            ("ask1_qty", pl.Float64),
            ("bid1_price", pl.Float64),
            ("bid1_qty", pl.Float64),
        ]
        assert schema_1 == expected_1

        schema_2 = OrderBookData.get_orderbook_schema(2)
        assert len(schema_2) == 9  # timestamp + 2 * 2 * 2 levels

    def test_infer_levels_no_price_columns(self):
        """Test level inference fails with no price columns."""
        df = pl.DataFrame({"timestamp": [1000], "qty": [1.0]})
        with pytest.raises(ValueError, match="No price columns found"):
            OrderBookData._infer_levels(df)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_validate_structure_valid(self, sample_orderbook_data, lazy):
        """Test structure validation with valid DataFrame and LazyFrame."""
        data = sample_orderbook_data.lazy() if lazy else sample_orderbook_data
        # Should not raise any exception
        OrderBookData._validate_structure(data)

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
        assert not ob.is_lazy

    def test_init_with_lazyframe(self, sample_orderbook_df):
        """Test OrderBook initialization with LazyFrame."""
        ob = OrderBook(data=sample_orderbook_df.lazy())
        assert ob.levels == 1
        assert len(ob) == 4
        assert ob.is_lazy

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
        assert not ob.is_lazy

    @pytest.mark.parametrize("lazy", [False, True])
    def test_len(self, sample_orderbook_df, lazy):
        """Test __len__ method with DataFrame and LazyFrame."""
        data = sample_orderbook_df.lazy() if lazy else sample_orderbook_df
        ob = OrderBook(data=data)
        assert len(ob) == 4

    @pytest.mark.parametrize("lazy", [False, True])
    def test_repr(self, sample_orderbook_df, lazy):
        """Test __repr__ method with DataFrame and LazyFrame."""
        data = sample_orderbook_df.lazy() if lazy else sample_orderbook_df
        ob = OrderBook(data=data)
        repr_str = repr(ob)
        assert "OrderBook" in repr_str
        assert "levels=1" in repr_str
        assert "snapshots=4" in repr_str
        mode = "lazy" if lazy else "eager"
        assert f"mode={mode}" in repr_str

    def test_levels_property(self, sample_multilevel_orderbook_df):
        """Test levels property."""
        ob = OrderBook(data=sample_multilevel_orderbook_df)
        assert ob.levels == 2

    @pytest.mark.parametrize("lazy", [False, True])
    def test_df_property(self, sample_orderbook_df, lazy):
        """Test df property returns the underlying DataFrame or LazyFrame."""
        data = sample_orderbook_df.lazy() if lazy else sample_orderbook_df
        ob = OrderBook(data=data)
        assert len(ob) == 4
        assert "timestamp" in ob.df.collect_schema().names()

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

            # Test eager mode
            ob = OrderBook.from_parquet(tmp_path, lazy=False)
            assert len(ob) == 2
            assert ob.levels == 1
            assert not ob.is_lazy

            # Test lazy mode
            ob_lazy = OrderBook.from_parquet(tmp_path, lazy=True)
            assert len(ob_lazy) == 2
            assert ob_lazy.levels == 1
            assert ob_lazy.is_lazy

    @pytest.mark.parametrize("lazy", [False, True])
    def test_to_parquet(self, sample_orderbook_df, lazy):
        """Test saving OrderBook to Parquet file from DataFrame and LazyFrame."""
        data = sample_orderbook_df.lazy() if lazy else sample_orderbook_df
        ob = OrderBook(data=data)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test.parquet"

            ob.to_parquet(tmp_path)

            # Verify file was created and can be read back
            loaded_df = pl.read_parquet(tmp_path)
            assert loaded_df.height == 4
            assert loaded_df.equals(sample_orderbook_df)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_select_levels_reduce(self, sample_multilevel_orderbook_df, lazy):
        """Test selecting fewer levels from orderbook with DataFrame and LazyFrame."""
        data = sample_multilevel_orderbook_df.lazy() if lazy else sample_multilevel_orderbook_df
        ob = OrderBook(data=data)
        ob_reduced = ob.select_levels(1)

        assert ob_reduced.levels == 1
        assert len(ob_reduced) == 3
        assert ob_reduced.is_lazy == lazy

        # Should only have level 1 columns
        expected_cols = ["timestamp", "ask1_price", "bid1_price", "ask1_qty", "bid1_qty"]
        assert set(ob_reduced.df.collect_schema().names()) == set(expected_cols)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_select_levels_same_or_more(self, sample_multilevel_orderbook_df, lazy):
        """Test selecting same or more levels returns same orderbook."""
        data = sample_multilevel_orderbook_df.lazy() if lazy else sample_multilevel_orderbook_df
        ob = OrderBook(data=data)

        # Same levels
        ob_same = ob.select_levels(2)
        assert ob_same is ob

        # More levels
        ob_more = ob.select_levels(5)
        assert ob_more is ob

    @pytest.mark.parametrize("lazy", [False, True])
    def test_sample_by_events(self, sample_orderbook_df, lazy):
        """Test sampling by events (every nth row) with DataFrame and LazyFrame."""
        data = sample_orderbook_df.lazy() if lazy else sample_orderbook_df
        ob = OrderBook(data=data)
        ob_sampled = ob.sample_by_events(2)

        assert len(ob_sampled) == 2
        assert ob_sampled.is_lazy == lazy
        # Should have timestamps 1000 and 1002 (every 2nd row)
        if lazy:
            timestamps = ob_sampled.df.collect()["timestamp"].to_list()
        else:
            timestamps = ob_sampled.df["timestamp"].to_list()
        assert timestamps == [1000, 1002]

    @pytest.mark.parametrize("lazy", [False, True])
    def test_sample_by_time_no_interpolation(self, lazy):
        """Test sampling by time without interpolation."""
        data = {
            "timestamp": [1000, 1050, 1100, 1150, 1200],
            "ask1_price": [50001.0, 50002.0, 50003.0, 50004.0, 50005.0],
            "ask1_qty": [2.0, 1.8, 2.5, 3.0, 2.2],
            "bid1_price": [50000.0, 50001.0, 50002.0, 50003.0, 50004.0],
            "bid1_qty": [1.5, 2.0, 2.5, 1.8, 1.9],
        }
        df = pl.LazyFrame(data) if lazy else pl.DataFrame(data)
        ob = OrderBook(data=df)

        # Sample every 100 time units
        ob_sampled = ob.sample_by_time(100, interpolate=False)

        # Should group timestamps and keep last value in each group
        if lazy:
            timestamps = ob_sampled.df.collect()["timestamp"].to_list()
        else:
            timestamps = ob_sampled.df["timestamp"].to_list()
        expected_timestamps = [1000, 1100, 1200]
        assert timestamps == expected_timestamps

    @pytest.mark.parametrize("lazy", [False, True])
    def test_sample_by_time_with_interpolation(self, lazy):
        """Test sampling by time with interpolation."""
        data = {
            "timestamp": [1000, 1150, 1300],
            "ask1_price": [50001.0, 50002.0, 50003.0],
            "ask1_qty": [2.0, 1.8, 2.5],
            "bid1_price": [50000.0, 50001.0, 50002.0],
            "bid1_qty": [1.5, 2.0, 2.5],
        }
        df = pl.DataFrame(data)
        df = df.lazy() if lazy else df
        ob = OrderBook(data=df)

        # Sample every 100 time units with interpolation
        ob_sampled = ob.sample_by_time(100, interpolate=True)
        assert len(ob_sampled) == 4

        # Should have evenly spaced timestamps
        if lazy:
            timestamps = ob_sampled.df.collect()["timestamp"].to_list()
        else:
            timestamps = ob_sampled.df["timestamp"].to_list()
        expected_timestamps = [1000, 1100, 1200, 1300]
        assert timestamps == expected_timestamps

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

    def test_collect_from_lazy(self, sample_orderbook_df):
        """Test converting lazy OrderBook to eager via collect()."""
        ob_lazy = OrderBook(data=sample_orderbook_df.lazy())
        assert ob_lazy.is_lazy

        ob_eager = ob_lazy.collect()
        assert not ob_eager.is_lazy
        assert isinstance(ob_eager.df, pl.DataFrame)
        assert len(ob_eager) == 4
        assert ob_eager.levels == 1

    def test_collect_from_eager_returns_self(self, sample_orderbook_df):
        """Test that collect() on eager OrderBook returns self."""
        ob_eager = OrderBook(data=sample_orderbook_df)
        assert not ob_eager.is_lazy

        ob_same = ob_eager.collect()
        assert ob_same is ob_eager

    def test_lazy_from_eager(self, sample_orderbook_df):
        """Test converting eager OrderBook to lazy via lazy()."""
        ob_eager = OrderBook(data=sample_orderbook_df)
        assert not ob_eager.is_lazy

        ob_lazy = ob_eager.lazy()
        assert ob_lazy.is_lazy
        assert isinstance(ob_lazy.df, pl.LazyFrame)
        assert len(ob_lazy) == 4
        assert ob_lazy.levels == 1

    def test_lazy_from_lazy_returns_self(self, sample_orderbook_df):
        """Test that lazy() on lazy OrderBook returns self."""
        ob_lazy = OrderBook(data=sample_orderbook_df.lazy())
        assert ob_lazy.is_lazy

        ob_same = ob_lazy.lazy()
        assert ob_same is ob_lazy

    def test_lazy_eager_roundtrip(self, sample_orderbook_df):
        """Test converting eager -> lazy -> eager preserves data."""
        ob_original = OrderBook(data=sample_orderbook_df)

        ob_lazy = ob_original.lazy()
        ob_back = ob_lazy.collect()

        assert ob_back.df.equals(ob_original.df)
        assert ob_back.levels == ob_original.levels

    @pytest.mark.parametrize("lazy", [False, True])
    def test_get_mid_prices(self, sample_orderbook_df, lazy):
        """Test computing mid prices with DataFrame and LazyFrame."""
        data = sample_orderbook_df.lazy() if lazy else sample_orderbook_df
        ob = OrderBook(data=data)

        mid_prices = ob.get_mid_prices()

        if lazy:
            assert isinstance(mid_prices, pl.LazyFrame)
            mid_prices = mid_prices.collect()

        assert "timestamp" in mid_prices.collect_schema().names()
        assert "mid_price" in mid_prices.collect_schema().names()
        assert mid_prices.height == 4

    @pytest.mark.parametrize("lazy", [False, True])
    def test_get_spreads(self, sample_orderbook_df, lazy):
        """Test computing spreads with DataFrame and LazyFrame."""
        data = sample_orderbook_df.lazy() if lazy else sample_orderbook_df
        ob = OrderBook(data=data)

        spreads = ob.get_spreads()

        if lazy:
            assert isinstance(spreads, pl.LazyFrame)
            spreads = spreads.collect()

        assert "timestamp" in spreads.collect_schema().names()
        assert "spread" in spreads.collect_schema().names()
        assert spreads.height == 4
