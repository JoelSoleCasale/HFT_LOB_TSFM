"""Tests for triple barrier method label extractors"""

import pytest
import polars as pl
from features.labels.barrier_labels import TripleBarrierLabel
from features.base.input_space import InputSpace
from core.orderbook import OrderBook


# Helper functions
def create_orderbook_from_prices(prices: list[float], spread: float = 0.5) -> OrderBook:
    """Create an OrderBook from a list of mid prices with specified spread.

    Args:
        prices: List of mid prices
        spread: Half-spread around mid price (ask = mid + spread, bid = mid - spread)

    Returns:
        OrderBook instance
    """
    timestamps = list(range(len(prices)))
    data = {
        "timestamp": timestamps,
        "ask1_price": [p + spread for p in prices],
        "ask1_qty": [1.0] * len(prices),
        "bid1_price": [p - spread for p in prices],
        "bid1_qty": [1.0] * len(prices),
    }
    return OrderBook(data=pl.DataFrame(data))


def create_input_space_from_prices(prices: list[float], spread: float = 0.5) -> InputSpace:
    """Create an InputSpace from a list of mid prices.

    Args:
        prices: List of mid prices
        spread: Half-spread around mid price

    Returns:
        InputSpace instance with orderbook_snapshots
    """
    orderbook = create_orderbook_from_prices(prices, spread)
    return InputSpace(orderbook_snapshots=orderbook)


def extract_labels(input_space: InputSpace, threshold: float, horizon: int) -> pl.DataFrame:
    """Extract barrier labels from input space.

    Args:
        input_space: InputSpace containing orderbook data
        threshold: Barrier threshold (e.g., 0.02 for 2%)
        horizon: Time horizon for barriers

    Returns:
        Collected DataFrame with labels
    """
    extractor = TripleBarrierLabel({"threshold": threshold, "horizon": horizon})
    return extractor.extract(input_space).collect()


def count_labels(labels: list[int]) -> tuple[int, int, int]:
    """Count occurrences of each label value.

    Args:
        labels: List of label values

    Returns:
        Tuple of (negative_count, neutral_count, positive_count)
    """
    negative_count = sum(1 for label in labels if label == -1)
    neutral_count = sum(1 for label in labels if label == 0)
    positive_count = sum(1 for label in labels if label == 1)
    return negative_count, neutral_count, positive_count


@pytest.fixture
def simple_price_sequence():
    """Create a simple price sequence for testing barrier logic"""
    # Prices: 100, 102, 104, 98, 96, 100, 101, 103, 105, 107
    prices = [100.0, 102.0, 104.0, 98.0, 96.0, 100.0, 101.0, 103.0, 105.0, 107.0]
    return create_orderbook_from_prices(prices).df


@pytest.fixture
def uptrend_sequence():
    """Create an upward trending price sequence"""
    # Start at 100, increase by 1% each step
    prices = [100 * (1.01**i) for i in range(20)]
    return create_orderbook_from_prices(prices).df


@pytest.fixture
def downtrend_sequence():
    """Create a downward trending price sequence"""
    # Start at 100, decrease by 1% each step
    prices = [100 * (0.99**i) for i in range(20)]
    return create_orderbook_from_prices(prices).df


@pytest.fixture
def sideways_sequence():
    """Create a sideways/ranging price sequence"""
    # Oscillate between 99 and 101
    prices = [100 + (i % 2) for i in range(20)]
    return create_orderbook_from_prices(prices).df


class TestTripleBarrierLabel:
    """Tests for TripleBarrierLabel"""

    def test_initialization(self):
        """Test basic initialization"""
        extractor = TripleBarrierLabel({"threshold": 0.02, "horizon": 5})
        assert extractor.threshold == 0.02
        assert extractor.horizon == 5
        assert "orderbook_snapshots" in extractor.dependencies

    def test_default_config(self):
        """Test default configuration values"""
        extractor = TripleBarrierLabel()
        assert extractor.threshold == 0.01  # 1% default
        assert extractor.horizon == 10

    def test_upper_barrier_hit(self):
        """Test label when upper barrier (take profit) is hit first"""
        # Create a sequence where price goes up by 3% within 5 steps
        prices = [100.0, 100.5, 101.0, 102.0, 103.5, 104.0, 103.0, 102.0, 101.0, 100.0]
        input_space = create_input_space_from_prices(prices)

        # Use 2% threshold and horizon of 10
        result = extract_labels(input_space, threshold=0.02, horizon=10)

        # First entry at 100, should hit upper barrier at 103.5 (>102)
        assert result["triple_barrier_0.02_10"][0] == 1

    def test_lower_barrier_hit(self):
        """Test label when lower barrier (stop loss) is hit first"""
        # Create a sequence where price goes down by 3% within 5 steps
        prices = [100.0, 99.5, 99.0, 98.0, 96.5, 96.0, 97.0, 98.0, 99.0, 100.0]
        input_space = create_input_space_from_prices(prices)

        # Use 2% threshold and horizon of 10
        result = extract_labels(input_space, threshold=0.02, horizon=10)

        # First entry at 100, should hit lower barrier at 96.5 (<98)
        assert result["triple_barrier_0.02_10"][0] == -1

    def test_time_barrier_hit(self):
        """Test label when neither profit/loss barrier hit within horizon"""
        # Create a sequence with minimal movement
        prices = [100.0] * 10  # Flat price
        input_space = create_input_space_from_prices(prices)

        # Use 2% threshold (need 102 or 98) and horizon of 5
        result = extract_labels(input_space, threshold=0.02, horizon=5)

        # All entries should hit time barrier (label = 0)
        assert all(result["triple_barrier_0.02_5"] == 0)

    def test_uptrend_mostly_positive(self, uptrend_sequence):
        """Test that uptrending prices mostly generate positive labels"""
        orderbook = OrderBook(data=uptrend_sequence)
        input_space = InputSpace(orderbook_snapshots=orderbook)

        result = extract_labels(input_space, threshold=0.015, horizon=10)

        labels = result["triple_barrier_0.015_10"].to_list()
        negative_count, _, positive_count = count_labels(labels)

        assert positive_count > negative_count

    def test_downtrend_mostly_negative(self, downtrend_sequence):
        """Test that downtrending prices mostly generate negative labels"""
        orderbook = OrderBook(data=downtrend_sequence)
        input_space = InputSpace(orderbook_snapshots=orderbook)

        result = extract_labels(input_space, threshold=0.015, horizon=10)

        labels = result["triple_barrier_0.015_10"].to_list()
        negative_count, _, positive_count = count_labels(labels)

        assert negative_count > positive_count

    def test_sideways_mostly_neutral(self, sideways_sequence):
        """Test that sideways prices mostly generate neutral labels"""
        orderbook = OrderBook(data=sideways_sequence)
        input_space = InputSpace(orderbook_snapshots=orderbook)

        # Use larger threshold so small movements don't trigger
        result = extract_labels(input_space, threshold=0.05, horizon=5)

        labels = result["triple_barrier_0.05_5"].to_list()
        _, neutral_count, _ = count_labels(labels)

        # Most should be neutral
        assert neutral_count > len(labels) / 2

    def test_output_structure(self, simple_price_sequence):
        """Test that output has correct structure"""
        orderbook = OrderBook(data=simple_price_sequence)
        input_space = InputSpace(orderbook_snapshots=orderbook)

        result = extract_labels(input_space, threshold=0.02, horizon=5)

        # Check columns
        assert "timestamp" in result.columns
        assert "triple_barrier_0.02_5" in result.columns
        assert len(result.columns) == 2

        # Check label values are in {-1, 0, 1}
        labels = result["triple_barrier_0.02_5"].to_list()
        assert set(labels).issubset({-1, 0, 1})

    def test_horizon_effect(self, simple_price_sequence):
        """Test that different horizons produce different results"""
        orderbook = OrderBook(data=simple_price_sequence)
        input_space = InputSpace(orderbook_snapshots=orderbook)

        # Short horizon
        result_short = extract_labels(input_space, threshold=0.02, horizon=2)
        # Long horizon
        result_long = extract_labels(input_space, threshold=0.02, horizon=8)

        # Longer horizon should have fewer neutral labels (more time to hit barriers)
        _, neutral_short, _ = count_labels(result_short["triple_barrier_0.02_2"].to_list())
        _, neutral_long, _ = count_labels(result_long["triple_barrier_0.02_8"].to_list())

        # This relationship should generally hold
        assert neutral_short >= neutral_long


class TestBarrierEdgeCases:
    """Test edge cases for barrier labeling"""

    def test_empty_dataframe(self):
        """Test behavior with empty dataframe"""
        # Create empty dataframe with correct schema
        schema = {
            "timestamp": pl.Int64,
            "ask1_price": pl.Float64,
            "ask1_qty": pl.Float64,
            "bid1_price": pl.Float64,
            "bid1_qty": pl.Float64,
        }
        df = pl.DataFrame(schema=schema)
        orderbook = OrderBook(data=df)
        input_space = InputSpace(orderbook_snapshots=orderbook)

        extractor = TripleBarrierLabel({"threshold": 0.02, "horizon": 5})
        result = extractor.extract(input_space).collect()

        assert result.height == 0

    def test_single_row(self):
        """Test with single price point"""
        input_space = create_input_space_from_prices([100.0])
        result = extract_labels(input_space, threshold=0.02, horizon=5)

        # Should return neutral (no future prices to check)
        assert result["triple_barrier_0.02_5"][0] == 0

    def test_end_of_sequence_labels(self):
        """Test that labels near end of sequence are neutral (insufficient future data)"""
        prices = [100.0 + i for i in range(10)]
        input_space = create_input_space_from_prices(prices)

        result = extract_labels(input_space, threshold=0.02, horizon=5)

        # Last few entries should be neutral (not enough future data)
        labels = result["triple_barrier_0.02_5"].to_list()
        # Last entry definitely should be 0
        assert labels[-1] == 0

    def test_very_small_threshold(self):
        """Test with very small threshold (sensitive to small movements)"""
        # Small movements
        prices = [100.0, 100.05, 100.1, 100.15, 100.1, 100.05, 100.0, 99.95, 99.9, 99.95]
        input_space = create_input_space_from_prices(prices)

        # Very small threshold (0.1%)
        result = extract_labels(input_space, threshold=0.001, horizon=5)

        labels = result["triple_barrier_0.001_5"].to_list()
        # Should have both positive and negative labels
        assert 1 in labels or -1 in labels

    def test_simultaneous_barrier_touch(self):
        """Test when price could theoretically touch both barriers in same step"""
        # This tests the priority (whichever is seen first in sequence)
        # Sharp movements
        prices = [100.0, 105.0, 95.0, 105.0, 95.0, 100.0, 102.0, 98.0, 101.0, 99.0]
        input_space = create_input_space_from_prices(prices)

        result = extract_labels(input_space, threshold=0.03, horizon=5)

        labels = result["triple_barrier_0.03_5"].to_list()
        # First entry at 100 should see 105 first (upper barrier hit)
        assert labels[0] == 1
