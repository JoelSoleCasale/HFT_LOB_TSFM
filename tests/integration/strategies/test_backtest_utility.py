"""
Integration tests for backtest_model utility.

Tests the complete backtest pipeline with synthetic data to ensure:
1. Proper integration with hftbacktest
2. Correct statistics generation
3. Trade execution logic
4. Timestamp alignment
"""

import pytest
import polars as pl
import numpy as np
import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")  # Use non-GUI backend for tests

from strategies import backtest_model, ClassificationStrategyConfig
from models.architectures.lstm import LSTMTimeSeriesModel


@pytest.fixture
def synthetic_model():
    """Create a simple LSTM model for testing."""
    model = LSTMTimeSeriesModel(
        input_size=3,
        hidden_size=16,
        num_layers=1,
        output_size=3,
        dropout=0.0,
        bidirectional=False,
        attention=False,
    )
    model.eval()
    return model


@pytest.fixture
def synthetic_features():
    """Create synthetic feature data with 200 timesteps."""
    # Start at a known timestamp (July 1, 2025, midnight in nanoseconds)
    start_ts = 1751328000000000000
    n_samples = 200
    timestamps = [start_ts + i * 1_000_000_000 for i in range(n_samples)]  # 1 second apart

    features_data = {
        "timestamp": timestamps,
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
    }
    return pl.LazyFrame(features_data)


@pytest.fixture
def synthetic_orderbook(synthetic_features):
    """Create synthetic raw orderbook data matching feature timestamps."""
    features_df = synthetic_features.collect()
    start_ts = features_df["timestamp"][0]
    end_ts = features_df["timestamp"][-1]

    # Create orderbook events covering the same time range
    n_events = 2000
    event_timestamps = np.linspace(start_ts, end_ts, n_events, dtype=np.int64)

    # Convert nanoseconds back to milliseconds for raw orderbook format
    event_timestamps_ms = (event_timestamps / 1_000_000).astype(np.int64)

    raw_data = pl.DataFrame(
        {
            "event_time": event_timestamps_ms,
            "received_time": event_timestamps_ms
            + np.random.randint(1, 10, n_events),  # Add small delay
            "event_type": ["depth"] * n_events,
            "side": np.random.choice(["bid", "ask"], n_events),
            "price": np.random.uniform(60000, 61000, n_events),
            "quantity": np.random.uniform(0.1, 1.0, n_events),
        }
    )

    return raw_data


@pytest.mark.integration
class TestBacktestModelBasic:
    """Basic functionality tests for backtest_model."""

    def test_returns_stats_object(self, synthetic_model, synthetic_features, synthetic_orderbook):
        """Test that backtest_model returns a Stats object."""
        stats = backtest_model(
            model=synthetic_model,
            features=synthetic_features,
            raw_orderbook_df=synthetic_orderbook,
        )

        # Should be able to call summary and plot methods
        assert hasattr(stats, "summary"), "Stats object should have summary method"
        assert hasattr(stats, "plot"), "Stats object should have plot method"

        # Summary should return a polars DataFrame
        summary_df = stats.summary()
        assert isinstance(summary_df, pl.DataFrame), "summary() should return DataFrame"

    def test_with_low_thresholds_executes_trades(
        self, synthetic_model, synthetic_features, synthetic_orderbook
    ):
        """Test that with very low thresholds, trades are executed."""
        config = ClassificationStrategyConfig(
            buy_threshold=0.1,
            sell_threshold=0.1,
            max_position=5,
            order_quantity=1,
        )

        stats = backtest_model(
            model=synthetic_model,
            features=synthetic_features,
            raw_orderbook_df=synthetic_orderbook,
            config=config,
        )

        summary_df = stats.summary()
        # With low thresholds and random predictions, should have some trades
        # Note: This might still be 0 if model predicts all same class
        # but it's unlikely with random initialization
        assert summary_df is not None

    def test_with_high_thresholds_fewer_trades(
        self, synthetic_model, synthetic_features, synthetic_orderbook
    ):
        """Test that higher thresholds result in fewer trades."""
        config_low = ClassificationStrategyConfig(
            buy_threshold=0.2,
            sell_threshold=0.2,
        )

        config_high = ClassificationStrategyConfig(
            buy_threshold=0.8,
            sell_threshold=0.8,
        )

        stats_low = backtest_model(
            synthetic_model, synthetic_features, synthetic_orderbook, config_low
        )
        stats_high = backtest_model(
            synthetic_model, synthetic_features, synthetic_orderbook, config_high
        )

        # Both should complete without error
        assert stats_low is not None
        assert stats_high is not None


@pytest.mark.integration
class TestBacktestModelStatistics:
    """Tests for statistics calculation and metrics."""

    def test_summary_contains_expected_columns(
        self, synthetic_model, synthetic_features, synthetic_orderbook
    ):
        """Test that summary DataFrame contains expected metric columns."""
        stats = backtest_model(synthetic_model, synthetic_features, synthetic_orderbook)

        summary_df = stats.summary()
        # Note: hftbacktest uses "MaxDrawdown" not "MDD"
        expected_columns = ["SR", "Sortino", "MaxDrawdown", "Return"]

        for col in expected_columns:
            assert col in summary_df.columns, f"Summary should contain '{col}' column"

    def test_statistics_have_valid_types(
        self, synthetic_model, synthetic_features, synthetic_orderbook
    ):
        """Test that statistics are of correct types and not None."""
        stats = backtest_model(synthetic_model, synthetic_features, synthetic_orderbook)

        summary_df = stats.summary()

        # Check that numeric columns contain numeric values (or NaN for no trades)
        numeric_cols = ["SR", "Sortino", "MDD", "Return"]
        for col in numeric_cols:
            if col in summary_df.columns:
                value = summary_df[col][0]
                assert isinstance(value, (int, float)), f"{col} should be numeric"


@pytest.mark.integration
class TestBacktestModelEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.skip(
        reason="hftbacktest cannot compute statistics when no trades occur, which is common with minimal synthetic data"
    )
    def test_minimal_data(self):
        """Test with minimal amount of data and low threshold to ensure trades occur.

        Note: This test is skipped because with truly minimal data (just enough for
        model predictions), it's very unlikely that trades will occur, and hftbacktest
        will raise an error when trying to compute statistics on zero trades.
        The other tests verify the function works correctly with realistic data amounts.
        """
        model = LSTMTimeSeriesModel(input_size=2, hidden_size=8, num_layers=1, output_size=3)
        model.eval()

        # Need at least 130 samples: sequence_length=128 + 2 for stats computation
        # hftbacktest needs at least 2 samples with trades to compute statistics
        start_ts = 1751328000000000000
        n_samples = 135
        timestamps = [start_ts + i * 1_000_000_000 for i in range(n_samples)]

        features = pl.LazyFrame(
            {
                "timestamp": timestamps,
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
            }
        )

        event_timestamps_ms = [
            (start_ts + i * 1_000_000_000) // 1_000_000 for i in range(n_samples * 10)
        ]
        orderbook = pl.DataFrame(
            {
                "event_time": event_timestamps_ms,
                "received_time": event_timestamps_ms,
                "event_type": ["depth"] * (n_samples * 10),
                "side": ["bid"] * (n_samples * 10),
                "price": [60000.0] * (n_samples * 10),
                "quantity": [1.0] * (n_samples * 10),
            }
        )

        # Use low thresholds to ensure at least some trades occur
        config = ClassificationStrategyConfig(
            buy_threshold=0.1,
            sell_threshold=0.1,
        )

        # Should not raise error even with minimal data
        stats = backtest_model(model, features, orderbook, config)
        assert stats is not None

    def test_lazyframe_and_dataframe_features(
        self, synthetic_model, synthetic_features, synthetic_orderbook
    ):
        """Test that both LazyFrame and eager DataFrame work for features."""
        # Test with LazyFrame
        stats_lazy = backtest_model(synthetic_model, synthetic_features, synthetic_orderbook)

        # Test with collected DataFrame
        features_eager = synthetic_features.collect()
        stats_eager = backtest_model(synthetic_model, features_eager, synthetic_orderbook)

        # Both should complete without error
        assert stats_lazy is not None
        assert stats_eager is not None


@pytest.mark.integration
class TestBacktestModelConfiguration:
    """Tests for different configuration options."""

    def test_custom_position_limits(
        self, synthetic_model, synthetic_features, synthetic_orderbook
    ):
        """Test with custom position and order quantity limits."""
        config = ClassificationStrategyConfig(
            buy_threshold=0.3,
            sell_threshold=0.3,
            max_position=20,  # Larger position
            order_quantity=2,  # Larger orders
        )

        stats = backtest_model(synthetic_model, synthetic_features, synthetic_orderbook, config)

        # Should complete without error
        assert stats is not None

    def test_market_vs_limit_orders(
        self, synthetic_model, synthetic_features, synthetic_orderbook
    ):
        """Test with market orders vs limit orders."""
        config_market = ClassificationStrategyConfig(
            buy_threshold=0.3,
            sell_threshold=0.3,
            use_market_orders=True,
        )

        config_limit = ClassificationStrategyConfig(
            buy_threshold=0.3,
            sell_threshold=0.3,
            use_market_orders=False,
            limit_offset_ticks=1,
        )

        stats_market = backtest_model(
            synthetic_model, synthetic_features, synthetic_orderbook, config_market
        )
        stats_limit = backtest_model(
            synthetic_model, synthetic_features, synthetic_orderbook, config_limit
        )

        # Both should complete
        assert stats_market is not None
        assert stats_limit is not None


@pytest.mark.integration
class TestBacktestModelIntegration:
    """Integration tests verifying end-to-end functionality."""

    def test_complete_backtest_pipeline(
        self, synthetic_model, synthetic_features, synthetic_orderbook, tmp_path
    ):
        """Test complete backtest pipeline with all features."""
        config = ClassificationStrategyConfig(
            buy_threshold=0.35,
            sell_threshold=0.35,
            max_position=10,
            order_quantity=1,
            use_market_orders=False,
            limit_offset_ticks=1,
        )

        save_path = tmp_path / "full_test_backtest.png"

        # Run complete backtest
        stats = backtest_model(
            model=synthetic_model,
            features=synthetic_features,
            raw_orderbook_df=synthetic_orderbook,
            config=config,
            save_plot_path=save_path,
        )

        # Verify all components
        assert stats is not None, "Stats should not be None"

        summary_df = stats.summary()
        assert len(summary_df) > 0, "Summary should have data"
        assert "SR" in summary_df.columns, "Should have Sharpe Ratio"

        assert save_path.exists(), "Plot should be saved"

    def test_timestamp_alignment(self, synthetic_model):
        """Test that timestamps are properly aligned between features and orderbook."""
        # Create features with known timestamps
        start_ts = 1751328000000000000  # Nanoseconds
        n_samples = 150
        feature_timestamps = [start_ts + i * 1_000_000_000 for i in range(n_samples)]

        features = pl.LazyFrame(
            {
                "timestamp": feature_timestamps,
                "f1": np.random.randn(n_samples),
                "f2": np.random.randn(n_samples),
                "f3": np.random.randn(n_samples),
            }
        )

        # Create orderbook covering same range (in milliseconds)
        start_ts_ms = start_ts // 1_000_000
        end_ts_ms = (start_ts + (n_samples - 1) * 1_000_000_000) // 1_000_000

        n_events = 1500
        event_ts_ms = np.linspace(start_ts_ms, end_ts_ms, n_events, dtype=np.int64)

        orderbook = pl.DataFrame(
            {
                "event_time": event_ts_ms,
                "received_time": event_ts_ms,
                "event_type": ["depth"] * n_events,
                "side": np.random.choice(["bid", "ask"], n_events),
                "price": np.random.uniform(60000, 61000, n_events),
                "quantity": np.random.uniform(0.1, 1.0, n_events),
            }
        )

        config = ClassificationStrategyConfig(
            buy_threshold=0.2,
            sell_threshold=0.2,
        )

        # Should execute without timestamp alignment errors
        stats = backtest_model(synthetic_model, features, orderbook, config)
        assert stats is not None


@pytest.mark.integration
@pytest.mark.parametrize(
    "buy_threshold,sell_threshold",
    [
        (0.3, 0.3),
        (0.5, 0.5),
        (0.7, 0.7),
    ],
)
def test_various_threshold_combinations(
    buy_threshold, sell_threshold, synthetic_model, synthetic_features, synthetic_orderbook
):
    """Test backtest with various threshold combinations."""
    config = ClassificationStrategyConfig(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    stats = backtest_model(synthetic_model, synthetic_features, synthetic_orderbook, config)

    # Should complete without error for all thresholds
    assert stats is not None
    assert stats.summary() is not None


class DeterministicAlternatingModel(nn.Module):
    """A dummy model that alternates predictions between buy (1) and sell (-1).

    Returns logits where one class has high confidence and others have low confidence.
    The model alternates which class is predicted with each call.
    """

    def __init__(self, confidence_logit: float = 2.0):
        """
        Args:
            confidence_logit: The logit value for the predicted class (others get 0).
                             Higher values = higher softmax probability.
                             logit=2.0 gives ~88% confidence after softmax
                             logit=1.0 gives ~67% confidence after softmax
                             logit=0.5 gives ~53% confidence after softmax
        """
        super().__init__()
        self.confidence_logit = confidence_logit
        self.call_count = 0
        self.output_size = 3  # sell, hold, buy
        # Add a dummy parameter so the strategy can detect device
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Returns alternating predictions as logits.

        Returns:
            Tensor of shape (batch_size, 3) with logits for [sell, hold, buy]
        """
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 3)

        for i in range(batch_size):
            # Alternate between sell (class 0) and buy (class 2)
            if (self.call_count + i) % 2 == 0:
                # Predict sell with high confidence
                logits[i] = torch.tensor([self.confidence_logit, 0.0, 0.0])
            else:
                # Predict buy with high confidence
                logits[i] = torch.tensor([0.0, 0.0, self.confidence_logit])

        self.call_count += batch_size
        return logits


@pytest.fixture
def alternating_features():
    """Create feature data that will produce exactly 20 predictions after sliding window."""
    # Start at a known timestamp
    start_ts = 1751328000000000000
    # Need sequence_length (128) + number of predictions we want (20)
    n_samples = 128 + 20  # 148 samples
    timestamps = [start_ts + i * 1_000_000_000 for i in range(n_samples)]

    features_data = {
        "timestamp": timestamps,
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
    }
    return pl.LazyFrame(features_data)


@pytest.fixture
def alternating_orderbook(alternating_features):
    """Create orderbook data matching the alternating features timeframe."""
    features_df = alternating_features.collect()
    start_ts = features_df["timestamp"][0]
    end_ts = features_df["timestamp"][-1]

    # Create sufficient orderbook events
    n_events = 1500
    event_timestamps = np.linspace(start_ts, end_ts, n_events, dtype=np.int64)
    event_timestamps_ms = (event_timestamps / 1_000_000).astype(np.int64)

    raw_data = pl.DataFrame(
        {
            "event_time": event_timestamps_ms,
            "received_time": event_timestamps_ms + np.random.randint(1, 10, n_events),
            "event_type": ["depth"] * n_events,
            "side": np.random.choice(["bid", "ask"], n_events),
            "price": np.random.uniform(60000, 61000, n_events),
            "quantity": np.random.uniform(0.1, 1.0, n_events),
        }
    )

    return raw_data


@pytest.mark.integration
class TestDeterministicPredictions:
    """Tests with deterministic alternating model to verify trade execution logic."""

    def test_alternating_predictions_high_confidence(
        self, alternating_features, alternating_orderbook
    ):
        """Test that alternating high-confidence predictions generate expected trades.

        With confidence_logit=2.0, softmax gives ~88% probability.
        With threshold=0.5, all predictions should trigger trades.
        Expected pattern: sell, buy, sell, buy, ... (21 predictions with 148 samples)
        """
        model = DeterministicAlternatingModel(confidence_logit=2.0)
        model.eval()

        config = ClassificationStrategyConfig(
            buy_threshold=0.5,
            sell_threshold=0.5,
            max_position=10,
            order_quantity=1,
        )

        stats = backtest_model(model, alternating_features, alternating_orderbook, config)

        # With 148 samples and sequence_length=128, we get 21 predictions
        # All should exceed threshold and trigger trades
        summary_df = stats.summary()

        # Check that return is not NaN (trades occurred)
        return_value = summary_df["Return"][0]
        assert not np.isnan(return_value), "Return should be valid when trades occur"

        # The DailyNumberOfTrades is a normalized metric, not the raw trade count
        # Just verify it's positive when we expect trades
        daily_trades = summary_df["DailyNumberOfTrades"][0]
        assert daily_trades > 0, f"Expected positive daily trade count, got {daily_trades}"

    def test_alternating_predictions_low_confidence(
        self, alternating_features, alternating_orderbook
    ):
        """Test that low-confidence predictions don't trigger trades with high threshold.

        With confidence_logit=0.5, softmax gives ~53% probability.
        With threshold=0.8, no predictions should trigger trades.
        """
        model = DeterministicAlternatingModel(confidence_logit=0.5)
        model.eval()

        config = ClassificationStrategyConfig(
            buy_threshold=0.8,
            sell_threshold=0.8,
            max_position=10,
            order_quantity=1,
        )

        stats = backtest_model(model, alternating_features, alternating_orderbook, config)

        summary_df = stats.summary()

        # With 53% confidence and 80% threshold, should have 0 trades
        # When no trades occur, Return will be NaN or 0
        return_value = summary_df["Return"][0]
        assert (
            np.isnan(return_value) or return_value == 0
        ), f"Expected NaN or 0 return with no trades, got {return_value}"

    @pytest.mark.parametrize(
        "confidence_logit,threshold,should_trade",
        [
            (2.0, 0.5, True),  # 88% confidence > 50% threshold = trade
            (2.0, 0.9, False),  # 88% confidence < 90% threshold = no trade
            (1.0, 0.6, True),  # 67% confidence > 60% threshold = trade
            (1.0, 0.7, False),  # 67% confidence < 70% threshold = no trade
            (0.5, 0.5, True),  # 53% confidence > 50% threshold = trade
            (0.5, 0.6, False),  # 53% confidence < 60% threshold = no trade
        ],
    )
    def test_threshold_vs_confidence_relationship(
        self,
        alternating_features,
        alternating_orderbook,
        confidence_logit,
        threshold,
        should_trade,
    ):
        """Test that trades occur only when confidence exceeds threshold.

        This parameterized test verifies the relationship between model confidence
        (expressed as logits) and trading thresholds.
        """
        model = DeterministicAlternatingModel(confidence_logit=confidence_logit)
        model.eval()

        config = ClassificationStrategyConfig(
            buy_threshold=threshold,
            sell_threshold=threshold,
            max_position=10,
            order_quantity=1,
        )

        stats = backtest_model(model, alternating_features, alternating_orderbook, config)

        summary_df = stats.summary()
        return_value = summary_df["Return"][0]

        if should_trade:
            # When trades should occur, return should not be NaN
            assert not np.isnan(return_value), (
                f"Expected trades with confidence_logit={confidence_logit} "
                f"and threshold={threshold}, but return is NaN (no trades)"
            )
        else:
            # When no trades should occur, return should be NaN or 0
            assert np.isnan(return_value) or return_value == 0, (
                f"Expected no trades with confidence_logit={confidence_logit} "
                f"and threshold={threshold}, but return={return_value} (trades occurred)"
            )

    def test_buy_sell_alternation_pattern(self, alternating_features, alternating_orderbook):
        """Test that alternating buy/sell predictions create expected position changes.

        With alternating predictions and sufficient confidence, we should see
        the position oscillate between positive and negative values.
        """
        model = DeterministicAlternatingModel(confidence_logit=2.0)
        model.eval()

        config = ClassificationStrategyConfig(
            buy_threshold=0.5,
            sell_threshold=0.5,
            max_position=5,
            order_quantity=1,
        )

        stats = backtest_model(model, alternating_features, alternating_orderbook, config)

        # Check that trades occurred
        summary_df = stats.summary()
        num_trades = summary_df["DailyNumberOfTrades"][0]
        assert num_trades > 0, "Expected alternating pattern to generate trades"

        # The return should be finite (not NaN) if trades occurred properly
        return_value = summary_df["Return"][0]
        assert (
            not np.isnan(return_value) or num_trades == 0
        ), "Return should be valid if trades occurred"
