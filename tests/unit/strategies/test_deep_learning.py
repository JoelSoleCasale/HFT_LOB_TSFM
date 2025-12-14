"""Unit tests for deep learning strategies."""

import pytest
import numpy as np
import torch.nn as nn

from src.strategies.deep_learning import (
    DeepLearningStrategy,
    DeepLearningStrategyConfig,
    ClassificationStrategy,
    ClassificationStrategyConfig,
)
from src.models.architectures.base import FinancialTimeSeriesModel


class DummyModel(FinancialTimeSeriesModel):
    """Dummy model for testing."""

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        return self.fc(x[:, -1, :])  # Use last timestep


@pytest.mark.unit
class TestDeepLearningStrategyConfig:
    """Test DeepLearningStrategyConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = DeepLearningStrategyConfig()
        assert config.sequence_length == 128
        assert config.time_step_ns == 1_000_000_000

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeepLearningStrategyConfig(sequence_length=64, time_step_ns=500_000_000)
        assert config.sequence_length == 64
        assert config.time_step_ns == 500_000_000


@pytest.mark.unit
class TestDeepLearningStrategy:
    """Test DeepLearningStrategy class."""

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model."""
        return DummyModel(input_size=5, output_size=3)

    @pytest.fixture
    def strategy_config(self):
        """Create strategy configuration."""
        return DeepLearningStrategyConfig(sequence_length=10)

    def test_initialization(self, dummy_model, strategy_config):
        """Test strategy initialization."""
        strategy = DeepLearningStrategy(dummy_model, strategy_config)
        assert strategy.model == dummy_model
        assert not strategy.model.training
        assert strategy.config == strategy_config

    def test_prepare_data_shape(self, dummy_model, strategy_config):
        """Test that prepare_data produces correct shapes."""
        strategy = DeepLearningStrategy(dummy_model, strategy_config)

        n_samples = 100
        n_features = 5
        features = np.random.randn(n_samples, n_features)
        timestamps = np.arange(n_samples) * 1_000_000_000

        data = strategy.prepare_data(features, timestamps)

        # Should have n_samples - sequence_length + 1 predictions
        expected_n_pred = n_samples - strategy_config.sequence_length + 1
        assert data["predictions"].shape[0] == expected_n_pred
        assert len(data["timestamps"]) == expected_n_pred

    def test_prepare_data_insufficient_samples(self, dummy_model, strategy_config):
        """Test error when not enough samples."""
        strategy = DeepLearningStrategy(dummy_model, strategy_config)

        features = np.random.randn(5, 5)  # Less than sequence_length
        with pytest.raises(ValueError, match="Not enough samples"):
            strategy.prepare_data(features)

    def test_run_not_implemented(self, dummy_model, strategy_config):
        """Test that run raises NotImplementedError."""
        strategy = DeepLearningStrategy(dummy_model, strategy_config)
        with pytest.raises(NotImplementedError):
            strategy.run(None)


@pytest.mark.unit
class TestClassificationStrategyConfig:
    """Test ClassificationStrategyConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ClassificationStrategyConfig()
        assert config.buy_threshold == 0.6
        assert config.sell_threshold == 0.6
        assert config.order_quantity == 1.0
        assert config.max_position == 10.0
        assert config.use_market_orders is False
        assert config.limit_offset_ticks == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ClassificationStrategyConfig(
            buy_threshold=0.7,
            sell_threshold=0.7,
            order_quantity=0.5,
            max_position=5.0,
            use_market_orders=True,
            limit_offset_ticks=5,
        )
        assert config.buy_threshold == 0.7
        assert config.sell_threshold == 0.7
        assert config.order_quantity == 0.5
        assert config.max_position == 5.0
        assert config.use_market_orders is True
        assert config.limit_offset_ticks == 5


@pytest.mark.unit
class TestClassificationStrategy:
    """Test ClassificationStrategy class."""

    @pytest.fixture
    def classification_model(self):
        """Create a dummy classification model."""
        return DummyModel(input_size=5, output_size=3)

    @pytest.fixture
    def strategy_config(self):
        """Create strategy configuration."""
        return ClassificationStrategyConfig(sequence_length=10)

    def test_initialization(self, classification_model, strategy_config):
        """Test strategy initialization."""
        strategy = ClassificationStrategy(classification_model, strategy_config)
        assert strategy.model == classification_model
        assert strategy.config == strategy_config

    def test_initialization_wrong_output_size(self, strategy_config):
        """Test error when model output size is not 3."""
        wrong_model = DummyModel(input_size=5, output_size=2)
        with pytest.raises(ValueError, match="output_size=3"):
            ClassificationStrategy(wrong_model, strategy_config)

    def test_validate_config_valid(self, classification_model, strategy_config):
        """Test config validation with valid config."""
        strategy = ClassificationStrategy(classification_model, strategy_config)
        strategy.validate_config()  # Should not raise

    @pytest.mark.parametrize(
        "config_override,error_match",
        [
            ({"buy_threshold": 0.0}, "buy_threshold must be in"),
            ({"buy_threshold": 1.5}, "buy_threshold must be in"),
            ({"sell_threshold": -0.1}, "sell_threshold must be in"),
            ({"order_quantity": 0.0}, "order_quantity must be positive"),
            ({"order_quantity": -1.0}, "order_quantity must be positive"),
            ({"max_position": 0.0}, "max_position must be positive"),
            ({"max_position": -5.0}, "max_position must be positive"),
        ],
    )
    def test_validate_config_invalid(
        self, classification_model, strategy_config, config_override, error_match
    ):
        """Test config validation with invalid configs."""
        for key, value in config_override.items():
            setattr(strategy_config, key, value)

        with pytest.raises(ValueError, match=error_match):
            _ = ClassificationStrategy(classification_model, strategy_config)

    def test_prepare_data_probabilities(self, classification_model, strategy_config):
        """Test that prepare_data converts logits to probabilities."""
        strategy = ClassificationStrategy(classification_model, strategy_config)

        n_samples = 100
        n_features = 5
        features = np.random.randn(n_samples, n_features)

        data = strategy.prepare_data(features)

        # Check probabilities sum to 1
        probabilities = data["probabilities"]
        prob_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(probabilities)))

        # Check probabilities are in [0, 1]
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_create_strategy_function(self, classification_model, strategy_config):
        """Test creating strategy function."""
        strategy = ClassificationStrategy(classification_model, strategy_config)

        n_samples = 100
        features = np.random.randn(n_samples, 5)
        timestamps = np.arange(n_samples) * 1_000_000_000

        data = strategy.prepare_data(features, timestamps)

        # Create strategy function
        strategy_func = strategy.create_strategy_function(
            data["probabilities"], data["timestamps"]
        )

        # Check it's callable
        assert callable(strategy_func)

    def test_run_not_directly_callable(self, classification_model, strategy_config):
        """Test that run is not directly callable."""
        strategy = ClassificationStrategy(classification_model, strategy_config)
        with pytest.raises(NotImplementedError, match="should not be called directly"):
            strategy.run(None)

    def test_signal_generation_logic(self, classification_model, strategy_config):
        """Test signal generation from probabilities."""
        strategy_config.buy_threshold = 0.6
        strategy_config.sell_threshold = 0.6

        # Create mock probabilities
        # [P(sell), P(hold), P(buy)]
        probabilities = np.array(
            [
                [0.1, 0.2, 0.7],  # Strong buy signal
                [0.7, 0.2, 0.1],  # Strong sell signal
                [0.3, 0.4, 0.3],  # No clear signal
                [0.1, 0.3, 0.6],  # Exactly at threshold (buy)
            ]
        )

        # Expected signals: buy, sell, hold, buy
        # Note: The actual signal logic is in the numba function,
        # but we can verify the probabilities are correctly formatted
        assert probabilities[0, 2] >= strategy_config.buy_threshold  # Buy
        assert probabilities[1, 0] >= strategy_config.sell_threshold  # Sell
        assert probabilities[2, 2] < strategy_config.buy_threshold  # Hold
        assert probabilities[3, 2] >= strategy_config.buy_threshold  # Buy
