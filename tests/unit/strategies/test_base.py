"""Unit tests for base strategy classes."""

import pytest
import numpy as np
from src.strategies.base import BaseStrategy, StrategyConfig


class DummyStrategy(BaseStrategy):
    """Dummy strategy for testing."""

    def run(self, hbt):
        return True


@pytest.mark.unit
class TestStrategyConfig:
    """Test StrategyConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = StrategyConfig()
        assert config.asset_no == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = StrategyConfig(asset_no=1)
        assert config.asset_no == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = StrategyConfig(asset_no=2)
        config_dict = config.to_dict()
        assert config_dict["asset_no"] == 2


@pytest.mark.unit
class TestBaseStrategy:
    """Test BaseStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""
        config = StrategyConfig()
        strategy = DummyStrategy(config)
        assert strategy.config == config

    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        config = StrategyConfig(asset_no=0)
        strategy = DummyStrategy(config)
        strategy.validate_config()  # Should not raise

    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        config = StrategyConfig(asset_no=-1)
        strategy = DummyStrategy(config)
        with pytest.raises(ValueError, match="asset_no must be non-negative"):
            strategy.validate_config()

    def test_prepare_data(self):
        """Test data preparation."""
        config = StrategyConfig()
        strategy = DummyStrategy(config)

        features = np.random.randn(100, 10)
        data = strategy.prepare_data(features)

        assert "features" in data
        assert np.array_equal(data["features"], features)

    def test_run_method_exists(self):
        """Test that run method is implemented."""
        config = StrategyConfig()
        strategy = DummyStrategy(config)
        assert callable(strategy.run)
