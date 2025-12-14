"""
Base strategy class for all trading strategies.

Strategies define how to interact with hftbacktest to execute trades
based on market data and signals (e.g., from models).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np

from hftbacktest import HashMapMarketDepthBacktest_TypeHint


@dataclass
class StrategyConfig:
    """Base configuration for strategies."""

    asset_no: int = 0  # Asset number in the backtest (0-indexed)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies compatible with hftbacktest.

    A strategy encapsulates the logic for generating trading signals and
    executing orders through the hftbacktest backtester.

    The strategy should implement the `run` method which contains the main
    trading loop that will be executed by hftbacktest.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config

    @abstractmethod
    def run(self, hbt: HashMapMarketDepthBacktest_TypeHint) -> bool:
        """
        Main strategy execution loop.

        This method should be decorated with @njit for performance and
        contain the core trading logic that interacts with hftbacktest.

        Args:
            hbt: hftbacktest backtester instance

        Returns:
            bool: True if strategy completed successfully, False otherwise

        Note:
            This method will typically:
            1. Loop through time using hbt.elapse()
            2. Access market data via hbt.depth()
            3. Generate trading signals
            4. Submit orders via hbt.submit_buy_order() or hbt.submit_sell_order()
            5. Manage positions and risk
        """
        raise NotImplementedError

    def prepare_data(self, features: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        """
        Prepare any data needed for the strategy execution.

        This method is called before running the strategy and can be used
        to preprocess features, load model weights, etc.

        Args:
            features: Feature array
            **kwargs: Additional data

        Returns:
            dict: Dictionary containing prepared data
        """
        return {"features": features}

    def validate_config(self) -> None:
        """
        Validate the strategy configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.asset_no < 0:
            raise ValueError("asset_no must be non-negative")
