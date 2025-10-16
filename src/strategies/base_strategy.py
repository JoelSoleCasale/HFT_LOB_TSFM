from abc import ABC, abstractmethod
from typing import Literal
from dataclasses import dataclass
from hftbacktest import HashMapMarketDepthBacktest_TypeHint
from hftbacktest import GTC, MARKET
import random
from dataclasses import field


@dataclass
class TradingSignal:
    """
    A dataclass representing a trading signal. Contains all necessary
    information to place an order.
    """

    action: Literal["BUY", "SELL"]
    price: float
    qty: float
    asset_no: int = 0  # default to 0 for single asset strategies
    order_id: int = field(default_factory=lambda: random.randint(0, 1_000_000))
    time_in_force: int = GTC
    order_type: int = MARKET
    wait: bool = False


class FeatureSpace(ABC):
    """
    Abstract base class for feature spaces.
    Strategies should define their own feature space by inheriting from this class.
    """

    pass


class BaseStrategy(ABC):
    def __init__(self, config: dict[str, object]):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.initialized = False

    @abstractmethod
    def initialize(self, data_loader):
        """Initialize strategy with historical data"""
        pass

    @abstractmethod
    def extract_features(self, hbt: HashMapMarketDepthBacktest_TypeHint) -> FeatureSpace:
        """Extract features from market data"""
        pass

    @abstractmethod
    def generate_signals(self, input_features: FeatureSpace) -> dict[str, TradingSignal]:
        """Generate trading signals based on market data"""
        pass

    def get_state(self) -> object:
        """Get current strategy state for saving/loading"""
        raise NotImplementedError

    def load_state(self, state: object):
        """Load strategy state"""
        raise NotImplementedError


if __name__ == "__main__":
    t1 = TradingSignal(action="BUY", price=100.0, qty=1.0)
    print(t1)
    t2 = TradingSignal(action="SELL", price=101.0, qty=2.0)
    print(t2)
