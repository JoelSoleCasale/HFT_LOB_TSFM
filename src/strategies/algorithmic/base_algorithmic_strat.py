from strategies.base_strategy import FeatureSpace, TradingSignal
from abc import ABC, abstractmethod


class BaseAlgorithmicStrategy(ABC):
    def __init__(self, config: dict[str, object]):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.initialized = False

    @abstractmethod
    def initialize(self, data_loader):
        """Initialize strategy with historical data"""
        pass

    @abstractmethod
    def extract_features(self, hbt) -> FeatureSpace:
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
