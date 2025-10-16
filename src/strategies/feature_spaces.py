from pyparsing import ABC
import polars as pl


class FeatureSpace(ABC):
    """
    Abstract base class for feature spaces.
    Strategies should define their own feature space by inheriting from this class.
    """

    pass


class OBSnapshot(FeatureSpace):
    """
    A feature space representing an order book snapshot.
    """

    def __init__(self, bids: pl.DataFrame, asks: pl.DataFrame):
        self.bids = bids
        self.asks = asks

    def __repr__(self):
        return f"OBSnapshot(bids={self.bids}, asks={self.asks})"
