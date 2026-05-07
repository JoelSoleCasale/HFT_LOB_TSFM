"""
Feature engineering module

Provides base classes, concrete extractors, and pipelines
for transforming raw market data into features and labels.
"""

from .base.input_space import InputSpace
from .base.feature_extractor import BaseFeatureExtractor
from .base.label_extractor import BaseLabelExtractor
from .pipeline import FeaturePipeline
from .registry import FeatureExtractorRegistry, LabelExtractorRegistry
from .extractors.orderbook_features import (
    MidPriceFeatures,
    OrderbookImbalanceFeatures,
    SpreadFeatures,
    AdvancedOrderbookFeatures,
)
from .extractors.trade_features import TradeFlowFeatures
from .labels.price_labels import MidPriceReturnLabel
from .labels.directional_labels import DirectionalLabel, SmoothedDirectionalLabel
from .labels.barrier_labels import TripleBarrierLabel

__all__ = [
    "InputSpace",
    "BaseFeatureExtractor",
    "BaseLabelExtractor",
    "FeaturePipeline",
    "FeatureExtractorRegistry",
    "LabelExtractorRegistry",
    "MidPriceFeatures",
    "OrderbookImbalanceFeatures",
    "SpreadFeatures",
    "TradeFlowFeatures",
    "AdvancedOrderbookFeatures",
    "MidPriceReturnLabel",
    "DirectionalLabel",
    "SmoothedDirectionalLabel",
    "TripleBarrierLabel",
]
