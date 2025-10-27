"""
Feature engineering module

Provides base classes, concrete extractors, and pipelines
for transforming raw market data into features and labels.
"""

# Base classes
from .base.input_space import InputSpace
from .base.feature_extractor import BaseFeatureExtractor
from .base.label_extractor import BaseLabelExtractor

# Pipelines
from .pipeline import FeaturePipeline

# Registries
from .registry import FeatureExtractorRegistry, LabelExtractorRegistry

# Expose commonly used extractors (optional)
from .extractors.orderbook_features import (
    MidPriceFeatures,
    OrderbookImbalanceFeatures,
    SpreadFeatures,
    AdvancedOrderbookFeatures,
)
from .extractors.trade_features import TradeFlowFeatures
from .labels.price_labels import MidPriceReturnLabel
from .labels.directional_labels import DirectionalLabel, SmoothedDirectionalLabel

__all__ = [
    # Base
    "InputSpace",
    "BaseFeatureExtractor",
    "BaseLabelExtractor",
    # Pipelines
    "FeaturePipeline",
    # Registries
    "FeatureExtractorRegistry",
    "LabelExtractorRegistry",
    # Common extractors
    "MidPriceFeatures",
    "OrderbookImbalanceFeatures",
    "SpreadFeatures",
    "TradeFlowFeatures",
    "AdvancedOrderbookFeatures",
    "MidPriceReturnLabel",
    "DirectionalLabel",
    "SmoothedDirectionalLabel",
]
