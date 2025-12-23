"""
Neural network architectures for financial time series prediction.

This module provides backward compatibility by re-exporting from factories.
The actual factory logic is now in factories.py.
"""

# Re-export for backward compatibility
from .factories import create_model, MODEL_CLASS_REGISTRY
from .architectures import (
    FinancialTimeSeriesModel,
    MLPTimeSeriesModel,
    LSTMTimeSeriesModel,
    TransformerTimeSeriesModel,
    DeepLOBModel,
)

__all__ = [
    "create_model",
    "MODEL_CLASS_REGISTRY",
    "FinancialTimeSeriesModel",
    "MLPTimeSeriesModel",
    "LSTMTimeSeriesModel",
    "TransformerTimeSeriesModel",
    "DeepLOBModel",
]
