"""
Neural network architectures for financial time series prediction.

This module provides a factory function for creating different model types.
The actual model implementations are in the architectures submodule.
"""

from .architectures import (
    FinancialTimeSeriesModel,
    MLPTimeSeriesModel,
    LSTMTimeSeriesModel,
    TransformerTimeSeriesModel,
)


# Factory function for creating models
def create_model(model_type: str, **kwargs) -> FinancialTimeSeriesModel:
    """
    Factory function to create models by type.

    Args:
        model_type: Type of model to create ("mlp", "lstm", "transformer")
        **kwargs: Model-specific parameters

    Returns:
        Instantiated model
    """
    model_classes = {
        "mlp": MLPTimeSeriesModel,
        "lstm": LSTMTimeSeriesModel,
        "transformer": TransformerTimeSeriesModel,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}"
        )

    return model_classes[model_type](**kwargs)
