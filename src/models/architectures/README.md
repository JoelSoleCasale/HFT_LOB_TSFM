# Model Architectures

This directory contains the neural network architectures for financial time series prediction. Each model architecture is in its own file, containing both the configuration dataclass and the model implementation.

## Structure

```
architectures/
├── __init__.py           # Exports all architectures and configs
├── base.py              # Base classes for all models
├── mlp.py               # MLP architecture and config
├── lstm.py              # LSTM architecture and config
└── transformer.py       # Transformer architecture and config
```

## Files

### `base.py`
Contains the base classes:
- `ModelArchitectureConfig`: Base configuration class for all architectures
- `FinancialTimeSeriesModel`: Base PyTorch model class

### `mlp.py`
Multi-layer Perceptron architecture:
- `MLPConfig`: Configuration for MLP models
- `MLPTimeSeriesModel`: MLP model implementation

### `lstm.py`
Long Short-Term Memory architecture:
- `LSTMConfig`: Configuration for LSTM models
- `LSTMTimeSeriesModel`: LSTM model implementation with optional bidirectionality and attention

### `transformer.py`
Transformer architecture:
- `TransformerConfig`: Configuration for Transformer models
- `TransformerTimeSeriesModel`: Transformer model implementation
- `PositionalEncoding`: Positional encoding helper class

## Usage

All architectures and configurations can be imported from the main models module:

```python
from src.models import (
    LSTMConfig,
    TransformerConfig,
    MLPConfig,
    create_model,
)

# Create a configuration
config = LSTMConfig(
    input_size=10,
    hidden_size=128,
    num_layers=2,
    output_size=3,
)

# Create a model using the factory function
model = create_model(
    model_type='lstm',
    input_size=10,
    hidden_size=128,
)
```

## Adding New Architectures

To add a new architecture:

1. Create a new file in this directory (e.g., `gru.py`)
2. Define a config class inheriting from `ModelArchitectureConfig`
3. Define a model class inheriting from `FinancialTimeSeriesModel`
4. Add exports to `__init__.py`
5. Update the factory function in `../model.py`
