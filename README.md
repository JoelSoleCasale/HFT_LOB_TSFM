# High-Frequency Cryptocurrency Trading ML

A modular deep learning framework for predicting price movements from limit orderbook data. This project processes high-frequency cryptocurrency market data and trains neural networks to forecast directional price changes.

## Key Features

- **Orderbook Data Processing**: Efficient handling of high-frequency orderbook snapshots using Polars with lazy evaluation
- **Feature Engineering Pipeline**: Registry-based, composable feature extractors for market microstructure patterns
- **Multiple Architectures**: LSTM, Transformer, and MLP models with PyTorch
- **Advanced Labeling**: Triple barrier method for directional prediction (up/neutral/down)
- **Training Infrastructure**: Built-in callbacks (early stopping, checkpointing), custom metrics (trade accuracy), and experiment tracking via Weights & Biases
- **GPU Support**: Automatic device detection (CUDA, Apple Silicon MPS, CPU fallback)

## Architecture

The project is organized into three main components:

1. **Data Management** (`src/data_manager/`): Downloads orderbook snapshots and trades from cryptohftdata API, processes and stores as Parquet files
2. **Feature Engineering** (`src/features/`): Extracts features (price spreads, orderbook imbalance, etc.) and generates labels using configurable pipelines
3. **Model Training** (`src/models/`): PyTorch-based training with extensive metrics, callbacks, and wandb integration

**Data Flow**: Raw Parquet → OrderBook (Polars) → InputSpace → FeaturePipeline → PyTorch Dataset → Model Training

## Setup

1. Install [uv](https://docs.astral.sh/uv/)
2. Install dependencies: `uv sync` (use `uv sync --extra cu126`, `uv sync --extra cu128`, or `uv sync --extra cu130` for specific CUDA versions)
3. Set up pre-commit hooks: `uv run poe hooks`
4. Create `.env` file with `CRYPTOHFTDATA_API_KEY=<your-key>` for data downloads

### Download and Process Data

Before training models, you need to download and preprocess the orderbook data:

```bash
# Download raw orderbook data (event-based)
# Configuration: config/data_manager/downloader/default.yaml
uv run src/data_manager/downloader/

# Preprocess to snapshot-based orderbook
# Configuration: config/data_manager/processing/default.yaml
uv run src/data_manager/processing/
```

The downloader fetches data from cryptohftdata API based on the configuration (exchange, symbol, date range). The processing step converts event-based orderbook updates into fixed-interval snapshots suitable for model training.

## Quick Start

Once data is downloaded and processed, you can train models:

```python
from pathlib import Path
from datetime import date
from core.orderbook import OrderBook
from features import InputSpace, FeaturePipeline, FeatureExtractorRegistry, TripleBarrierLabel
from models import ModelConfig, LSTMConfig, train_model

# Load orderbook data
orderbook = OrderBook.from_parquet("data/orderbook_snapshots/binance_futures/BTCUSDT/2025-07-01_L20.parquet")

# Extract features
input_space = InputSpace(orderbook=orderbook)
pipeline = FeaturePipeline()
pipeline.add_extractor(FeatureExtractorRegistry.create("mid_price"))
pipeline.add_extractor(FeatureExtractorRegistry.create("orderbook_imbalance"))
features = pipeline.extract_all(input_space).collect()

# Generate labels
labeler = TripleBarrierLabel(upper=0.001, lower=-0.001, horizon=10)
labels = labeler.generate(input_space).collect()

# Train model
config = ModelConfig(architecture=LSTMConfig(hidden_size=64, num_layers=2))
trainer = train_model(features, labels, config)
```

See [`examples/`](examples/) for complete workflows.

## Project Structure

- **`src/data_manager/`**: Data downloading and preprocessing
- **`src/features/`**: Feature extraction and label generation
- **`src/models/`**: Neural network architectures and training
- **`src/embeddings/`**: Pretrained model embeddings (Chronos)
- **`config/`**: YAML configuration files for data and models
- **`examples/`**: Complete usage examples for training and evaluation
- **`tests/`**: Unit and integration tests

## Testing

- `uv run poe test`: Run all tests (excluding slow tests)
- `uv run poe test-full`: Run all tests including slow embedding tests
- `uv run poe test-unit`: Unit tests only
- `uv run poe test-integration`: Integration tests only
- `uv run poe test-cov`: Generate coverage report

**Note**: Slow tests (marked with `@pytest.mark.slow`) include embedding generation and can take several minutes.

## Data Requirements

Orderbook data is downloaded from [cryptohftdata](https://cryptohftdata.com/) API. Data schema includes:
- Timestamp (nanosecond precision)
- Bid/ask prices and quantities (configurable levels)
- Stored as compressed Parquet files organized by exchange/symbol/date

## Model Configuration

Models are configured using dataclass composition:

```python
config = ModelConfig(
    data=DataConfig(sequence_length=10, batch_size=32, num_classes=3),
    training=TrainingConfig(learning_rate=0.001, num_epochs=100),
    logging=LoggingConfig(wandb_enabled=True),
    architecture=LSTMConfig(hidden_size=64, num_layers=2)  # Or TransformerConfig, MLPConfig
)
```

## License

This project is developed as part of a Bachelor's thesis (TFG) at Universitat Politècnica de Catalunya (UPC).
