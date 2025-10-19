"""
Example script showing how to use the models module with the existing feature pipeline.

This script demonstrates how to:
1. Load orderbook data
2. Extract features and labels using the existing pipeline
3. Train a deep learning model using the models module
4. Evaluate the trained model
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features import (
    InputSpace,
    FeaturePipeline,
    FeatureExtractorRegistry,
    DirectionalLabel,
)
from models import ModelConfig, train_model
from utils import date_range
from datetime import date, timedelta
from definitions import ROOT_DIR
from core.orderbook import OrderBook


def main():
    """Main function demonstrating model training."""

    # Configuration
    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 2

    print("Loading orderbook data...")

    # Load orderbook data (same as in testing.ipynb)
    def get_ob_path(date: date) -> Path:
        return (
            ROOT_DIR
            / f'data/orderbook_snapshots/binance_futures/BTCUSDT/{date.strftime("%Y-%m-%d")}_L20.parquet'
        )

    ob_paths = [
        get_ob_path(d) for d in date_range(FIRST_DATE, FIRST_DATE + timedelta(days=N_DAYS))
    ]
    orderbook_data = (
        OrderBook.from_parquet(ob_paths)
        .select_levels(5)
        .sample_by_time(time_delta=100_000_000, interpolate=True)
    )

    # Ignore first second
    orderbook_data._data._df = orderbook_data.df[10:]

    print("Creating input space and feature pipeline...")

    # Create input space
    input_space = InputSpace(
        orderbook_snapshots=orderbook_data,
    )

    # Build feature pipeline
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(FeatureExtractorRegistry.create("mid_price")).add_extractor(
        FeatureExtractorRegistry.create("orderbook_imbalance")
    )

    # Build label pipeline
    directional_return_label = DirectionalLabel(config={"horizons": [100], "threshold": 0.00005})

    print("Extracting features and labels...")

    # Extract features and labels
    features = feature_pipeline.extract_all(input_space).lazy()
    labels = directional_return_label.extract(input_space).lazy()

    print(f"Features shape: {features.collect().shape}")
    print(f"Labels shape: {labels.collect().shape}")

    # Create model configuration
    config = ModelConfig(
        model_type="lstm",  # Try "mlp", "lstm", or "transformer"
        input_size=2,  # mid_price + orderbook_imbalance
        hidden_size=64,
        num_layers=2,
        output_size=3,  # -1, 0, 1 for directional labels
        sequence_length=10,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50,
        early_stopping_patience=10,
        project_name="financial-models",
        experiment_name="btc_directional_prediction",
        wandb_enabled=True,  # Set to False if you don't want wandb logging
    )

    print("Starting model training...")
    print(f"Model configuration: {config}")

    # Train the model
    trainer, results = train_model(
        features=features,
        labels=labels,
        config=config,
        feature_columns=None,  # Will auto-detect
        label_columns=None,  # Will auto-detect
    )

    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Total training time: {results['total_training_time']:.2f} seconds")

    if results["test_metrics"]:
        print(f"Test accuracy: {results['test_metrics'].get('val_accuracy', 'N/A'):.4f}")

    # Example of making predictions
    print("\nMaking predictions on a sample...")

    # You can also load the trained model later
    from models import load_model, LSTMTimeSeriesModel

    # Load the trained model
    loaded_model, loaded_config, loaded_metadata = load_model(
        config.model_save_path,
        LSTMTimeSeriesModel(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
        ),
    )

    print("Model loaded successfully!")
    print(f"Feature names: {loaded_metadata.get('feature_names', 'N/A')}")
    print(f"Label names: {loaded_metadata.get('label_names', 'N/A')}")


if __name__ == "__main__":
    main()
