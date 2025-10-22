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
from models import (
    ModelConfig,
    LSTMConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
    train_model,
)
from utils import date_range
from datetime import date, timedelta
from definitions import ROOT_DIR
from core.orderbook import OrderBook


def main():
    """Main function demonstrating model training."""

    # Configuration
    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 10

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
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(5)
        .sample_by_time(time_delta=1_000_000_000, interpolate=True)
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
    directional_return_label = DirectionalLabel(config={"horizons": [20], "threshold": 1e-4})

    print("Extracting features and labels...")

    # Extract features and labels
    features = feature_pipeline.extract_all(input_space).lazy()
    labels = directional_return_label.extract(input_space).lazy()

    print(f"Features shape: {features.collect().shape}")
    print(f"Labels shape: {labels.collect().shape}")

    # Print label percentage class distributions
    labels_df = labels.collect()
    label_counts = labels_df[directional_return_label.label_names[0]].value_counts()
    total_samples = len(labels_df)

    print("\nLabel distribution:")
    for label_value, count in label_counts.iter_rows():
        percentage = (count / total_samples) * 100
        print(f"  Label {label_value}: {count:,} samples ({percentage:.2f}%)")

    # Create model configuration with new structure
    # Example 1: LSTM Configuration
    lstm_config = LSTMConfig(
        input_size=4,  # mid_price + orderbook_imbalance
        hidden_size=64,
        num_layers=2,
        output_size=3,  # -1, 0, 1 for directional labels
        dropout=0.2,
        bidirectional=False,
        attention=False,
    )

    # Example 2: Transformer Configuration (uncomment to use)
    # transformer_config = TransformerConfig(
    #     input_size=4,
    #     d_model=64,
    #     nhead=8,
    #     num_layers=2,
    #     output_size=3,
    #     dropout=0.2,
    #     dim_feedforward=256,
    # )

    # Example 3: MLP Configuration (uncomment to use)
    # mlp_config = MLPConfig(
    #     input_size=4,
    #     output_size=3,
    #     hidden_sizes=[64, 32],
    #     dropout=0.2,
    # )

    config = ModelConfig(
        architecture=lstm_config,  # Use lstm_config, transformer_config, or mlp_config
        data=DataConfig(
            sequence_length=10,
            batch_size=32,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
        ),
        training=TrainingConfig(
            learning_rate=0.001,
            num_epochs=2,
            early_stopping_patience=5,
            optimizer="adam",  # "adam", "adamw", "sgd", "rmsprop"
            scheduler="cosine",  # "cosine", "step", "plateau", None
            loss_function="cross_entropy",  # "cross_entropy", "mse", "mae", "focal"
        ),
        logging=LoggingConfig(
            project_name="financial-models",
            experiment_name="btc_directional_prediction",
            wandb_enabled=True,
            log_confusion_matrix=True,
            log_learning_curves=True,
        ),
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


if __name__ == "__main__":
    main()
