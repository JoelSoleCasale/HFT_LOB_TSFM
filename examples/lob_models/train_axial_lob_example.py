"""
Example script for training Axial-LOB model on limit order book data.

This script demonstrates how to:
1. Load orderbook data
2. Extract features using AxialLOBFeatures extractor
3. Create directional labels
4. Train the Axial-LOB model
5. Evaluate the trained model
"""

from pathlib import Path
import warnings
from datetime import date, timedelta

from features import (
    InputSpace,
    FeaturePipeline,
    FeatureExtractorRegistry,
    TripleBarrierLabel,
)
from models import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
    train_model,
)
from models.architectures.lob import AxialLOBConfig
from utils import date_range
from definitions import ROOT_DIR
from core.orderbook import OrderBook

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028


def main():
    """Main function demonstrating Axial-LOB model training."""

    # Configuration
    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 10  # Use 10 days for training

    print("=" * 80)
    print("Axial-LOB Model Training Example")
    print("=" * 80)
    print(f"\nLoading orderbook data for {N_DAYS} days starting from {FIRST_DATE}...")

    # Load orderbook data
    def get_ob_path(date: date) -> Path:
        return (
            ROOT_DIR
            / f'data/orderbook_snapshots/binance_futures/BTCUSDT/{date.strftime("%Y-%m-%d")}_L20.parquet'
        )

    ob_paths = [
        get_ob_path(d) for d in date_range(FIRST_DATE, FIRST_DATE + timedelta(days=N_DAYS - 1))
    ]

    # Load with 10 levels for Axial-LOB (needs 40 features)
    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(10)  # Axial-LOB uses 10 levels
        .sample_by_time(time_delta=100_000_000, interpolate=True)
    )

    print("✓ Orderbook data loaded successfully")
    print("\nCreating input space and feature pipeline...")

    # Create input space
    input_space = InputSpace(
        orderbook_snapshots=orderbook_data,
    )

    # Build feature pipeline with AxialLOB features
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(
        FeatureExtractorRegistry.create("axial_lob_features", config={"levels": 10})
    )

    # Build label pipeline
    directional_return_label = TripleBarrierLabel(config={"horizon": 200, "threshold": 3e-4})

    print("✓ Feature pipeline created with AxialLOBFeatures extractor")
    print("\nExtracting features and labels...")

    # Extract features and labels
    features = feature_pipeline.extract_all(input_space)
    labels = directional_return_label.extract(input_space)

    features_collected = features.collect()
    labels_collected = labels.collect()

    print(f"✓ Features extracted: {features_collected.shape}")
    print("  Expected: 40 features (10 levels × 4 features per level)")
    print(f"✓ Labels extracted: {labels_collected.shape}")

    # Print label distribution
    label_counts = labels_collected[directional_return_label.label_names[0]].value_counts()
    total_samples = len(labels_collected)

    print("\nLabel distribution:")
    for label_value, count in label_counts.iter_rows():
        percentage = (count / total_samples) * 100
        label_name = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}.get(label_value, label_value)
        print(f"  {label_name:8s} (label={label_value:2d}): {count:,} samples ({percentage:.2f}%)")

    # Calculate class weights for imbalanced data
    class_weights = [0.0, 0.0, 0.0]
    for label_value, count in label_counts.iter_rows():
        freq = count / total_samples
        class_weights[label_value + 1] = 1 / freq

    print(f"\nClass weights for focal loss: {[f'{w:.2f}' for w in class_weights]}")

    # Create Axial-LOB configuration
    print("\n" + "=" * 80)
    print("Configuring Axial-LOB Model")
    print("=" * 80)

    axial_lob_config = AxialLOBConfig(
        input_size=40,  # 10 levels × 4 features per level
        output_size=3,  # -1, 0, 1 for directional labels
        c_in=32,
        c_out=32,
        c_final=4,
        n_heads=4,
        pool_kernel=(1, 4),
        pool_stride=(1, 4),
        sequence_length=40,  # Axial-LOB uses 40 timesteps
    )

    print("\nArchitecture: Axial-LOB")
    print(f"  Input features: {axial_lob_config.input_size}")
    print(f"  Output classes: {axial_lob_config.output_size}")
    print(f"  Sequence length: {axial_lob_config.sequence_length}")

    # Create full model configuration
    config = ModelConfig(
        architecture=axial_lob_config,
        data=DataConfig(
            sequence_length=40,  # Axial-LOB uses 40 timesteps
            batch_size=64,  # Paper uses 64
            stride=5,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            device="cuda",
        ),
        training=TrainingConfig(
            learning_rate=0.01,  # Paper uses 0.01
            num_epochs=50,
            early_stopping_patience=10,
            optimizer="sgd",  # Paper uses SGD
            scheduler="cosine",
            loss_function="cross_entropy",  # Paper uses categorical cross-entropy
            # loss_params={"weight": torch.tensor(class_weights)}, # Can pass weights if needed
            mixed_precision=False,
        ),
        logging=LoggingConfig(
            project_name="financial-models-lob",
            experiment_name="axial_lob_btc_directional_prediction",
            wandb_enabled=True,
            log_confusion_matrix=True,
            log_trade_accuracy_vs_threshold=True,
            lambda_value=directional_return_label.threshold,
            theta_values=[0.0, 1e-4, 4e-4],
        ),
    )

    print("\n" + "=" * 80)
    print("Starting Model Training")
    print("=" * 80)
    print("\nData configuration:")
    print(f"  Sequence length: {config.data.sequence_length}")
    print(f"  Batch size: {config.data.batch_size}")
    print(
        f"  Train/Val/Test split: {config.data.train_split:.0%}/{config.data.val_split:.0%}/{config.data.test_split:.0%}"
    )
    print(f"  Device: {config.data.device}")

    print("\nTraining configuration:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Loss function: {config.training.loss_function}")

    # Train the model
    trainer, results = train_model(
        features=features,
        labels=labels,
        config=config,
        feature_columns=None,  # Use all features except timestamp
        label_columns=None,  # Use all label columns
    )

    print("\n" + "=" * 80)
    print("Training Completed Successfully!")
    print("=" * 80)
    print("\nFinal Results:")
    print(f"  Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"  Best validation loss: {results['best_val_loss']:.4f}")
    print(f"  Total training time: {results.get('training_time', 'N/A')}")

    if "test_metrics" in results:
        print("\nTest Set Performance:")
        for metric, value in results["test_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
