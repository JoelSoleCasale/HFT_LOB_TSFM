"""
Example script for training CTABL model on limit order book data.

CTABL architecture: Bilinear layers + Temporal Attention Augmented Bilinear layer.
Default template per paper: [40x10] -> [120x5] -> [3x1]
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
from models.architectures.lob import CTABLConfig
from utils import date_range
from definitions import ROOT_DIR
from core.orderbook import OrderBook

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028


def main():
    """Train CTABL on 10 days of LOB snapshots for BTCUSDT."""

    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 10

    print("=" * 80)
    print("CTABL Model Training Example")
    print("=" * 80)
    print(f"\nLoading orderbook data for {N_DAYS} days starting from {FIRST_DATE}...")

    def get_ob_path(d: date) -> Path:
        return (
            ROOT_DIR
            / f"data/orderbook_snapshots/binance_futures/BTCUSDT/{d.strftime('%Y-%m-%d')}_L20.parquet"
        )

    ob_paths = [
        get_ob_path(d) for d in date_range(FIRST_DATE, FIRST_DATE + timedelta(days=N_DAYS - 1))
    ]

    # Load with 10 levels (40 features) and sample sequences of length 10 for CTABL
    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(10)
        .sample_by_time(time_delta=100_000_000, interpolate=True)
    )

    print("✓ Orderbook data loaded successfully")
    print("\nCreating input space and feature pipeline...")

    input_space = InputSpace(orderbook_snapshots=orderbook_data)

    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(
        FeatureExtractorRegistry.create("ctabl_features", config={"levels": 10})
    )

    directional_return_label = TripleBarrierLabel(config={"horizon": 200, "threshold": 3e-4})

    print("✓ Feature pipeline created with CTABLFeatures extractor")
    print("\nExtracting features and labels...")

    features = feature_pipeline.extract_all(input_space)
    labels = directional_return_label.extract(input_space)

    features_collected = features.collect()
    labels_collected = labels.collect()

    print(f"✓ Features extracted: {features_collected.shape}")
    print("  Expected: 40 features (10 levels × 4 features per level)")
    print(f"✓ Labels extracted: {labels_collected.shape}")

    label_counts = labels_collected[directional_return_label.label_names[0]].value_counts()
    total_samples = len(labels_collected)

    print("\nLabel distribution:")
    for label_value, count in label_counts.iter_rows():
        percentage = (count / total_samples) * 100
        label_name = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}.get(label_value, label_value)
        print(f"  {label_name:8s} (label={label_value:2d}): {count:,} samples ({percentage:.2f}%)")

    class_weights = [0.0, 0.0, 0.0]
    for label_value, count in label_counts.iter_rows():
        freq = count / total_samples
        class_weights[label_value + 1] = 1 / freq

    print(f"\nClass weights for focal loss: {[f'{w:.2f}' for w in class_weights]}")

    print("\n" + "=" * 80)
    print("Configuring CTABL Model")
    print("=" * 80)

    ctabl_config = CTABLConfig(
        input_size=40,
        output_size=3,
        time_steps=10,
        hidden_dims=((120, 5),),
        output_dims=(3, 1),
        dropout=0.1,
        activation="relu",
    )

    print("\nArchitecture: CTABL")
    print(f"  Input features: {ctabl_config.input_size}")
    print(f"  Time steps: {ctabl_config.time_steps}")
    print(f"  Hidden dims: {ctabl_config.hidden_dims}")
    print(f"  Output dims: {ctabl_config.output_dims}")
    print(f"  Dropout: {ctabl_config.dropout}")

    config = ModelConfig(
        architecture=ctabl_config,
        data=DataConfig(
            sequence_length=10,
            batch_size=256,
            stride=5,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            device="cpu",
        ),
        training=TrainingConfig(
            learning_rate=0.01,  # per paper/examples
            num_epochs=100,
            early_stopping_patience=5,
            optimizer="adam",
            scheduler="cosine",
            loss_function="focal",
            loss_params={"alpha": class_weights, "gamma": 2.0},
            mixed_precision=True,
        ),
        logging=LoggingConfig(
            project_name="financial-models-lob",
            experiment_name="ctabl_btc_directional_prediction",
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
    print(f"  Mixed precision: {config.training.mixed_precision}")

    trainer, results = train_model(
        features=features,
        labels=labels,
        config=config,
        feature_columns=None,
        label_columns=None,
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
