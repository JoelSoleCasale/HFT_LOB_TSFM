"""
Example script showing how to use the models module with the existing feature pipeline.

This script demonstrates how to:
1. Load orderbook data
2. Extract features and labels using the existing pipeline
3. Train a deep learning model using the models module
4. Evaluate the trained model
"""

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
    LSTMConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
    train_model,
)
from utils import date_range, get_ob_path
from core.orderbook import OrderBook

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028


def main() -> None:
    """Main function demonstrating model training."""

    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 10

    print("Loading orderbook data...")

    ob_paths = [
        get_ob_path(d) for d in date_range(FIRST_DATE, FIRST_DATE + timedelta(days=N_DAYS - 1))
    ]
    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(5)
        .sample_by_time(time_delta=100_000_000, interpolate=True)
    )

    print("Creating input space and feature pipeline...")

    input_space = InputSpace(
        orderbook_snapshots=orderbook_data,
    )

    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(FeatureExtractorRegistry.create("advanced_orderbook"))

    directional_return_label = TripleBarrierLabel(config={"horizon": 200, "threshold": 3e-4})

    print("Extracting features and labels...")

    features = feature_pipeline.extract_all(input_space)
    labels = directional_return_label.extract(input_space)

    print(f"Features shape: {features.collect().shape}")
    print(f"Labels shape: {labels.collect().shape}")

    labels_df = labels.collect()
    label_counts = labels_df[directional_return_label.label_names[0]].value_counts()
    total_samples = len(labels_df)

    print("\nLabel distribution:")
    for label_value, count in label_counts.iter_rows():
        percentage = (count / total_samples) * 100
        print(f"  Label {label_value}: {count:,} samples ({percentage:.2f}%)")

    lstm_config = LSTMConfig(
        input_size=len(features.columns) - 1,  # exclude timestamp
        hidden_size=64,
        num_layers=1,
        output_size=3,  # -1, 0, 1 for directional labels
        dropout=0.2,
        bidirectional=True,
        attention=False,
    )

    class_weights = [0.0, 0.0, 0.0]

    for label_value, count in label_counts.iter_rows():
        freq = count / total_samples
        class_weights[label_value + 1] = 1 / freq

    config = ModelConfig(
        architecture=lstm_config,
        data=DataConfig(
            sequence_length=256,
            batch_size=1024,
            stride=5,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            device="cuda",
        ),
        training=TrainingConfig(
            learning_rate=0.001,
            num_epochs=50,
            early_stopping_patience=5,
            optimizer="adam",  # "adam", "adamw", "sgd", "rmsprop"
            scheduler="cosine",  # "cosine", "step", "plateau", None
            loss_function="focal",  # "cross_entropy", "mse", "mae", "focal"
            loss_params={"alpha": class_weights, "gamma": 2.0},
            mixed_precision=True,
        ),
        logging=LoggingConfig(
            project_name="financial-models",
            experiment_name="small_noAtt_lstm_btc_directional_prediction",
            wandb_enabled=True,
            log_confusion_matrix=True,
            log_trade_accuracy_vs_threshold=True,
            lambda_value=directional_return_label.threshold,
            theta_values=[0.0, 1e-4, 4e-4],
        ),
    )

    print("Starting model training...")
    print(f"Model configuration: {config}")

    trainer, results = train_model(
        features=features,
        labels=labels,
        config=config,
        feature_columns=None,
        label_columns=None,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
