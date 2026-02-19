"""
Example script for training DeepLOB model on limit order book data.

This script demonstrates how to:
1. Load orderbook data
2. Extract features using DeepLOBFeatures extractor
3. Create directional labels
4. Train the DeepLOB model
5. Evaluate the trained model
"""

# ruff: noqa: F403, F405

import warnings

from features import (
    FeaturePipeline,
    FeatureExtractorRegistry,
)
from models import (
    ModelConfig,
    train_model,
)
from models.architectures.lob import DeepLOBConfig
from lob_utils import *  # noqa: F403, F405

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028


def main():
    """Main function demonstrating DeepLOB model training."""

    # Configuration
    LEVELS = DEFAULT_LEVELS  # DeepLOB uses 10 levels (40 features)
    SEQUENCE_LENGTH = DEFAULT_SEQUENCE_LENGTH

    print_training_header("DeepLOB", DEFAULT_FIRST_DATE, DEFAULT_N_DAYS)

    # Load with default sampling
    orderbook_data = load_orderbook_data()

    print_data_loaded()

    # Create input space
    input_space = create_input_space(orderbook_data)

    # Build feature pipeline with DeepLOB features
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(
        FeatureExtractorRegistry.create("deeplob_features", config={"levels": LEVELS})
    )

    print_pipeline_created("DeepLOBFeatures")

    # Extract features and labels
    features = feature_pipeline.extract_all(input_space)
    labels, label_extractor = extract_labels(input_space)

    features_collected = features.collect()
    labels_collected = labels.collect()

    print_features_extracted(features_collected.shape, labels_collected.shape, LEVELS * 4, LEVELS)

    # Print label distribution and calculate class weights
    print_label_distribution(labels_collected, label_extractor.label_names[0])
    class_weights = calculate_class_weights(labels_collected, label_extractor.label_names[0])

    # Create DeepLOB configuration
    INPUT_SIZE = LEVELS * 4

    deeplob_config = DeepLOBConfig(
        input_size=INPUT_SIZE,
        output_size=3,  # -1, 0, 1 for directional labels
        conv_filters=16,
        inception_filters=32,
        lstm_hidden_size=32,
        dropout=0.2,
    )

    print_architecture_config(
        "DeepLOB",
        {
            "Input features": deeplob_config.input_size,
            "Output classes": deeplob_config.output_size,
            "Conv filters": deeplob_config.conv_filters,
            "Inception filters": deeplob_config.inception_filters,
            "LSTM hidden size": deeplob_config.lstm_hidden_size,
            "Dropout": deeplob_config.dropout,
        },
    )

    if max(class_weights) / min(class_weights) > 5.0:
        print("Warning: High class imbalance detected. " "Using a weighted loss function.")
        loss_used = {
            "loss_function": "focal",
            "loss_params": {"alpha": class_weights, "gamma": 2.0},
        }
    else:
        print("Class imbalance within acceptable range. Using standard loss function.")
        loss_used = {"loss_function": "cross_entropy"}

    # Create full model configuration
    config = ModelConfig(
        architecture=deeplob_config,
        data=create_data_config(
            sequence_length=SEQUENCE_LENGTH,
        ),
        training=create_training_config(
            **loss_used,
        ),
        logging=create_logging_config(
            experiment_name=f"deeplob-{loss_used['loss_function'].upper()}",
            lambda_value=label_extractor.threshold,
        ),
    )

    print_training_start(config)

    # Train the model
    trainer, results = train_model(
        features=features,
        labels=labels,
        config=config,
        feature_columns=None,  # Use all features except timestamp
        label_columns=None,  # Use all label columns
    )


if __name__ == "__main__":
    main()
