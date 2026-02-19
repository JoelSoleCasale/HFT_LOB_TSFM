"""
Example script for training DeepLOB-Attention model on limit order book data.

This script demonstrates how to:
1. Load orderbook data
2. Extract features using DeepLOBAttentionFeatures extractor
3. Create directional labels
4. Train the DeepLOB-Attention model
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
from models.architectures.lob import DeepLOBAttentionConfig
from lob_utils import *  # noqa: F403, F405

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028


def main():
    """Main function demonstrating DeepLOB-Attention model training."""

    # Configuration
    LEVELS = DEFAULT_LEVELS  # 10 levels -> 40 features

    print_training_header("DeepLOB-Attention", DEFAULT_FIRST_DATE, DEFAULT_N_DAYS)

    # Load with default sampling
    orderbook_data = load_orderbook_data()

    print_data_loaded()

    # Create input space
    input_space = create_input_space(orderbook_data)

    # Build feature pipeline with DeepLOBAttention features
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(
        FeatureExtractorRegistry.create("deeplob_attention_features", config={"levels": LEVELS})
    )

    print_pipeline_created("DeepLOBAttentionFeatures")

    # Extract features and labels
    features = feature_pipeline.extract_all(input_space)
    labels, label_extractor = extract_labels(input_space)

    features_collected = features.collect()
    labels_collected = labels.collect()

    print_features_extracted(features_collected.shape, labels_collected.shape, LEVELS * 4, LEVELS)

    # Print label distribution and calculate class weights
    print_label_distribution(labels_collected, label_extractor.label_names[0])
    class_weights = calculate_class_weights(labels_collected, label_extractor.label_names[0])

    # Create DeepLOB-Attention configuration
    SEQUENCE_LENGTH = 128  # Paper uses 50 timesteps for attention models
    INPUT_SIZE = LEVELS * 4

    attention_config = DeepLOBAttentionConfig(
        input_size=INPUT_SIZE,
        output_size=3,  # -1, 0, 1 for directional labels
        conv_filters=32,
        inception_filters=64,
        lstm_hidden_size=64,
        dropout=0.0,
        activation="leaky_relu",
        leaky_relu_slope=0.01,
        use_batch_norm=True,
    )

    print_architecture_config(
        "DeepLOB-Attention",
        {
            "Input features": attention_config.input_size,
            "Output classes": attention_config.output_size,
            "Conv filters": attention_config.conv_filters,
            "Inception filters": attention_config.inception_filters,
            "LSTM hidden size": attention_config.lstm_hidden_size,
            "Dropout": attention_config.dropout,
        },
    )

    if max(class_weights) / min(class_weights) > 5.0:
        print("Warning: High class imbalance detected. " "Using a weighted loss function.")
        loss_used = {
            "loss_function": "focal",
            "loss_params": {"alpha": class_weights, "gamma": 1.0},
        }
    else:
        print("Class imbalance within acceptable range. Using standard loss function.")
        loss_used = {"loss_function": "cross_entropy"}

    # Create full model configuration
    config = ModelConfig(
        architecture=attention_config,
        data=create_data_config(
            sequence_length=SEQUENCE_LENGTH,
        ),
        training=create_training_config(
            **loss_used,
        ),
        logging=create_logging_config(
            experiment_name=f"deeplob_attention-{loss_used['loss_function'].upper()}",
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
