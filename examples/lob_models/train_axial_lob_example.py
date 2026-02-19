"""
Example script for training Axial-LOB model on limit order book data.

This script demonstrates how to:
1. Load orderbook data
2. Extract features using AxialLOBFeatures extractor
3. Create directional labels
4. Train the Axial-LOB model
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
from models.architectures.lob import AxialLOBConfig
from lob_utils import *  # noqa: F403, F405

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028


def main():
    """Main function demonstrating Axial-LOB model training."""

    # Configuration
    LEVELS = DEFAULT_LEVELS  # Axial-LOB uses 10 levels (40 features)

    print_training_header("Axial-LOB", DEFAULT_FIRST_DATE, DEFAULT_N_DAYS)

    # Load orderbook data with default sampling
    orderbook_data = load_orderbook_data()

    print_data_loaded()

    # Create input space
    input_space = create_input_space(orderbook_data)

    # Build feature pipeline with AxialLOB features
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(
        FeatureExtractorRegistry.create("axial_lob_features", config={"levels": LEVELS})
    )

    print_pipeline_created("AxialLOBFeatures")

    # Extract features and labels
    features = feature_pipeline.extract_all(input_space)
    labels, label_extractor = extract_labels(input_space)

    features_collected = features.collect()
    labels_collected = labels.collect()

    print_features_extracted(features_collected.shape, labels_collected.shape, LEVELS * 4, LEVELS)

    # Print label distribution and calculate class weights
    print_label_distribution(labels_collected, label_extractor.label_names[0])
    class_weights = calculate_class_weights(labels_collected, label_extractor.label_names[0])

    # Create Axial-LOB configuration
    SEQUENCE_LENGTH = 128  # Axial-LOB uses 40 timesteps
    INPUT_SIZE = LEVELS * 4

    axial_lob_config = AxialLOBConfig(
        input_size=INPUT_SIZE,
        output_size=3,  # -1, 0, 1 for directional labels
        c_in=32,
        c_out=32,
        c_final=4,
        n_heads=4,
        pool_kernel=(1, 4),
        pool_stride=(1, 4),
        sequence_length=SEQUENCE_LENGTH,
    )

    print_architecture_config(
        "Axial-LOB",
        {
            "Input features": axial_lob_config.input_size,
            "Output classes": axial_lob_config.output_size,
            "Sequence length": axial_lob_config.sequence_length,
            "Conv channels in": axial_lob_config.c_in,
            "Conv channels out": axial_lob_config.c_out,
            "Number of heads": axial_lob_config.n_heads,
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
        architecture=axial_lob_config,
        data=create_data_config(
            sequence_length=SEQUENCE_LENGTH,
            batch_size=128,
        ),
        training=create_training_config(
            **loss_used,
        ),
        logging=create_logging_config(
            experiment_name=f"axial_lob-{loss_used['loss_function'].upper()}",
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
