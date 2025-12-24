"""
Example script for training CTABL model on limit order book data.

CTABL architecture: Bilinear layers + Temporal Attention Augmented Bilinear layer.
Default template per paper: [40x10] -> [120x5] -> [3x1]
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028

from features import (
    FeaturePipeline,
    FeatureExtractorRegistry,
)
from models import (
    ModelConfig,
    train_model,
)
from models.architectures.lob import CTABLConfig
from lob_utils import *


def main():
    """Train CTABL on 10 days of LOB snapshots for BTCUSDT."""

    LEVELS = DEFAULT_LEVELS  # 40 features

    print_training_header("CTABL", DEFAULT_FIRST_DATE, DEFAULT_N_DAYS)

    # Load with default sampling
    orderbook_data = load_orderbook_data()

    print_data_loaded()

    input_space = create_input_space(orderbook_data)

    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(
        FeatureExtractorRegistry.create("ctabl_features", config={"levels": LEVELS})
    )

    print_pipeline_created("CTABLFeatures")

    features = feature_pipeline.extract_all(input_space)
    labels, label_extractor = extract_labels(
        input_space, threshold=3e-4
    )  # Override default threshold

    features_collected = features.collect()
    labels_collected = labels.collect()

    print_features_extracted(features_collected.shape, labels_collected.shape, LEVELS * 4, LEVELS)

    print_label_distribution(labels_collected, label_extractor.label_names[0])
    class_weights = calculate_class_weights(labels_collected, label_extractor.label_names[0])

    SEQUENCE_LENGTH = 128
    INPUT_SIZE = LEVELS * 4

    ctabl_config = CTABLConfig(
        input_size=INPUT_SIZE,
        output_size=3,
        time_steps=SEQUENCE_LENGTH,
        hidden_dims=((120, 5),),
        output_dims=(3, 1),
        dropout=0.1,
        activation="relu",
    )

    print_architecture_config(
        "CTABL",
        {
            "Input features": ctabl_config.input_size,
            "Time steps": ctabl_config.time_steps,
            "Hidden dims": ctabl_config.hidden_dims,
            "Output dims": ctabl_config.output_dims,
            "Dropout": ctabl_config.dropout,
        },
    )

    config = ModelConfig(
        architecture=ctabl_config,
        data=create_data_config(
            sequence_length=SEQUENCE_LENGTH,
            batch_size=128,
        ),
        training=create_training_config(
            # loss_function="focal",
            # loss_params={"alpha": class_weights, "gamma": 2.0},
            gradient_clip_norm=1.0,  # Add gradient clipping for stability
        ),
        logging=create_logging_config(
            experiment_name="ctabl",
            lambda_value=label_extractor.threshold,
        ),
    )

    print_training_start(config)

    trainer, results = train_model(
        features=features,
        labels=labels,
        config=config,
        feature_columns=None,
        label_columns=None,
    )


if __name__ == "__main__":
    main()
