"""
Script for running multiple model training experiments sequentially.

This script allows you to define a list of ModelConfig objects and trains them
one after another, useful for hyperparameter sweeps or architecture comparisons.

Usage:
    # Run all experiments
    python train_multiple_experiments.py

    # Run specific experiments by index (1-based)
    python train_multiple_experiments.py --experiments 1 3 5

    # Run a range of experiments
    python train_multiple_experiments.py --experiments 2 3 4
"""

from datetime import date, timedelta
import warnings
import argparse

from loguru import logger

from features import (
    InputSpace,
    FeaturePipeline,
    FeatureExtractorRegistry,
    DirectionalLabel,
)
from models import (
    ModelConfig,
    LSTMConfig,
    TransformerConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
    train_model,
)
from utils import date_range, get_ob_path
from core.orderbook import OrderBook

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # noqa: B028


def load_data(first_date: date, n_days: int, sample_interval: int = 100_000_000):
    """
    Load and preprocess orderbook data.

    Args:
        first_date: First date to load
        n_days: Number of days to load
        sample_interval: Sampling interval in nanoseconds

    Returns:
        Tuple of (features, labels, label_counts, total_samples)
    """
    logger.info(f"Loading orderbook data from {first_date} for {n_days} day(s)...")

    ob_paths = [
        get_ob_path(d) for d in date_range(first_date, first_date + timedelta(days=n_days - 1))
    ]
    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(5)
        .sample_by_time(time_delta=sample_interval, interpolate=True)
    )

    # Skip first 10 samples (first second) via private attribute — OrderBook has no
    # public slice API that rewrites the internal frame without re-validation.
    orderbook_data._data._df = orderbook_data.df[10:]

    logger.info("Creating input space and feature pipeline...")

    input_space = InputSpace(orderbook_snapshots=orderbook_data)

    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(FeatureExtractorRegistry.create("advanced_orderbook"))

    directional_return_label = DirectionalLabel(config={"horizon": 200, "threshold": 3e-4})

    logger.info("Extracting features and labels...")

    features = feature_pipeline.extract_all(input_space)
    labels = directional_return_label.extract(input_space)

    logger.info(f"Features shape: {features.collect().shape}")
    logger.info(f"Labels shape: {labels.collect().shape}")

    labels_df = labels.collect()
    label_counts = labels_df[directional_return_label.label_names[0]].value_counts()
    total_samples = len(labels_df)

    logger.info("Label distribution:")
    for label_value, count in label_counts.iter_rows():
        percentage = (count / total_samples) * 100
        logger.info(f"  Label {label_value}: {count:,} samples ({percentage:.2f}%)")

    return features, labels, label_counts, total_samples


def calculate_class_weights(label_counts, total_samples):
    """
    Calculate class weights based on label distribution.

    Args:
        label_counts: Label value counts
        total_samples: Total number of samples

    Returns:
        List of class weights
    """
    class_weights = [0.0, 0.0, 0.0]
    for label_value, count in label_counts.iter_rows():
        freq = count / total_samples
        class_weights[label_value + 1] = 1 / freq
    return class_weights


def create_experiment_configs(
    input_size: int, class_weights: list, use_tqdm: bool = True
) -> list[ModelConfig]:
    """
    Define all experiment configurations here.

    Args:
        input_size: Number of input features
        class_weights: Class weights for loss function
        use_tqdm: Whether to enable tqdm progress bars

    Returns:
        List of ModelConfig objects to train sequentially
    """

    experiments = []

    data_config = DataConfig(
        sequence_length=256,
        batch_size=512,
        stride=5,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        device="cuda",
    )
    train_config = TrainingConfig(
        learning_rate=0.001,
        num_epochs=100,
        early_stopping_patience=10,
        optimizer="adam",
        scheduler="cosine",
        loss_function="focal",
        loss_params={"alpha": class_weights, "gamma": 1.0},
        mixed_precision=True,
        use_tqdm=use_tqdm,
    )

    def get_logging_config(name: str) -> LoggingConfig:
        return LoggingConfig(
            project_name="feat-model",
            experiment_name=f"{name}_alpha=5e-4",
            wandb_enabled=True,
            log_confusion_matrix=True,
            log_trade_accuracy_vs_threshold=True,
        )

    # Experiment 1: small LSTM
    experiments.append(
        ModelConfig(
            architecture=LSTMConfig(
                input_size=input_size,
                hidden_size=64,
                num_layers=1,
                output_size=3,
                dropout=0.2,
                bidirectional=True,
                attention=False,
            ),
            data=data_config,
            training=train_config,
            logging=get_logging_config("small_lstm_no_attention_seq64"),
        )
    )

    # Experiment 2: small LSTM wider
    experiments.append(
        ModelConfig(
            architecture=LSTMConfig(
                input_size=input_size,
                hidden_size=128,
                num_layers=1,
                output_size=3,
                dropout=0.2,
                bidirectional=True,
                attention=False,
            ),
            data=data_config,
            training=train_config,
            logging=get_logging_config("small_lstm_no_attention_seq128"),
        )
    )

    # Experiment 3: LSTM with attention
    experiments.append(
        ModelConfig(
            architecture=LSTMConfig(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                output_size=3,
                dropout=0.2,
                bidirectional=True,
                attention=True,
            ),
            data=data_config,
            training=train_config,
            logging=get_logging_config("lstm_attention_seq256"),
        )
    )

    # Experiment 4: LSTM without attention
    experiments.append(
        ModelConfig(
            architecture=LSTMConfig(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                output_size=3,
                dropout=0.2,
                bidirectional=True,
                attention=False,
            ),
            data=data_config,
            training=train_config,
            logging=get_logging_config("lstm_no_attention_seq256"),
        )
    )

    # Experiment 5: Transformer
    experiments.append(
        ModelConfig(
            architecture=TransformerConfig(
                input_size=input_size,
                d_model=64,
                nhead=8,
                num_layers=3,
                output_size=3,
                dropout=0.2,
                dim_feedforward=256,
            ),
            data=DataConfig(
                sequence_length=256,
                batch_size=512,
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                device="cuda",
            ),
            training=TrainingConfig(
                learning_rate=0.001,
                num_epochs=50,
                early_stopping_patience=5,
                optimizer="adamw",
                scheduler="cosine",
                loss_function="focal",
                loss_params={"alpha": class_weights, "gamma": 2.0},
                mixed_precision=True,
                use_tqdm=use_tqdm,
            ),
            logging=get_logging_config("transformer_seq256"),
        )
    )

    # Experiment 6: LSTM with longer sequence
    experiments.append(
        ModelConfig(
            architecture=LSTMConfig(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                output_size=3,
                dropout=0.2,
                bidirectional=True,
                attention=True,
            ),
            data=DataConfig(
                sequence_length=512,
                batch_size=512,
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                device="cuda",
            ),
            training=train_config,
            logging=get_logging_config("lstm_attention_seq512"),
        )
    )

    return experiments


def run_experiments(
    features,
    labels,
    experiments: list[ModelConfig],
    selected_indices: list[int] | None = None,
    feature_columns=None,
    label_columns=None,
):
    """
    Run all experiments sequentially.

    Args:
        features: Feature data
        labels: Label data
        experiments: List of ModelConfig objects
        selected_indices: Optional list of experiment indices (1-based) to run. If None, run all.
        feature_columns: Optional list of feature column names
        label_columns: Optional list of label column names

    Returns:
        List of (trainer, results) tuples for each experiment
    """
    if selected_indices is not None:
        invalid_indices = [i for i in selected_indices if i < 1 or i > len(experiments)]
        if invalid_indices:
            raise ValueError(
                f"Invalid experiment indices: {invalid_indices}. "
                f"Valid range is 1-{len(experiments)}"
            )

        experiments_to_run = [(i, experiments[i - 1]) for i in sorted(selected_indices)]
        logger.info(f"Running {len(experiments_to_run)} selected experiments: {selected_indices}")
    else:
        experiments_to_run = [(i, exp) for i, exp in enumerate(experiments, 1)]
        logger.info(f"Running all {len(experiments)} experiments sequentially")

    results_list = []

    for experiment_num, (idx, config) in enumerate(experiments_to_run, 1):
        logger.info(
            f"EXPERIMENT {experiment_num}/{len(experiments_to_run)} "
            f"(#{idx}): {config.get_logging_config().experiment_name}"
        )
        logger.info(f"Architecture: {config.get_architecture_config().__class__.__name__}")
        logger.info(f"Sequence Length: {config.get_data_config().sequence_length}")
        logger.info(f"Batch Size: {config.get_data_config().batch_size}")
        logger.info(f"Learning Rate: {config.get_training_config().learning_rate}")
        logger.info(f"Optimizer: {config.get_training_config().optimizer}")
        logger.info(f"Max Epochs: {config.get_training_config().num_epochs}")

        try:
            trainer, results = train_model(
                features=features,
                labels=labels,
                config=config,
                feature_columns=feature_columns,
                label_columns=label_columns,
            )

            results_list.append((idx, trainer, results))
            logger.info(f"EXPERIMENT {experiment_num} (#{idx}) COMPLETED SUCCESSFULLY")

        except Exception as e:
            logger.error(f"EXPERIMENT {experiment_num} (#{idx}) FAILED WITH ERROR: {e}")
            results_list.append((idx, None, None))
            continue

    return results_list


def print_summary(results_list: list, experiments: list[ModelConfig]) -> None:
    """
    Print summary of all experiments.

    Args:
        results_list: List of (idx, trainer, results) tuples
        experiments: List of ModelConfig objects
    """
    logger.info("EXPERIMENT SUMMARY")

    for idx, trainer, results in results_list:
        config = experiments[idx - 1]
        exp_name = config.get_logging_config().experiment_name

        if results is None:
            logger.info(f"{idx}. {exp_name}: FAILED")
        else:
            test_acc = results.get("test_accuracy", "N/A")
            test_trade_acc = results.get("test_trade_accuracy", "N/A")
            test_strict_trade_acc = results.get("test_strict_trade_accuracy", "N/A")

            logger.info(f"{idx}. {exp_name}:")
            logger.info(
                f"   Test Accuracy: {test_acc:.4f}"
                if test_acc != "N/A"
                else f"   Test Accuracy: {test_acc}"
            )
            logger.info(
                f"   Trade Accuracy: {test_trade_acc:.4f}"
                if test_trade_acc != "N/A"
                else f"   Trade Accuracy: {test_trade_acc}"
            )
            logger.info(
                f"   Strict Trade Accuracy: {test_strict_trade_acc:.4f}"
                if test_strict_trade_acc != "N/A"
                else f"   Strict Trade Accuracy: {test_strict_trade_acc}"
            )


def main() -> None:
    """Main function for running multiple experiments."""

    parser = argparse.ArgumentParser(
        description="Run multiple model training experiments sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python train_multiple_experiments.py

  # Run specific experiments by index (1-based)
  python train_multiple_experiments.py --experiments 1 3 5

  # Run a range of experiments
  python train_multiple_experiments.py --experiments 2 3 4

  # List available experiments without running
  python train_multiple_experiments.py --list

  # Run experiments without tqdm progress bars
  python train_multiple_experiments.py --no-tqdm
        """,
    )
    parser.add_argument(
        "--experiments",
        "-e",
        type=int,
        nargs="+",
        metavar="INDEX",
        help="Indices of experiments to run (1-based). If not specified, all experiments will run.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available experiments and exit without running.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars during training.",
    )

    args = parser.parse_args()

    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 10
    SAMPLE_INTERVAL = 100_000_000  # nanoseconds

    features, labels, label_counts, total_samples = load_data(
        first_date=FIRST_DATE,
        n_days=N_DAYS,
        sample_interval=SAMPLE_INTERVAL,
    )

    class_weights = calculate_class_weights(label_counts, total_samples)

    input_size = len(features.columns) - 1  # exclude timestamp

    experiments = create_experiment_configs(input_size, class_weights, use_tqdm=not args.no_tqdm)

    logger.info(f"Total experiments defined: {len(experiments)}")

    if args.list:
        logger.info("Available experiments:")
        for i, config in enumerate(experiments, 1):
            arch = config.get_architecture_config().__class__.__name__
            exp_name = config.get_logging_config().experiment_name
            seq_len = config.get_data_config().sequence_length
            batch_size = config.get_data_config().batch_size
            lr = config.get_training_config().learning_rate
            logger.info(f"  {i}. {exp_name}")
            logger.info(f"     Architecture: {arch}")
            logger.info(f"     Sequence Length: {seq_len}, Batch Size: {batch_size}, LR: {lr}")
        return

    results_list = run_experiments(
        features=features,
        labels=labels,
        experiments=experiments,
        selected_indices=args.experiments,
        feature_columns=None,
        label_columns=None,
    )

    print_summary(results_list, experiments)

    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
