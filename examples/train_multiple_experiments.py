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

from pathlib import Path
from datetime import date, timedelta
import warnings
import argparse

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
from utils import date_range
from definitions import ROOT_DIR
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
    print(f"\n{'='*80}")
    print(f"Loading orderbook data from {first_date} for {n_days} day(s)...")
    print(f"{'='*80}")

    # Load orderbook data
    def get_ob_path(d: date) -> Path:
        return (
            ROOT_DIR
            / f'data/orderbook_snapshots/binance_futures/BTCUSDT/{d.strftime("%Y-%m-%d")}_L20.parquet'
        )

    ob_paths = [
        get_ob_path(d) for d in date_range(first_date, first_date + timedelta(days=n_days - 1))
    ]
    orderbook_data = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(5)
        .sample_by_time(time_delta=sample_interval, interpolate=True)
    )

    # Ignore first second
    orderbook_data._data._df = orderbook_data.df[10:]

    print("Creating input space and feature pipeline...")

    # Create input space
    input_space = InputSpace(orderbook_snapshots=orderbook_data)

    # Build feature pipeline
    feature_pipeline = FeaturePipeline()
    feature_pipeline.add_extractor(FeatureExtractorRegistry.create("advanced_orderbook"))

    # Build label pipeline
    directional_return_label = DirectionalLabel(config={"horizon": 200, "threshold": 3e-4})

    print("Extracting features and labels...")

    # Extract features and labels
    features = feature_pipeline.extract_all(input_space)
    labels = directional_return_label.extract(input_space)

    print(f"Features shape: {features.collect().shape}")
    print(f"Labels shape: {labels.collect().shape}")

    # Calculate label distribution
    labels_df = labels.collect()
    label_counts = labels_df[directional_return_label.label_names[0]].value_counts()
    total_samples = len(labels_df)

    print("\nLabel distribution:")
    for label_value, count in label_counts.iter_rows():
        percentage = (count / total_samples) * 100
        print(f"  Label {label_value}: {count:,} samples ({percentage:.2f}%)")

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
        batch_size=1024,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        device="cuda",
    )
    train_config = TrainingConfig(
        learning_rate=0.005,
        num_epochs=50,
        early_stopping_patience=5,
        optimizer="adam",
        scheduler="cosine",
        loss_function="focal",
        loss_params={"alpha": class_weights, "gamma": 2.0},
        mixed_precision=True,
        use_tqdm=use_tqdm,
    )

    def get_logging_config(name: str) -> LoggingConfig:
        return LoggingConfig(
            project_name="financial-models",
            experiment_name=f"{name}_alpha=5e-4",
            wandb_enabled=True,
            log_confusion_matrix=True,
            log_trade_accuracy_vs_threshold=True,
        )

    # Experiment 1: LSTM with attention
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

    # Experiment 2: LSTM without attention
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

    # Experiment 3: Transformer
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

    # Experiment 4: LSTM with longer sequence
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

    # # Experiment 5: Deep LSTM
    # experiments.append(
    #     ModelConfig(
    #         architecture=LSTMConfig(
    #             input_size=input_size,
    #             hidden_size=128,
    #             num_layers=4,
    #             output_size=3,
    #             dropout=0.3,
    #             bidirectional=True,
    #             attention=True,
    #         ),
    #         data=data_config,
    #         training=train_config,
    #         logging=get_logging_config("lstm_deep4_attention_seq256"),
    #     )
    # )

    # # Experiment 6: MLP
    # experiments.append(
    #     ModelConfig(
    #         architecture=MLPConfig(
    #             input_size=input_size * data_config.sequence_length,  # Flattened input
    #             output_size=3,
    #             hidden_sizes=[256, 128, 64],
    #             dropout=0.2,
    #         ),
    #         data=data_config,
    #         training=train_config,
    #         logging=get_logging_config("mlp_seq1"),
    #     )
    # )

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
    # Determine which experiments to run
    if selected_indices is not None:
        # Validate indices
        invalid_indices = [i for i in selected_indices if i < 1 or i > len(experiments)]
        if invalid_indices:
            raise ValueError(
                f"Invalid experiment indices: {invalid_indices}. "
                f"Valid range is 1-{len(experiments)}"
            )

        experiments_to_run = [(i, experiments[i - 1]) for i in sorted(selected_indices)]
        print(f"\n{'='*80}")
        print(f"Running {len(experiments_to_run)} selected experiments: {selected_indices}")
        print(f"{'='*80}\n")
    else:
        experiments_to_run = [(i, exp) for i, exp in enumerate(experiments, 1)]
        print(f"\n{'='*80}")
        print(f"Running all {len(experiments)} experiments sequentially")
        print(f"{'='*80}\n")

    results_list = []

    for experiment_num, (idx, config) in enumerate(experiments_to_run, 1):
        print(f"\n{'='*80}")
        print(
            f"EXPERIMENT {experiment_num}/{len(experiments_to_run)} "
            f"(#{idx}): {config.get_logging_config().experiment_name}"
        )
        print(f"{'='*80}")
        print(f"Architecture: {config.get_architecture_config().__class__.__name__}")
        print(f"Sequence Length: {config.get_data_config().sequence_length}")
        print(f"Batch Size: {config.get_data_config().batch_size}")
        print(f"Learning Rate: {config.get_training_config().learning_rate}")
        print(f"Optimizer: {config.get_training_config().optimizer}")
        print(f"Max Epochs: {config.get_training_config().num_epochs}")
        print(f"{'='*80}\n")

        try:
            # Train the model
            trainer, results = train_model(
                features=features,
                labels=labels,
                config=config,
                feature_columns=feature_columns,
                label_columns=label_columns,
            )

            results_list.append((idx, trainer, results))

            print(f"\n{'='*80}")
            print(f"EXPERIMENT {experiment_num} (#{idx}) COMPLETED SUCCESSFULLY")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {experiment_num} (#{idx}) FAILED WITH ERROR:")
            print(f"{str(e)}")
            print(f"{'='*80}\n")
            results_list.append((idx, None, None))
            continue

    return results_list


def print_summary(results_list: list, experiments: list[ModelConfig]):
    """
    Print summary of all experiments.

    Args:
        results_list: List of (idx, trainer, results) tuples
        experiments: List of ModelConfig objects
    """
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    for idx, trainer, results in results_list:
        config = experiments[idx - 1]
        exp_name = config.get_logging_config().experiment_name

        if results is None:
            print(f"{idx}. {exp_name}: FAILED")
        else:
            test_acc = results.get("test_accuracy", "N/A")
            test_trade_acc = results.get("test_trade_accuracy", "N/A")
            test_strict_trade_acc = results.get("test_strict_trade_accuracy", "N/A")

            print(f"{idx}. {exp_name}:")
            print(
                f"   Test Accuracy: {test_acc:.4f}"
                if test_acc != "N/A"
                else f"   Test Accuracy: {test_acc}"
            )
            print(
                f"   Trade Accuracy: {test_trade_acc:.4f}"
                if test_trade_acc != "N/A"
                else f"   Trade Accuracy: {test_trade_acc}"
            )
            print(
                f"   Strict Trade Accuracy: {test_strict_trade_acc:.4f}"
                if test_strict_trade_acc != "N/A"
                else f"   Strict Trade Accuracy: {test_strict_trade_acc}"
            )
            print()

    print(f"{'='*80}\n")


def main():
    """Main function for running multiple experiments."""

    # Parse command-line arguments
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

    # Data loading configuration
    FIRST_DATE = date(2025, 7, 1)
    N_DAYS = 1
    SAMPLE_INTERVAL = 100_000_000  # nanoseconds

    # Load data once (shared across all experiments)
    features, labels, label_counts, total_samples = load_data(
        first_date=FIRST_DATE,
        n_days=N_DAYS,
        sample_interval=SAMPLE_INTERVAL,
    )

    # Calculate class weights
    class_weights = calculate_class_weights(label_counts, total_samples)

    # Get input size
    input_size = len(features.columns) - 1  # exclude timestamp

    # Create experiment configurations
    experiments = create_experiment_configs(input_size, class_weights, use_tqdm=not args.no_tqdm)

    print(f"\n{'='*80}")
    print(f"Total experiments defined: {len(experiments)}")
    print(f"{'='*80}")

    # List experiments if requested
    if args.list:
        print("\nAvailable experiments:")
        for i, config in enumerate(experiments, 1):
            arch = config.get_architecture_config().__class__.__name__
            exp_name = config.get_logging_config().experiment_name
            seq_len = config.get_data_config().sequence_length
            batch_size = config.get_data_config().batch_size
            lr = config.get_training_config().learning_rate
            print(f"  {i}. {exp_name}")
            print(f"     Architecture: {arch}")
            print(f"     Sequence Length: {seq_len}, Batch Size: {batch_size}, LR: {lr}")
        print(f"\n{'='*80}\n")
        return

    # Run all experiments or selected ones
    results_list = run_experiments(
        features=features,
        labels=labels,
        experiments=experiments,
        selected_indices=args.experiments,
        feature_columns=None,
        label_columns=None,
    )

    # Print summary
    print_summary(results_list, experiments)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
