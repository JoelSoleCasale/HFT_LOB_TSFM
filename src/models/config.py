"""
Configuration management for model training.
"""

from dataclasses import dataclass, field
from pathlib import Path
import yaml
from definitions import ROOT_DIR
from .architectures.base import ModelArchitectureConfig
from .architectures.mlp import MLPConfig
from .architectures.lstm import LSTMConfig
from .architectures.transformer import TransformerConfig


@dataclass
class DataConfig:
    """Configuration for data processing and splitting."""

    sequence_length: int = 10
    stride: int = 1
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    device: str = "cpu"
    num_classes: int = 3
    class_names: list[str] | None = field(default_factory=lambda: ["-1", "0", "1"])

    def __post_init__(self):
        """Validate data configuration."""
        # Ensure splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Train, validation, and test splits must sum to 1.0, got {total_split}"
            )


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    optimizer: str = "adam"  # "adam", "sgd", "adamw", "rmsprop"
    scheduler: str | None = None  # "cosine", "step", "plateau", None
    scheduler_params: dict[str, object] = field(default_factory=dict)
    loss_function: str = "cross_entropy"  # "cross_entropy", "mse", "mae", "focal"
    loss_params: dict[str, object] = field(default_factory=dict)
    gradient_clip_norm: float | None = None
    mixed_precision: bool = False
    use_tqdm: bool = True  # Enable/disable tqdm progress bars during training


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""

    project_name: str = "financial-models"
    experiment_name: str | None = None
    log_file: str = str(ROOT_DIR / "logs/training.log")
    model_save_path: str = str(ROOT_DIR / "model_checkpoint/trained_model.pth")
    checkpoint_dir: str = str(ROOT_DIR / "model_checkpoint/checkpoints")

    # Wandb configuration
    wandb_enabled: bool = True
    wandb_project: str = "financial-models"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    # Metrics logging
    log_metrics_frequency: int = 1  # Log every N epochs
    log_confusion_matrix: bool = True
    log_trade_accuracy_vs_threshold: bool = True
    log_predictions: bool = False

    # Expected return parameters
    lambda_value: float = 5e-4  # Horizontal barrier distance for expected return
    theta_values: list[float] = field(default_factory=lambda: [0.0, 5e-4])  # Commission rates


@dataclass
class ModelConfig:
    """Main configuration class that combines all sub-configurations."""

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    architecture: ModelArchitectureConfig = field(default_factory=LSTMConfig)
    other: dict[str, object] = field(default_factory=dict)

    # Device configuration
    device: str = "auto"
    seed: int = 42

    def __post_init__(self):
        """Validate the complete configuration."""
        # Validate architecture type matches the config type
        if isinstance(self.architecture, LSTMConfig) and self.architecture.model_type != "lstm":
            raise ValueError("LSTMConfig must have model_type='lstm'")
        elif (
            isinstance(self.architecture, TransformerConfig)
            and self.architecture.model_type != "transformer"
        ):
            raise ValueError("TransformerConfig must have model_type='transformer'")
        elif isinstance(self.architecture, MLPConfig) and self.architecture.model_type != "mlp":
            raise ValueError("MLPConfig must have model_type='mlp'")

    def to_dict(self) -> dict[str, object]:
        """Convert configuration to dictionary for logging."""
        return {
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
            "architecture": self.architecture.__dict__,
            "other": self.other,
            "device": self.device,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, object]) -> "ModelConfig":
        """Create configuration from dictionary."""
        from .factories import create_architecture_config

        # Extract sub-configurations
        data_config = DataConfig(**config_dict.get("data", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))

        arch_dict = config_dict.get("architecture", {})
        model_type = arch_dict.get("model_type", "lstm")
        architecture = create_architecture_config(model_type, arch_dict)
        other = config_dict.get("other", {})

        return cls(
            data=data_config,
            training=training_config,
            logging=logging_config,
            architecture=architecture,
            other=other,
            device=config_dict.get("device", "auto"),
            seed=config_dict.get("seed", 42),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def get_architecture_config(self) -> ModelArchitectureConfig:
        """Get the architecture configuration."""
        return self.architecture

    def get_training_config(self) -> TrainingConfig:
        """Get the training configuration."""
        return self.training

    def get_data_config(self) -> DataConfig:
        """Get the data configuration."""
        return self.data

    def get_logging_config(self) -> LoggingConfig:
        """Get the logging configuration."""
        return self.logging
