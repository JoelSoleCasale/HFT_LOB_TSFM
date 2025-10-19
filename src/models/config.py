"""
Configuration management for model training.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""

    # Model architecture
    model_type: str = "lstm"  # "lstm", "transformer", "mlp"
    input_size: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    output_size: int = 3  # For classification: -1, 0, 1

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5

    # Data parameters
    sequence_length: int = 10
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Logging and saving
    project_name: str = "financial-models"
    experiment_name: Optional[str] = None
    log_file: str = "training.log"
    model_save_path: str = "models/trained_model.pth"
    checkpoint_dir: str = "models/checkpoints"

    # Wandb configuration
    wandb_enabled: bool = True
    wandb_project: str = "financial-models"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = None

    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.wandb_tags is None:
            self.wandb_tags = []

        # Ensure splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Train, validation, and test splits must sum to 1.0, got {total_split}"
            )

        # Validate model type
        valid_models = ["lstm", "transformer", "mlp"]
        if self.model_type not in valid_models:
            raise ValueError(f"Model type must be one of {valid_models}, got {self.model_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for wandb logging."""
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "weight_decay": self.weight_decay,
            "sequence_length": self.sequence_length,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

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
