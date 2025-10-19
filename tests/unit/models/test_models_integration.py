"""
Integration tests for the models module.
"""

import pytest
import torch
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from models import ModelConfig, create_model, FinancialDataset
from models.data import create_dataloaders


class TestModelsIntegration:
    """Test the models module integration."""

    def test_model_creation(self):
        """Test that models can be created successfully."""
        # Test MLP model
        mlp_model = create_model(
            "mlp", input_size=5, sequence_length=10, hidden_sizes=[32, 16], output_size=3
        )
        assert isinstance(mlp_model, torch.nn.Module)

        # Test LSTM model
        lstm_model = create_model(
            "lstm", input_size=5, hidden_size=32, num_layers=2, output_size=3
        )
        assert isinstance(lstm_model, torch.nn.Module)

        # Test Transformer model
        transformer_model = create_model(
            "transformer", input_size=5, d_model=32, nhead=4, num_layers=2, output_size=3
        )
        assert isinstance(transformer_model, torch.nn.Module)

    def test_model_forward_pass(self):
        """Test that models can perform forward pass."""
        batch_size = 4
        sequence_length = 10
        input_size = 5
        output_size = 3

        # Create test input
        x = torch.randn(batch_size, sequence_length, input_size)

        # Test MLP
        mlp_model = create_model(
            "mlp", input_size=input_size, sequence_length=sequence_length, output_size=output_size
        )
        mlp_output = mlp_model(x)
        assert mlp_output.shape == (batch_size, output_size)

        # Test LSTM
        lstm_model = create_model(
            "lstm", input_size=input_size, hidden_size=32, output_size=output_size
        )
        lstm_output = lstm_model(x)
        assert lstm_output.shape == (batch_size, output_size)

        # Test Transformer
        transformer_model = create_model(
            "transformer", input_size=input_size, d_model=32, output_size=output_size
        )
        transformer_output = transformer_model(x)
        assert transformer_output.shape == (batch_size, output_size)

    def test_financial_dataset(self):
        """Test FinancialDataset creation and usage."""
        # Create sample data
        n_samples = 100
        n_features = 3
        n_labels = 1

        features_data = np.random.randn(n_samples, n_features)
        labels_data = np.random.randint(0, 3, (n_samples, n_labels))

        # Create Polars DataFrames
        features_df = pl.DataFrame(
            {
                "feature_1": features_data[:, 0],
                "feature_2": features_data[:, 1],
                "feature_3": features_data[:, 2],
            }
        )

        labels_df = pl.DataFrame(
            {
                "label": labels_data[:, 0],
            }
        )

        # Create dataset
        dataset = FinancialDataset(
            features=features_df.lazy(),
            labels=labels_df.lazy(),
            sequence_length=5,
            feature_columns=["feature_1", "feature_2", "feature_3"],
            label_columns=["label"],
        )

        # Test dataset properties
        assert len(dataset) > 0
        assert len(dataset) <= n_samples - 5 + 1  # Account for sequence length

        # Test getting a sample
        sequence, label = dataset[0]
        assert sequence.shape == (5, 3)  # (sequence_length, n_features)
        assert label.shape == (1,)  # (n_labels,)

    def test_model_config(self):
        """Test ModelConfig functionality."""
        # Test basic config creation
        config = ModelConfig(model_type="lstm", hidden_size=64, num_epochs=10)

        assert config.model_type == "lstm"
        assert config.hidden_size == 64
        assert config.num_epochs == 10

        # Test config validation
        with pytest.raises(ValueError):
            ModelConfig(model_type="invalid_model")

        with pytest.raises(ValueError):
            ModelConfig(train_split=0.5, val_split=0.3, test_split=0.1)  # Doesn't sum to 1.0

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["model_type"] == "lstm"

        # Test from_dict
        new_config = ModelConfig.from_dict(config_dict)
        assert new_config.model_type == config.model_type
        assert new_config.hidden_size == config.hidden_size

    def test_dataloader_creation(self):
        """Test dataloader creation."""
        # Create sample data
        n_samples = 200
        n_features = 3
        n_labels = 1

        features_data = np.random.randn(n_samples, n_features)
        labels_data = np.random.randint(0, 3, (n_samples, n_labels))

        # Create Polars DataFrames
        features_df = pl.DataFrame(
            {
                "feature_1": features_data[:, 0],
                "feature_2": features_data[:, 1],
                "feature_3": features_data[:, 2],
            }
        )

        labels_df = pl.DataFrame(
            {
                "label": labels_data[:, 0],
            }
        )

        # Create config
        config = ModelConfig(
            sequence_length=5, batch_size=16, train_split=0.7, val_split=0.2, test_split=0.1
        )

        # Create dataloaders
        train_loader, val_loader, test_loader, scaler = create_dataloaders(
            features_df.lazy(), labels_df.lazy(), config
        )

        # Test dataloaders
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

        # Test getting a batch
        batch = next(iter(train_loader))
        sequences, labels = batch
        assert sequences.shape[0] <= config.batch_size
        assert sequences.shape[1] == config.sequence_length
        assert sequences.shape[2] == n_features
        assert labels.shape[1] == n_labels


if __name__ == "__main__":
    pytest.main([__file__])
