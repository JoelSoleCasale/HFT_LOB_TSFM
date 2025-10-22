"""
Data handling and preprocessing for model training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import polars as pl
import numpy as np
from typing import Tuple, Optional, List
from loguru import logger
from sklearn.preprocessing import StandardScaler
from .config import DataConfig


class FinancialDataset(Dataset):
    """
    PyTorch Dataset for financial time series data.

    This dataset handles features and labels from the feature extraction pipeline,
    creating sequences for time series modeling.
    """

    def __init__(
        self,
        features: pl.LazyFrame,
        labels: pl.LazyFrame,
        sequence_length: int = 10,
        feature_columns: Optional[List[str]] = None,
        label_columns: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            features: Polars LazyFrame containing features
            labels: Polars LazyFrame containing labels
            sequence_length: Length of input sequences
            feature_columns: List of feature column names to use
            label_columns: List of label column names to use
            scaler: Pre-fitted scaler for features
            fit_scaler: Whether to fit the scaler on the data
        """
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.label_columns = label_columns

        # Collect the data
        features_df = features.collect()
        labels_df = labels.collect()

        # Ensure dataframes are aligned by timestamp
        if "timestamp" in features_df.columns and "timestamp" in labels_df.columns:
            # Join on timestamp to ensure alignment
            combined = features_df.join(labels_df, on="timestamp", how="inner")
            features_df = combined.select(features_df.columns)
            labels_df = combined.select(labels_df.columns)

        # Select feature columns
        if feature_columns is None:
            # Exclude timestamp and other non-feature columns
            exclude_cols = ["timestamp", "time"]
            self.feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        else:
            self.feature_columns = feature_columns

        # Select label columns
        if label_columns is None:
            # Use all label columns
            self.label_columns = [
                col for col in labels_df.columns if col not in ["timestamp", "time"]
            ]
        else:
            self.label_columns = label_columns

        # Extract feature data
        features_array = features_df.select(self.feature_columns).to_numpy()
        labels_array = labels_df.select(self.label_columns).to_numpy()

        # Handle missing values by dropping rows with any nulls/NaNs across features and labels together
        self.features, self.labels = self._handle_missing_values(features_array, labels_array)

        # Map classification labels to contiguous class indices if single-column labels are provided
        # Expecting directional labels in {-1, 0, 1} → map to {0, 1, 2}
        if self.labels.ndim == 2 and self.labels.shape[1] == 1:
            labels_1d = self.labels.squeeze(1)
            unique_vals = np.unique(labels_1d[~np.isnan(labels_1d)])
            # Only apply mapping if labels look like directional labels
            if set(unique_vals.tolist()).issuperset({-1, 0, 1}) or set(
                unique_vals.tolist()
            ).issubset({-1, 0, 1}):
                mapping = {-1: 0, 0: 1, 1: 2}
                # Vectorized mapping; for any unexpected values, fall back to 1 (neutral)
                mapped = np.vectorize(lambda v: mapping.get(int(v), 1))(labels_1d)
                self.labels = mapped.astype(np.int64)
            else:
                # Otherwise, try casting to integers safely
                self.labels = labels_1d.astype(np.int64)
        elif self.labels.ndim == 1:
            # Already 1D labels
            self.labels = self.labels.astype(np.int64)

        # Scale features
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.features = self.scaler.transform(self.features)

        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        # Ensure labels are 1D LongTensor of class indices for CrossEntropyLoss
        if isinstance(self.labels, np.ndarray) and self.labels.ndim > 1:
            self.labels = torch.LongTensor(self.labels.squeeze(-1))
        else:
            self.labels = torch.LongTensor(self.labels)

        # Calculate valid sequence indices
        self.valid_indices = self._get_valid_sequence_indices()

        logger.info(f"Dataset created with {len(self.valid_indices)} valid sequences")
        logger.info(f"Features shape: {self.features.shape}")
        logger.info(f"Labels shape: {self.labels.shape}")
        logger.info(f"Feature columns: {self.feature_columns}")
        logger.info(f"Label columns: {self.label_columns}")

    def _handle_missing_values(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing values by dropping rows with any nulls/NaNs across features and labels together.

        Args:
            features: NumPy array of feature data
            labels: NumPy array of label data

        Returns:
            Tuple of (filtered_features, filtered_labels) with missing value rows removed
        """
        # Create boolean mask for rows with missing values in features
        features_has_nan = np.isnan(features).any(axis=1)
        labels_has_nan = np.isnan(labels).any(axis=1)

        # Invert to get mask of valid rows (no missing values)
        valid_rows_mask = ~(features_has_nan | labels_has_nan)

        before_rows = len(features)
        after_rows = valid_rows_mask.sum()

        if after_rows < before_rows:
            logger.info(
                f"Dropped {before_rows - after_rows} rows with missing values "
                f"(from {before_rows} to {after_rows})."
            )

        return features[valid_rows_mask], labels[valid_rows_mask]

    def _get_valid_sequence_indices(self) -> List[int]:
        """Get indices where we can create valid sequences."""
        valid_indices = []
        for i in range(len(self.features) - self.sequence_length + 1):
            # Check if the sequence has any NaN values
            if not torch.isnan(self.features[i : i + self.sequence_length]).any():
                valid_indices.append(i)
        return valid_indices

    def __len__(self) -> int:
        """Return the number of valid sequences."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its corresponding label.

        Args:
            idx: Index of the sequence

        Returns:
            Tuple of (sequence, label)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        # Get the sequence
        sequence = self.features[start_idx:end_idx]

        # Get the label (use the last label in the sequence)
        label = self.labels[end_idx - 1]  # tensor scalar of class index

        return sequence, label

    def get_feature_names(self) -> List[str]:
        """Get the names of feature columns."""
        return self.feature_columns

    def get_label_names(self) -> List[str]:
        """Get the names of label columns."""
        return self.label_columns

    def get_scaler(self) -> StandardScaler:
        """Get the fitted scaler."""
        return self.scaler


def create_dataloaders(
    features: pl.LazyFrame,
    labels: pl.LazyFrame,
    data_config: DataConfig,
    feature_columns: Optional[List[str]] = None,
    label_columns: Optional[List[str]] = None,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Create train, validation, and test dataloaders.

    Args:
        features: Polars LazyFrame containing features
        labels: Polars LazyFrame containing labels
        data_config: Data configuration
        feature_columns: List of feature column names to use
        label_columns: List of label column names to use
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Create the full dataset
    full_dataset = FinancialDataset(
        features=features,
        labels=labels,
        sequence_length=data_config.sequence_length,
        feature_columns=feature_columns,
        label_columns=label_columns,
        fit_scaler=True,
    )

    # Get the scaler
    scaler = full_dataset.get_scaler()

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(data_config.train_split * total_size)
    val_size = int(data_config.val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    logger.info(
        f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, scaler


def prepare_data_for_training(
    features: pl.LazyFrame,
    labels: pl.LazyFrame,
    data_config: DataConfig,
    feature_columns: Optional[List[str]] = None,
    label_columns: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, List[str], List[str]]:
    """
    Prepare data for training with automatic column selection.

    Args:
        features: Polars LazyFrame containing features
        labels: Polars LazyFrame containing labels
        data_config: Data configuration
        feature_columns: Optional list of feature columns to use
        label_columns: Optional list of label columns to use

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler, feature_names, label_names)
    """
    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        features, labels, data_config, feature_columns, label_columns
    )

    # Get feature and label names from the first batch
    sample_batch = next(iter(train_loader))
    sample_sequence, sample_label = sample_batch

    # Create a temporary dataset to get column names
    temp_dataset = FinancialDataset(
        features=features,
        labels=labels,
        sequence_length=data_config.sequence_length,
        feature_columns=feature_columns,
        label_columns=label_columns,
        scaler=scaler,
        fit_scaler=False,
    )

    feature_names = temp_dataset.get_feature_names()
    label_names = temp_dataset.get_label_names()

    return train_loader, val_loader, test_loader, scaler, feature_names, label_names
