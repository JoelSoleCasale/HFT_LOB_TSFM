"""
Data handling and preprocessing for model training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import polars as pl
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from .config import DataConfig
from models.utils import get_device


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
        stride: int = 1,
        feature_columns: list[str] | None = None,
        label_columns: list[str] | None = None,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
        device: str = "auto",
    ):
        """
        Initialize the dataset.

        Args:
            features: Polars LazyFrame containing features
            labels: Polars LazyFrame containing labels
            sequence_length: Length of input sequences
            stride: Step size between consecutive sequences (default: 1 for no skipping)
            feature_columns: List of feature column names to use
            label_columns: List of label column names to use
            scaler: Pre-fitted scaler for features
            fit_scaler: Whether to fit the scaler on the data
            device: Device to store tensors on ('cpu', 'cuda', or torch.device)
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.device = get_device(device)

        logger.debug("Collecting LazyFrames into DataFrames...")
        # Collect the data
        features_df = features.collect()
        labels_df = labels.collect()
        logger.debug(f"Collected features: {features_df.shape}, labels: {labels_df.shape}")

        # Ensure dataframes are aligned by timestamp
        if "timestamp" in features_df.columns and "timestamp" in labels_df.columns:
            logger.debug("Aligning features and labels by timestamp...")
            # Join on timestamp to ensure alignment
            combined = features_df.join(labels_df, on="timestamp", how="inner")
            features_df = combined.select(features_df.columns)
            labels_df = combined.select(labels_df.columns)
            logger.debug(f"Aligned data shape: {combined.shape}")

        # Select feature columns
        if feature_columns is None:
            # Exclude timestamp and other non-feature columns
            exclude_cols = ["timestamp", "time"]
            self.feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        else:
            self.feature_columns = feature_columns
        logger.debug(f"Selected {len(self.feature_columns)} feature columns")

        # Select label columns
        if label_columns is None:
            # Use all label columns
            self.label_columns = [
                col for col in labels_df.columns if col not in ["timestamp", "time"]
            ]
        else:
            self.label_columns = label_columns
        logger.debug(f"Selected {len(self.label_columns)} label columns")

        # Extract feature data
        features_df = features_df.select(self.feature_columns).to_numpy()
        labels_df = labels_df.select(self.label_columns).to_numpy()
        logger.debug(
            f"Converted to numpy - features: {features_df.shape}, labels: {labels_df.shape}"
        )

        # TODO: add missing value handling
        self.features, self.labels = features_df, labels_df

        # Map classification labels to contiguous class indices if single-column labels are provided
        # Expecting directional labels in {-1, 0, 1} → map to {0, 1, 2}
        logger.debug("Processing label encoding...")
        if self.labels.ndim == 2 and self.labels.shape[1] == 1:
            labels_1d = self.labels.squeeze(1)
            unique_vals = np.unique(labels_1d)
            # Only apply mapping if labels look like directional labels
            if set(unique_vals.tolist()).issubset({-1, 0, 1}):
                logger.debug(f"Mapping directional labels {unique_vals} to class indices")
                mapping = {-1: 0, 0: 1, 1: 2}
                # Vectorized mapping; for any unexpected values, fall back to 1 (neutral)
                mapped = np.vectorize(lambda v: mapping.get(int(v), 1))(labels_1d)
                self.labels = mapped.astype(np.int64)
            else:
                # Otherwise, try casting to integers safely
                logger.debug(f"Casting labels to int64, unique values: {unique_vals}")
                self.labels = labels_1d.astype(np.int64)
        elif self.labels.ndim == 1:
            # Already 1D labels
            self.labels = self.labels.astype(np.int64)

        # Scale features
        logger.debug(
            f"Scaling features (scaler={'provided' if scaler else 'none'}, fit={fit_scaler})..."
        )
        if scaler is not None:
            self.scaler = scaler
            if fit_scaler:
                # If scaler provided AND fit_scaler=True, refit it
                self.features = self.scaler.fit_transform(self.features)
            else:
                # If scaler provided and fit_scaler=False, just transform
                self.features = self.scaler.transform(self.features)
        else:
            # No scaler provided
            if fit_scaler:
                # Create and fit new scaler
                self.scaler = StandardScaler()
                self.features = self.scaler.fit_transform(self.features)
            else:
                # No scaler and no fitting - leave features as is
                self.scaler = None
                # Features remain unchanged

        # Convert to tensors and move to specified device
        logger.debug(f"Converting to tensors and moving to device: {self.device}...")
        self.features = torch.FloatTensor(
            self.features,
        ).to(self.device)
        # Ensure labels are 1D LongTensor of class indices for CrossEntropyLoss
        if isinstance(self.labels, np.ndarray) and self.labels.ndim > 1:
            self.labels = torch.LongTensor(self.labels.squeeze(-1)).to(self.device)
        else:
            self.labels = torch.LongTensor(self.labels).to(self.device)

        # Calculate valid sequence indices
        logger.debug("Calculating valid sequence indices...")
        self.valid_indices = self._get_valid_sequence_indices()

        logger.info(f"Dataset created with {len(self.valid_indices)} valid sequences")
        logger.info(f"Sequence length: {self.sequence_length}, Stride: {self.stride}")
        logger.info(f"Features shape: {self.features.shape}")
        logger.info(f"Labels shape: {self.labels.shape}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Feature columns: {self.feature_columns}")
        logger.info(f"Label columns: {self.label_columns}")

    def _get_valid_sequence_indices(self) -> list[int]:
        """Get indices where we can create valid sequences."""
        if not torch.isnan(self.features).any():
            return list(range(0, len(self.features) - self.sequence_length + 1, self.stride))

        logger.warning(
            "Missing values detected in features; calculating valid sequences accordingly."
        )

        valid_indices = []
        for i in range(0, len(self.features) - self.sequence_length + 1, self.stride):
            # Check if the sequence has any NaN values
            if not torch.isnan(self.features[i : i + self.sequence_length]).any():
                valid_indices.append(i)
        return valid_indices

    def __len__(self) -> int:
        """Return the number of valid sequences."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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

    def get_feature_names(self) -> list[str]:
        """Get the names of feature columns."""
        return self.feature_columns

    def get_label_names(self) -> list[str]:
        """Get the names of label columns."""
        return self.label_columns

    def get_scaler(self) -> StandardScaler:
        """Get the fitted scaler."""
        return self.scaler


def create_dataloaders(
    features: pl.LazyFrame,
    labels: pl.LazyFrame,
    data_config: DataConfig,
    feature_columns: list[str] | None = None,
    label_columns: list[str] | None = None,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Create train, validation, and test dataloaders with proper temporal splitting.

    IMPORTANT: This function implements time series best practices:
    - Uses sequential (temporal) splits, not random splits
    - Fits scaler ONLY on training data
    - Does NOT shuffle data (preserves temporal order)
    - Train set contains earliest data, test set contains latest data

    Args:
        features: Polars LazyFrame containing features
        labels: Polars LazyFrame containing labels
        data_config: Data configuration
        feature_columns: List of feature column names to use
        label_columns: List of label column names to use
        random_seed: Random seed for reproducibility (unused; splits are temporal)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    logger.info("Creating dataset with temporal splits (no data leakage)...")

    # Create the full dataset WITHOUT fitting scaler yet
    # We need to fit scaler only on training data
    full_dataset = FinancialDataset(
        features=features,
        labels=labels,
        sequence_length=data_config.sequence_length,
        stride=data_config.stride,
        feature_columns=feature_columns,
        label_columns=label_columns,
        scaler=None,
        fit_scaler=False,  # Don't fit yet!
        device=data_config.device,
    )

    # Calculate split indices (temporal order - NO shuffling!)
    total_size = len(full_dataset)
    train_end = int(data_config.train_split * total_size)
    val_end = train_end + int(data_config.val_split * total_size)

    logger.info(f"Total sequences: {total_size:,}")
    logger.info(f"Train indices: 0 to {train_end:,} ({data_config.train_split*100:.1f}%)")
    logger.info(f"Val indices: {train_end:,} to {val_end:,} ({data_config.val_split*100:.1f}%)")
    logger.info(f"Test indices: {val_end:,} to {total_size:,} ({data_config.test_split*100:.1f}%)")

    # Create indices for each split
    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, total_size))

    # Fit scaler ONLY on training data (this is critical!)
    train_features = full_dataset.features[train_indices]
    scaler = StandardScaler()
    scaler.fit(train_features.cpu().numpy())

    logger.info("Fitted scaler on training data only")
    logger.info(f"Feature means: {scaler.mean_[:5]}...")  # Show first 5
    logger.info(f"Feature stds: {scaler.scale_[:5]}...")

    # Apply scaler to ALL data (transform, not fit_transform)
    full_dataset.features = torch.FloatTensor(
        scaler.transform(full_dataset.features.cpu().numpy())
    ).to(full_dataset.device)
    full_dataset.scaler = scaler

    # Create subsets (maintaining temporal order)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create dataloaders - NO SHUFFLING for time series!
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,  # Critical: no shuffling for time series
        num_workers=data_config.num_workers,
        pin_memory=torch.cuda.is_available() and data_config.device == "cpu",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=torch.cuda.is_available() and data_config.device == "cpu",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=torch.cuda.is_available() and data_config.device == "cpu",
    )

    logger.info(
        f"Created temporal dataloaders - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, scaler


def prepare_data_for_training(
    features: pl.LazyFrame,
    labels: pl.LazyFrame,
    data_config: DataConfig,
    feature_columns: list[str] | None = None,
    label_columns: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler, list[str], list[str]]:
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

    # Create a temporary dataset to get column names
    temp_dataset = FinancialDataset(
        features=features,
        labels=labels,
        sequence_length=data_config.sequence_length,
        stride=data_config.stride,
        feature_columns=feature_columns,
        label_columns=label_columns,
        scaler=scaler,
        fit_scaler=False,
    )

    feature_names = temp_dataset.get_feature_names()
    label_names = temp_dataset.get_label_names()

    return train_loader, val_loader, test_loader, scaler, feature_names, label_names
