"""Configuration dataclass for PCA processing."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PCAConfig:
    """Configuration for incremental PCA on embeddings.

    Attributes:
        n_components: Number of principal components to keep
        chunk_size: Number of rows to process per chunk during fitting/transformation
        whiten: Whether to whiten the transformed components
        save_path: Optional path to save fitted PCA model
    """

    n_components: int
    chunk_size: int = 20_000
    whiten: bool = False
    save_path: Path | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.n_components <= 0:
            raise ValueError(f"n_components must be positive, got {self.n_components}")

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
