"""Abstract base class for all feature extractors"""

from abc import ABC, abstractmethod
import polars as pl
from features.base.input_space import InputSpace


class BaseFeatureExtractor(ABC):
    """Abstract base for all feature extractors"""

    def __init__(self, config: dict[str, object] | None = None):
        self.config = config or {}
        self.feature_names: list[str] = []
        self.dependencies: list[str] = []  # which InputSpace attributes needed

    @abstractmethod
    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        """
        Transform input space to feature space
        Returns: LazyFrame with columns as features, indexed by timestamp
        """
        pass

    def get_feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces"""
        return self.feature_names

    def get_dependencies(self) -> list[str]:
        """Return which InputSpace fields are required"""
        return self.dependencies

    def validate_input(self, input_space: InputSpace) -> bool:
        """Check if required dependencies are present"""
        for dep in self.dependencies:
            if getattr(input_space, dep) is None:
                raise ValueError(f"Missing required dependency: {dep}")
        return True
