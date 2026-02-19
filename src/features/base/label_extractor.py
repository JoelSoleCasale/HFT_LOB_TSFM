"""Abstract base class for all label extractors"""

from abc import ABC, abstractmethod
import polars as pl
from features.base.input_space import InputSpace


class BaseLabelExtractor(ABC):
    """Abstract base for all label extractors"""

    def __init__(self, config: dict[str, object] = None):
        self.config = config or {}
        self.label_names: list[str] = []
        self.dependencies: list[str] = []

    @abstractmethod
    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        """
        Transform input space to label space
        Returns: LazyFrame with columns as labels, indexed by timestamp
        """
        pass

    def get_dependencies(self) -> list[str]:
        """Return which InputSpace fields are required"""
        return self.dependencies

    def validate_input(self, input_space: InputSpace) -> bool:
        """Check if required dependencies are present"""
        for dep in self.dependencies:
            if getattr(input_space, dep) is None:
                raise ValueError(f"Missing required dependency: {dep}")
        return True
