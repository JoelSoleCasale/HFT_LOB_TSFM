"""Abstract base class for all label extractors"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import polars as pl
from features.base.input_space import InputSpace


class BaseLabelExtractor(ABC):
    """Abstract base for all label extractors"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.label_names: List[str] = []
        self.dependencies: List[str] = []
        self.lookahead_window: Optional[int] = None  # prevent leakage

    @abstractmethod
    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        """
        Transform input space to label space
        Returns: LazyFrame with columns as labels, indexed by timestamp
        """
        pass

    @abstractmethod
    def get_label_names(self) -> List[str]:
        """Return list of label names this extractor produces"""
        pass

    def validate_no_leakage(self, timestamp: int) -> bool:
        """Ensure labels only use past/current data at timestamp"""
        # Implement as needed
        return True
