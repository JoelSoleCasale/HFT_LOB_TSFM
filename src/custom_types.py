from pathlib import Path
from typing import Any, Protocol, TypeAlias
import polars as pl


class DataRequest(Protocol):
    """Protocol for data request parameters."""

    def get_path(self) -> Path:
        """Return parameters to construct cache path."""
        ...


FeatureVector: TypeAlias = pl.LazyFrame
LabelVector: TypeAlias = pl.LazyFrame
FeatureConfig: TypeAlias = dict[str, Any]
