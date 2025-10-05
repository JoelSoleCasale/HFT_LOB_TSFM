from pathlib import Path
from typing import Protocol


class DataRequest(Protocol):
    """Protocol for data request parameters."""

    def get_path(self) -> Path:
        """Return parameters to construct cache path."""
        ...
