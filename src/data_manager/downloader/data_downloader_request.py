from pathlib import Path
from datetime import date
from dataclasses import dataclass


@dataclass
class RawDataRequest:
    """
    A class representing a data request for downloading cryptocurrency data.
    """

    data_type: str
    symbol: str
    exchange: str
    date: date

    def get_path(self) -> Path:
        """Return parameters to construct cache path."""
        return (
            Path(self.data_type)
            / self.exchange
            / self.symbol
            / f"{self.date.strftime('%Y-%m-%d')}.parquet"
        )
