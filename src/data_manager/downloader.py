import itertools
from pathlib import Path
from loguru import logger
import cryptohftdata as chd
from tqdm import tqdm
import pandas as pd
import yaml
from definitions import ROOT_DIR


class DataDownloader:
    """
    A class to download high-frequency cryptocurrency data using the cryptohftdata library.
    """

    def __init__(
        self,
        base_folder: str = ROOT_DIR / "data/",
        relevant_features_path: str | None = None,
    ):
        """
        Initializes the DataDownloader.

        Args:
            base_folder (str, optional): The base directory to store downloaded data. Defaults to ".data/".
            relevant_features_path (Optional[str], optional): Path to a YAML file with relevant features to keep.
                Defaults to None, which keeps all features.
        """
        self.base_folder = Path(base_folder)
        self.relevant_features = None
        if relevant_features_path:
            with open(relevant_features_path, "r") as f:
                self.relevant_features = yaml.safe_load(f)

    def download_data(
        self,
        data_type: str | list[str],
        symbol: str | list[str],
        exchange: str | list[str],
        date: str | list[str],
        skip_existing: bool = True,
    ):
        """
        Downloads data for the given data types, symbols, exchanges, and dates.

        It iterates through the Cartesian product of all provided lists.

        Args:
            data_type (Union[str, List[str]]): The type(s) of data to download. Must be one of:
                "orderbook", "trades", "ticker", "mark_price", "funding_rates", "open_interest", "liquidations".
            symbol (Union[str, List[str]]): The trading symbol(s) (e.g., 'BTCUSDT').
            exchange (Union[str, List[str]]): The exchange(s) (e.g., 'binance-futures').
            date (Union[str, List[str]]): The date(s) in 'YYYY-MM-DD' format.
            skip_existing (bool, optional): If True, skips download if the file already exists. Defaults to True.
        """
        allowed_types = {
            "orderbook",
            "trades",
            "ticker",
            "mark_price",
            "funding_rates",
            "open_interest",
            "liquidations",
        }
        data_types = [data_type] if isinstance(data_type, str) else data_type
        for dt in data_types:
            if dt not in allowed_types:
                raise ValueError(
                    f"Invalid data_type '{dt}'. Must be one of {allowed_types}."
                )

        symbols = [symbol] if isinstance(symbol, str) else symbol
        exchanges = [exchange] if isinstance(exchange, str) else exchange
        dates = [date] if isinstance(date, str) else date

        combinations = list(itertools.product(data_types, symbols, exchanges, dates))

        for dt, sym, ex, d in tqdm(combinations, desc="Downloading data"):
            try:
                self._download_single(dt, sym, ex, d, skip_existing)
            except Exception as e:
                logger.error(f"Failed to download {dt} for {sym} on {ex} for {d}: {e}")

    def _download_single(
        self,
        data_type: str,
        symbol: str,
        exchange: str,
        date: str,
        skip_existing: bool,
    ):
        """
        Downloads a single data file.
        """
        output_dir = self.base_folder / data_type / exchange / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{date}.parquet"

        if skip_existing and output_file.exists():
            logger.info(f"Skipping existing file: {output_file}")
            return

        logger.info(f"Downloading {data_type} for {symbol} on {exchange} for {date}")

        download_function = getattr(chd, f"get_{data_type}")

        df: pd.DataFrame = download_function(
            symbol=symbol,
            exchange=exchange,
            start_date=date,
            end_date=date,
        )

        if df.empty:
            logger.warning(
                f"No data for {data_type} for {symbol} on {exchange} for {date}"
            )
            return

        if self.relevant_features and data_type in self.relevant_features:
            features_to_keep = self.relevant_features[data_type]
            if features_to_keep:
                # Filter out columns that are not in the dataframe
                features_to_keep = [
                    col for col in features_to_keep if col in df.columns
                ]
                df = df[features_to_keep]

        df.to_parquet(output_file)
        logger.success(f"Successfully downloaded and saved to {output_file}")
