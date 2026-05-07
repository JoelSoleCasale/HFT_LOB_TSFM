import itertools
import tempfile
from pathlib import Path
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import polars as pl
import yaml
from definitions import ROOT_DIR
from typing import Literal
from data_manager.downloader.data_downloader_request import RawDataRequest
from data_manager.downloader.api_client import CryptoDataAPIClient
from dotenv import load_dotenv
import os


class DataDownloader:
    """
    A class to download high-frequency cryptocurrency data from the cryptohftdata API.
    """

    def __init__(
        self,
        base_folder: str = ROOT_DIR / "data/",
        relevant_features_path: str | None = None,
        cryptohftdata_api_key: str | None = None,
        max_workers: int = 4,
        strict_download: bool = True,
    ):
        """
        Initializes the DataDownloader.

        Args:
            base_folder (str, optional): The base directory to store downloaded data. Defaults to ".data/".
            relevant_features_path (Optional[str], optional): Path to a YAML file with relevant features to keep.
                Defaults to None, which keeps all features.
            cryptohftdata_api_key (Optional[str], optional): API key for authentication.
            max_workers (int, optional): Maximum number of threads for parallel preprocessing. Defaults to 4.
            strict_download (bool, optional): If True, raises an error a file fails to download.
        """
        self.base_folder = Path(base_folder)
        self.relevant_features = None
        self.max_workers = max_workers
        self.strict_download = strict_download

        if relevant_features_path:
            with open(relevant_features_path, "r") as f:
                self.relevant_features = yaml.safe_load(f)

        if cryptohftdata_api_key is None:
            load_dotenv()
            self.api_key = os.getenv("CRYPTOHFTDATA_API_KEY")
            if not self.api_key:
                raise ValueError("CRYPTOHFTDATA_API_KEY environment variable not set.")
        else:
            self.api_key = cryptohftdata_api_key

        self.api_client = CryptoDataAPIClient(api_key=self.api_key)

    def download_data(
        self,
        data_type: str | list[str],
        symbol: str | list[str],
        exchange: str | list[str],
        date_val: str | list[str],
        skip_existing: bool = True,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
    ) -> None:
        """
        Downloads data for the given data types, symbols, exchanges, and dates.
        It iterates through the Cartesian product of all provided lists.

        Args:
            data_type (Union[str, List[str]]): The type(s) of data to download. Must be one of:
                "orderbook", "trades", "ticker", "mark_price", "funding_rates", "open_interest", "liquidations".
            symbol (Union[str, List[str]]): The trading symbol(s) (e.g., 'BTCUSDT').
            exchange (Union[str, List[str]]): The exchange(s) (e.g., 'binance-futures').
            date_val (Union[str, List[str]]): The date(s) in 'YYYY-MM-DD' format.
            skip_existing (bool, optional): If True, skips download if the file already exists. Defaults to True.
            reference_ts (Literal["received_time", "event_time"], optional): The timestamp reference to use.
                Defaults to "received_time". The downloaded data will be sorted by this timestamp.
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
                raise ValueError(f"Invalid data_type '{dt}'. Must be one of {allowed_types}.")

        symbols = [symbol] if isinstance(symbol, str) else symbol
        exchanges = [exchange] if isinstance(exchange, str) else exchange
        dates = [date_val] if isinstance(date_val, str) else date_val

        for dt, sym, ex, d in itertools.product(data_types, symbols, exchanges, dates):
            try:
                date_obj = datetime.strptime(d, "%Y-%m-%d").date()
                request = RawDataRequest(data_type=dt, symbol=sym, exchange=ex, date=date_obj)

                file_path = self.base_folder / request.get_path()

                if skip_existing and file_path.exists():
                    logger.info(f"File already exists, skipping: {file_path}")
                    continue

                self._download_single_request(request, reference_ts=reference_ts)

            except Exception as e:
                logger.error(f"Failed to download {dt} for {sym} on {ex} for {d}: {e}")

    def _basic_preprocessing(
        self,
        df: pl.DataFrame,
        request: RawDataRequest,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
    ) -> pl.DataFrame:
        """
        Apply basic preprocessing to a DataFrame.

        Args:
            df: Input DataFrame
            request: Raw data request with metadata
            reference_ts: Timestamp column to sort by

        Returns:
            Preprocessed DataFrame
        """
        if self.relevant_features and request.data_type in self.relevant_features:
            features_to_keep = self.relevant_features[request.data_type]
            if features_to_keep:
                features_to_keep = [col for col in features_to_keep if col in df.columns]
                df = df[features_to_keep]

        for col in df.select(pl.col(pl.Utf8)).columns:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
                logger.debug(f"Casted column {col} to Float64")
            except Exception:
                pass

        if reference_ts in df.columns:
            logger.debug(f"Sorting by {reference_ts}")
            df = df.sort(by=reference_ts)
        else:
            logger.warning(f"Reference timestamp '{reference_ts}' not in columns. Skipping sort.")

        return df

    def _process_and_save_hourly_file(
        self,
        hour: int,
        request: RawDataRequest,
        temp_path: Path,
        reference_ts: Literal["received_time", "event_time"],
    ) -> tuple[int, bool]:
        """
        Download, process and save a single hourly file.

        This method is designed to be called in parallel threads.

        Args:
            hour: Hour of the day
            request: Raw data request with metadata
            temp_path: Temporary directory path
            reference_ts: Timestamp column to sort by

        Returns:
            Tuple of (hour, success_flag)
        """
        try:
            df = self.api_client.download_hourly_file(request, hour)

            if df is None or len(df) == 0:
                logger.debug(f"No data for hour {hour}")
                return (hour, False)

            df = self._basic_preprocessing(df, request, reference_ts=reference_ts)

            temp_file = temp_path / f"hour_{hour:02d}.parquet"
            df.write_parquet(temp_file)
            logger.debug(f"Saved preprocessed hour {hour} to temp file: {temp_file}")
            return (hour, True)
        except Exception as e:
            logger.error(f"Failed to process hour {hour}: {e}")
            return (hour, False)

    def _download_single_request(
        self,
        request: RawDataRequest,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
    ) -> None:
        """
        Downloads hourly data files for a single day, applies preprocessing to each,
        saves them to a temporary directory, then merges into a single parquet file.

        Uses multithreading to download and preprocess multiple hours in parallel
        for better performance.
        """
        logger.info(
            f"Downloading {request.data_type} for {request.symbol} on {request.exchange} for {request.date}"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.debug(f"Using temporary directory: {temp_path}")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for hour in range(24):
                    future = executor.submit(
                        self._process_and_save_hourly_file,
                        hour,
                        request,
                        temp_path,
                        reference_ts,
                    )
                    futures.append(future)

                files_processed = 0
                for future in as_completed(futures):
                    hour, success = future.result()
                    if success:
                        files_processed += 1

            if files_processed == 0:
                logger.warning(
                    f"No data for {request.data_type} for {request.symbol} on {request.exchange} for {request.date}"
                )
                return

            if files_processed < 24 and self.strict_download:
                raise RuntimeError(
                    f"Only {files_processed}/24 hourly files were downloaded and processed for {request.symbol} on {request.exchange} for {request.date}. "
                    "Aborting due to strict_download=True."
                )

            logger.info(f"Merging {files_processed} hourly files into single parquet file")

            lf = pl.scan_parquet(str(temp_path / "hour_*.parquet"))

            file_path = self.base_folder / request.get_path()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            lf.sink_parquet(str(file_path))

            logger.success(f"Saved merged file to disk: {file_path}")

    def get_data(
        self,
        data_type: str,
        symbol: str,
        exchange: str,
        date: str | date,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
    ) -> pl.LazyFrame:
        """
        Get data for a single request by reading from disk.

        Args:
            data_type (str): The type of data to get.
            symbol (str): The trading symbol.
            exchange (str): The exchange.
            date (str): The date in 'YYYY-MM-DD' format.
            reference_ts (Literal["received_time", "event_time"], optional): The timestamp reference to use.
                Defaults to "received_time". This parameter is used if the file doesn't exist and needs to be downloaded.

        Returns:
            pl.LazyFrame: The requested data as a DataFrame.

        Raises:
            FileNotFoundError: If the file doesn't exist on disk.
        """
        date_obj = datetime.strptime(date, "%Y-%m-%d").date() if isinstance(date, str) else date
        request = RawDataRequest(
            data_type=data_type, symbol=symbol, exchange=exchange, date=date_obj
        )

        file_path = self.base_folder / request.get_path()

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}. Attempting to download...")
            self._download_single_request(request, reference_ts=reference_ts)

            if not file_path.exists():
                raise FileNotFoundError(f"Failed to download or file does not exist: {file_path}")

        logger.debug(f"Reading from disk: {file_path}")
        df = pl.scan_parquet(file_path)
        return df
