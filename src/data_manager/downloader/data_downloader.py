import itertools
from pathlib import Path
from datetime import datetime
from loguru import logger
import cryptohftdata as chd
import polars as pl
import yaml
from definitions import ROOT_DIR
from typing import Literal
from utils import cache_func
from data_manager.downloader.data_downloader_request import RawDataRequest


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

        # Apply cache decorator to the download function
        self._cached_download_single = cache_func(
            self._download_single_uncached, self.base_folder, load_cached_data=False
        )

    def download_data(
        self,
        data_type: str | list[str],
        symbol: str | list[str],
        exchange: str | list[str],
        date: str | list[str],
        skip_existing: bool = True,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
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
        dates = [date] if isinstance(date, str) else date

        for dt, sym, ex, d in itertools.product(data_types, symbols, exchanges, dates):
            try:
                # Convert string date to date object
                date_obj = datetime.strptime(d, "%Y-%m-%d").date()
                request = RawDataRequest(data_type=dt, symbol=sym, exchange=ex, date=date_obj)
                if not skip_existing:
                    _, from_cache = (
                        self._download_single_uncached(request, reference_ts=reference_ts),
                        False,
                    )
                else:
                    _, from_cache = self._cached_download_single(
                        request, reference_ts=reference_ts
                    )

                if from_cache:
                    logger.info(f"Loaded from cache: {self.base_folder / request.get_path()}")
                else:
                    logger.success(
                        f"Downloaded and cached: {self.base_folder / request.get_path()}"
                    )

            except Exception as e:
                logger.error(f"Failed to download {dt} for {sym} on {ex} for {d}: {e}")

    def _download_single_uncached(
        self,
        request: RawDataRequest,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
    ) -> pl.DataFrame | None:
        """
        Downloads a single data file and returns the DataFrame.
        This method is wrapped by the cache_func decorator.
        """
        logger.info(
            f"Downloading {request.data_type} for {request.symbol} on {request.exchange} for {request.date}"
        )

        download_function = getattr(chd, f"get_{request.data_type}")

        # Convert date back to string for the API call
        date_str = request.date.strftime("%Y-%m-%d")
        df: pl.DataFrame = pl.DataFrame(
            download_function(
                symbol=request.symbol,
                exchange=request.exchange,
                start_date=date_str,
                end_date=date_str,
            )
        )

        if len(df) == 0:
            logger.warning(
                f"No data for {request.data_type} for {request.symbol} on {request.exchange} for {request.date}"
            )
            return None

        if self.relevant_features and request.data_type in self.relevant_features:
            features_to_keep = self.relevant_features[request.data_type]
            if features_to_keep:
                # Filter out columns that are not in the dataframe
                features_to_keep = [col for col in features_to_keep if col in df.columns]
                df = df[features_to_keep]

        # check if any string can be converted to float
        for col in df.select(pl.col(pl.Utf8)).columns:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
                logger.debug(f"Casted column {col} to Float64")
            except Exception:
                logger.debug(f"Could not cast column {col} to Float64, keeping as is")

        # sort by reference timestamp
        if reference_ts in df.columns:
            logger.debug(f"Sorting by {reference_ts}")
            df = df.sort(by=reference_ts)
        else:
            logger.warning(f"Reference timestamp '{reference_ts}' not in columns. Skipping sort.")

        return df

    def get_data(
        self,
        data_type: str,
        symbol: str,
        exchange: str,
        date: str,
        reference_ts: Literal["received_time", "event_time"] = "received_time",
    ) -> tuple[pl.DataFrame, bool]:
        """
        Get data for a single request, either from cache or by downloading.

        Args:
            data_type (str): The type of data to get.
            symbol (str): The trading symbol.
            exchange (str): The exchange.
            date (str): The date in 'YYYY-MM-DD' format.
            reference_ts (Literal["received_time", "event_time"], optional): The timestamp reference to use.
                Defaults to "received_time".

        Returns:
            tuple[pl.DataFrame, bool]: A tuple containing the DataFrame and a boolean indicating
                if the data was loaded from cache (True) or downloaded (False).
        """
        # Convert string date to date object
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        request = RawDataRequest(
            data_type=data_type, symbol=symbol, exchange=exchange, date=date_obj
        )
        return self._cached_download_single(request, reference_ts=reference_ts)
