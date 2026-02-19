"""
Custom API client for downloading cryptocurrency data from CryptoHFTData API.
"""

import io
import time

import polars as pl
import requests
import zstandard as zstd
from loguru import logger

from data_manager.downloader.data_downloader_request import RawDataRequest


class CryptoDataAPIClient:
    """
    Client for downloading cryptocurrency data from CryptoHFTData API.
    Downloads data in hourly chunks.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.cryptohftdata.com",
        timeout: int = 300,
        max_retries: int = 10,
    ):
        """
        Initialize the API client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url
        self.download_endpoint = f"{self.base_url}/download"
        self.timeout = timeout
        self.max_retries = max_retries

    def _generate_file_path(
        self,
        request: RawDataRequest,
        hour: int,
    ) -> str:
        """
        Generate R2 file path based on the directory structure.

        Format: [exchange]/[YYYY-MM-DD]/[HH]/[symbol]_[type].parquet.zst

        Args:
            request: RawDataRequest containing exchange, symbol, data_type, and date
            hour: Hour of the day (0-23)
        """
        date_str = request.date.strftime("%Y-%m-%d")
        hour_str = f"{hour:02d}"
        return f"{request.exchange}/{date_str}/{hour_str}/{request.symbol}_{request.data_type}.parquet.zst"

    def download_hourly_file(
        self,
        request: RawDataRequest,
        hour: int,
        retry_count: int = 0,
        sleep_seconds: int = 10,
    ) -> pl.DataFrame | None:
        """
        Download a single hourly file from the API.

        Args:
            request: RawDataRequest containing exchange, symbol, data_type, and date
            hour: Hour of the day (0-23)
            retry_count: Current retry attempt number
            sleep_seconds: Seconds to wait before retrying

        Returns:
            pl.DataFrame or None if file doesn't exist or download fails
        """
        file_path = self._generate_file_path(request, hour)

        # Build URL with API key authentication
        url = f"{self.download_endpoint}?file={file_path}&api_key={self.api_key}"

        try:
            logger.debug(f"Downloading file: {file_path}")
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == 404:
                logger.warning(f"File not found: {file_path}")
                return None
            elif response.status_code == 401:
                logger.error("Authentication failed - invalid API key")
                return None
            elif response.status_code != 200:
                logger.warning(
                    f"Failed to download {file_path}: HTTP {response.status_code}, waiting {sleep_seconds}s before retry"
                )
                time.sleep(sleep_seconds)
                if retry_count < self.max_retries:
                    logger.info(f"Retrying download ({retry_count + 1}/{self.max_retries})")
                    return self.download_hourly_file(request, hour, retry_count + 1)
                return None

            # Check if response is empty
            if len(response.content) == 0:
                logger.warning(f"Empty response for {file_path}")
                return None

            # Decompress and read the file
            df = self._decompress_and_read(response.content, file_path)

            if df is not None:
                logger.debug(f"Successfully downloaded {file_path}: {len(df)} records")

            return df

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout downloading {file_path}")
            if retry_count < self.max_retries:
                logger.info(f"Retrying download ({retry_count + 1}/{self.max_retries})")
                return self.download_hourly_file(request, hour, retry_count + 1)
            return None
        except Exception as e:
            logger.warning(f"Error downloading {file_path}: {str(e)}")
            return None

    def _decompress_and_read(self, content: bytes, file_path: str) -> pl.DataFrame | None:
        """
        Decompress zstd content and read as parquet DataFrame.

        Args:
            content: Raw bytes content
            file_path: File path for logging purposes

        Returns:
            pl.DataFrame or None if decompression/reading fails
        """
        try:
            # Check if content is already plain parquet (PAR1 magic)
            if content.startswith(b"PAR1"):
                df = pl.read_parquet(io.BytesIO(content))
                return df

            # Try to decompress as zstd
            try:
                dctx = zstd.ZstdDecompressor()
                decompressed_data = dctx.decompress(content)
                df = pl.read_parquet(io.BytesIO(decompressed_data))
                return df
            except zstd.ZstdError as zst_error:
                logger.warning(f"Zstd decompression failed for {file_path}: {str(zst_error)}")
                # Fallback: try reading as plain parquet
                try:
                    return pl.read_parquet(io.BytesIO(content))
                except Exception as plain_error:
                    logger.error(
                        f"Failed to read {file_path} as both compressed and plain parquet: "
                        f"zstd={zst_error}, plain={plain_error}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
