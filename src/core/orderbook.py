from sortedcontainers import SortedDict
import polars as pl
from polars._typing import FileSource


class OrderBookSnapshot:
    def __init__(self):
        self.bid = SortedDict(lambda k: -k)
        self.ask = SortedDict()

    def __getitem__(self, key):
        if key == "bid":
            return self.bid
        if key == "ask":
            return self.ask
        raise KeyError(key)

    def update(self, side: str, price: float, quantity: float):
        """Update the orderbook snapshot with a new order or cancellation."""
        if quantity == 0:
            self[side].pop(price, None)
        else:
            self[side][price] = quantity


class OrderBookData:
    """Immutable data container for orderbook snapshots."""

    def __init__(self, data: pl.DataFrame | None = None, levels: int | None = None):
        if data is not None:
            self._validate_structure(data)
            self._df = data.set_sorted("timestamp")
            self._levels = OrderBookData._infer_levels(data)
        elif levels is not None:
            self._df = pl.DataFrame(schema=self.get_orderbook_schema(levels))
            self._levels = levels
        else:
            raise ValueError("Either data or levels must be provided")

    @property
    def df(self) -> pl.DataFrame:
        return self._df.clone()

    @property
    def levels(self) -> int:
        return self._levels

    @staticmethod
    def get_orderbook_schema(levels: int) -> list[tuple[str, pl.DataType]]:
        """Generate schema for an orderbook snapshot dataframe."""
        schema = [("timestamp", pl.Int64)]
        schema.extend(
            [
                (f"{s}{i}_{t}", pl.Float64)
                for t in ["price", "qty"]
                for i in range(1, levels + 1)
                for s in ["ask", "bid"]
            ]
        )
        return schema

    @staticmethod
    def get_orderbook_columns(levels: int) -> list[str]:
        """Generate list of column names for an orderbook snapshot dataframe."""
        return [col for col, _ in OrderBookData.get_orderbook_schema(levels)]

    @staticmethod
    def _infer_levels(df: pl.DataFrame) -> int:
        """Infer levels from DataFrame structure."""
        price_columns = [col for col in df.columns if col.endswith("_price")]
        if not price_columns:
            raise ValueError("No price columns found")
        return len(price_columns) // 2

    @staticmethod
    def _validate_structure(df: pl.DataFrame):
        """Validate that the DataFrame has correct orderbook structure."""
        levels = OrderBookData._infer_levels(df)
        expected_columns = set(OrderBookData.get_orderbook_columns(levels))
        expected_schema = OrderBookData.get_orderbook_schema(levels)
        if set(df.columns) != expected_columns:
            raise ValueError("DataFrame does not have correct orderbook column names.")

        for col_name, expected_dtype in expected_schema:
            if df[col_name].dtype != expected_dtype:
                raise ValueError(
                    f"Column '{col_name}' has incorrect dtype. Expected {expected_dtype}, got {df[col_name].dtype}."
                )
        if not df["timestamp"].is_sorted():
            raise ValueError("DataFrame 'timestamp' column is not sorted.")


class OrderBook:
    def __init__(
        self, data: OrderBookData | pl.DataFrame | None = None, levels: int | None = None
    ):
        self._data = data if isinstance(data, OrderBookData) else OrderBookData(data, levels)

    def __len__(self) -> int:
        """Return number of orderbook snapshots."""
        return self._data.df.height

    def __repr__(self) -> str:
        return f"OrderBook(levels={self.levels}, snapshots={len(self)})"

    @property
    def levels(self) -> int:
        return self._data.levels

    @property
    def df(self) -> pl.DataFrame:
        return self._data.df

    @classmethod
    def from_parquet(cls, source: FileSource) -> "OrderBook":
        """
        Create an OrderBook instance from a Parquet file.

        Args:
            source (FileSource): Path or file-like object pointing to the Parquet file
                                containing orderbook data.

        Returns:
            OrderBook: A new OrderBook instance populated with data from the Parquet file.

        Raises:
            FileNotFoundError: If the specified Parquet file cannot be found.
            PolarsError: If the Parquet file cannot be read or has invalid format.

        Example:
            >>> orderbook = OrderBook.from_parquet("data/orderbook.parquet")
            >>> print(len(orderbook))
            1000
        """
        return cls(data=pl.read_parquet(source))

    def to_parquet(self, file: FileSource) -> None:
        """
        Save the OrderBook data to a Parquet file.

        Args:
            file (FileSource): Path or file-like object where the Parquet file will be saved.

        Returns:
            None

        Raises:
            PolarsError: If there is an error writing the DataFrame to Parquet format.

        Example:
            >>> orderbook = OrderBook(levels=5)
            >>> orderbook.to_parquet("data/orderbook.parquet")
        """
        self._data.df.write_parquet(file)

    def select_levels(self, levels: int) -> "OrderBook":
        """
        Select specified number of levels from the orderbook DataFrame.
        If levels >= current level count, returns the full DataFrame.
        Args:
            levels (int): Number of levels to select.
        Returns:
            OrderBook: A new OrderBook instance with the selected levels.
        """
        if levels >= self.levels:
            return self
        df = self.df.select(OrderBookData.get_orderbook_columns(levels))
        return OrderBook(df)

    # ================== Order Book sampling methods ==================

    def sample_by_events(self, n: int) -> "OrderBook":
        """
        Take every nth row from the orderbook DataFrame.
        Args:
            n (int): Step size for sampling rows.
        Returns:
            OrderBook: A new OrderBook instance with the sampled data.
        """
        return OrderBook(self.df.gather_every(n))

    def sample_by_time(self, time_delta: int, interpolate: bool = False) -> "OrderBook":
        """
        Sample the orderbook DataFrame at regular time intervals.
        Args:
            time_delta (int): Time interval in for sampling.
            interpolate (bool): Whether to interpolate missing timestamps. If True,
                                the timestamps in the resulting DataFrame will be
                                evenly spaced by time_delta, with missing values
                                forward-filled. If False, only existing timestamps
                                will be retained.

        Returns:
            OrderBook: A new OrderBook instance with the sampled data.
        """

        res_df = self.df.group_by_dynamic(
            "timestamp", every=f"{time_delta}i", closed="right", label="right"
        ).agg(pl.all().last())

        if interpolate and res_df.height > 1:
            res_df = res_df.upsample(time_column="timestamp", every=f"{time_delta}i").fill_null(
                strategy="forward"
            )

        return OrderBook(res_df)
