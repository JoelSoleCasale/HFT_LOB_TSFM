from sortedcontainers import SortedDict
import polars as pl
from polars._typing import FileSource
from typing import TypeVar, Generic


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

    def get_top_levels(self, levels: int) -> dict[str, float]:
        """Get the top N levels of the orderbook."""
        res = {}
        # Add ask levels
        for j, (ask_price, ask_qty) in enumerate(self.ask.items()[:levels]):
            res[f"ask{j+1}_price"] = ask_price
            res[f"ask{j+1}_qty"] = ask_qty

        # Add bid levels
        for j, (bid_price, bid_qty) in enumerate(self.bid.items()[:levels]):
            res[f"bid{j+1}_price"] = bid_price
            res[f"bid{j+1}_qty"] = bid_qty

        return res


DFType = TypeVar("DFType", pl.DataFrame, pl.LazyFrame)


class OrderBookData(Generic[DFType]):
    """Immutable data container for orderbook snapshots."""

    def __init__(
        self,
        data: DFType | None = None,
        levels: int | None = None,
        allow_duplicates: bool = False,
    ):
        self._is_lazy = isinstance(data, pl.LazyFrame) if data is not None else False

        if data is not None and levels is not None:
            raise ValueError("Provide either data or levels, not both.")

        if data is not None:
            self._validate_structure(data)
            self._levels = OrderBookData._infer_levels(data)
            col_order = self.get_orderbook_columns(self._levels)

            self._df = data.select(col_order).set_sorted("timestamp")
        elif levels is not None:
            self._levels = levels
            schema = self.get_orderbook_schema(levels)
            self._df = pl.DataFrame(schema=schema)
        else:
            raise ValueError("Either data or levels must be provided")

        if not allow_duplicates and not self._is_lazy:
            # Only remove duplicates for eager DataFrames
            self._remove_duplicated_rows()

    @property
    def df(self) -> DFType:
        return self._df.clone()

    @property
    def levels(self) -> int:
        return self._levels

    @property
    def is_lazy(self) -> bool:
        return self._is_lazy

    @staticmethod
    def get_orderbook_schema(levels: int) -> list[tuple[str, pl.DataType]]:
        """Generate schema for an orderbook snapshot dataframe."""
        schema = [("timestamp", pl.Int64)]
        schema.extend(
            [
                (f"{s}{i}_{t}", pl.Float64)
                for i in range(1, levels + 1)
                for s in ["ask", "bid"]
                for t in ["price", "qty"]
            ]
        )
        return schema

    @staticmethod
    def get_orderbook_columns(levels: int) -> list[str]:
        """Generate list of column names for an orderbook snapshot dataframe."""
        return [col for col, _ in OrderBookData.get_orderbook_schema(levels)]

    @staticmethod
    def _infer_levels(df: pl.DataFrame | pl.LazyFrame) -> int:
        """Infer levels from DataFrame structure."""
        price_columns = [col for col in df.collect_schema().names() if col.endswith("_price")]

        if not price_columns:
            raise ValueError("No price columns found")
        return len(price_columns) // 2

    @staticmethod
    def _validate_structure(df: pl.DataFrame | pl.LazyFrame):
        """Validate that the DataFrame has correct orderbook structure."""
        levels = OrderBookData._infer_levels(df)
        expected_columns = set(OrderBookData.get_orderbook_columns(levels))
        expected_schema = OrderBookData.get_orderbook_schema(levels)

        if set(df.collect_schema().names()) != expected_columns:
            raise ValueError("DataFrame does not have correct orderbook column names.")

        # For schema validation, we can only check LazyFrame schema, not data
        for col_name, expected_dtype in expected_schema:
            if df.collect_schema()[col_name] != expected_dtype:
                raise ValueError(
                    f"Column '{col_name}' has incorrect dtype. Expected {expected_dtype}, got {df.schema[col_name]}."
                )

        # For LazyFrames, we can't easily check if timestamp is sorted without collecting
        if isinstance(df, pl.DataFrame) and not df["timestamp"].is_sorted():
            raise ValueError("DataFrame 'timestamp' column is not sorted.")

    def _remove_duplicated_rows(self) -> None:
        """Remove duplicated rows (rows identical to previous row) in place."""
        float_cols = [c for c in self._df.collect_schema().names() if c != "timestamp"]

        mask = pl.any_horizontal([pl.col(col) != pl.col(col).shift(1) for col in float_cols]) | (
            pl.arange(0, pl.len()) == 0
        )

        self._df = self._df.filter(mask)


class OrderBook(Generic[DFType]):
    def __init__(
        self,
        data: OrderBookData | pl.DataFrame | pl.LazyFrame | None = None,
        levels: int | None = None,
    ):
        if isinstance(data, OrderBookData):
            self._data = data
        else:
            self._data = OrderBookData(data, levels)

    def __len__(self) -> int:
        """Return number of orderbook snapshots."""
        if self._data.is_lazy:
            return self._data.df.select(pl.count()).collect().item()
        else:
            return self._data.df.height

    def __repr__(self) -> str:
        mode = "lazy" if self._data.is_lazy else "eager"
        return f"OrderBook(levels={self.levels}, snapshots={len(self)}, mode={mode})"

    @property
    def levels(self) -> int:
        return self._data.levels

    @property
    def df(self) -> DFType:
        return self._data.df

    @property
    def is_lazy(self) -> bool:
        return self._data.is_lazy

    @classmethod
    def from_parquet(cls, source: FileSource, lazy: bool = False) -> "OrderBook":
        """Create an OrderBook from a Parquet file. Pass ``lazy=True`` to read as LazyFrame."""
        if lazy:
            return cls(data=pl.scan_parquet(source))
        else:
            return cls(data=pl.read_parquet(source))

    def to_parquet(self, file: FileSource) -> None:
        """Save the OrderBook data to a Parquet file."""
        if self._data.is_lazy:
            self._data.df.collect().write_parquet(file)
        else:
            self._data.df.write_parquet(file)

    def collect(self) -> "OrderBook[pl.DataFrame]":
        """
        Convert a lazy OrderBook to an eager OrderBook by collecting the data.

        Returns:
            OrderBook: A new OrderBook instance with collected DataFrame.
        """
        if not self._data.is_lazy:
            return self

        return OrderBook(self._data.df.collect())

    def lazy(self) -> "OrderBook[pl.LazyFrame]":
        """
        Convert an eager OrderBook to a lazy OrderBook.

        Returns:
            OrderBook: A new OrderBook instance with LazyFrame.
        """
        if self._data.is_lazy:
            return self

        return OrderBook(self._data.df.lazy())

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
            time_delta (int): Time interval (in timestamp units) for sampling.
            interpolate (bool): Whether to interpolate missing timestamps. If True,
                                the timestamps in the resulting DataFrame will be
                                evenly spaced by time_delta, with missing values
                                forward-filled. If False, only existing timestamps
                                will be retained.

        Returns:
            OrderBook: A new OrderBook instance with the sampled data.
        """
        res_lf = self.df.group_by_dynamic(
            "timestamp", every=f"{time_delta}i", closed="right", label="right"
        ).agg(pl.all().last())

        if interpolate:
            first_ts = res_lf.select(pl.col("timestamp").first()).lazy().collect().item()
            last_ts = res_lf.select(pl.col("timestamp").last()).lazy().collect().item()

            full_range = pl.select(
                pl.int_range(first_ts, last_ts + time_delta, time_delta).alias("timestamp"),
                eager=not self._data.is_lazy,
            )

            res_lf = full_range.join_asof(res_lf, on="timestamp").fill_null(strategy="forward")

        return OrderBook(OrderBookData(res_lf, allow_duplicates=True))

    def get_mid_prices(self) -> DFType:
        """
        Compute mid prices for each orderbook snapshot.
        Returns:
            pl.DataFrame or pl.LazyFrame: DataFrame/LazyFrame with 'timestamp' and 'mid_price' columns.
        """
        bid_price_col = "bid1_price"
        ask_price_col = "ask1_price"

        mid_prices = self.df.select(
            [
                pl.col("timestamp"),
                ((pl.col(bid_price_col) + pl.col(ask_price_col)) / 2).alias("mid_price"),
            ]
        )
        return mid_prices

    def get_spreads(self) -> DFType:
        """
        Compute spreads for each orderbook snapshot.
        Returns:
            pl.DataFrame or pl.LazyFrame: DataFrame/LazyFrame with 'timestamp' and 'spread' columns.
        """
        bid_price_col = "bid1_price"
        ask_price_col = "ask1_price"

        spreads = self.df.select(
            [
                pl.col("timestamp"),
                (pl.col(ask_price_col) - pl.col(bid_price_col)).alias("spread"),
            ]
        )
        return spreads
