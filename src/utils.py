import polars as pl
from typing import Iterator


def iter_slices(
    df: pl.DataFrame | pl.LazyFrame, n_rows: int = 10_000
) -> Iterator[pl.DataFrame | pl.LazyFrame]:
    """Iterate over DataFrame/LazyFrame in slices."""
    if isinstance(df, pl.DataFrame):
        for offset in range(0, df.height, n_rows):
            yield df.slice(offset, n_rows)
    else:
        row_count = df.select(pl.len()).collect().item()
        for offset in range(0, row_count, n_rows):
            yield df.slice(offset, n_rows).collect()
