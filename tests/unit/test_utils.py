"""Unit tests for utils.py module."""

import tempfile
from pathlib import Path
from datetime import date
import polars as pl
import pytest
from src.utils import cache_func, iter_slices
from src.data_manager.downloader.data_downloader_request import RawDataRequest


class TestIterSlices:
    """Tests for iter_slices function."""

    def test_iter_slices_dataframe_single_slice(self):
        """Test iter_slices with DataFrame that fits in one slice."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        slices = list(iter_slices(df, n_rows=5))

        assert len(slices) == 1
        assert slices[0].equals(df)

    def test_iter_slices_dataframe_multiple_slices(self):
        """Test iter_slices with DataFrame that requires multiple slices."""
        df = pl.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
        slices = list(iter_slices(df, n_rows=3))

        assert len(slices) == 4  # 10 rows / 3 = 3.33, so 4 slices

        # Check first slice
        expected_first = pl.DataFrame({"a": [0, 1, 2], "b": [10, 11, 12]})
        assert slices[0].equals(expected_first)

        # Check last slice
        expected_last = pl.DataFrame({"a": [9], "b": [19]})
        assert slices[3].equals(expected_last)

    def test_iter_slices_empty_dataframe(self):
        """Test iter_slices with empty DataFrame."""
        df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Int64, "b": pl.Int64})
        slices = list(iter_slices(df, n_rows=5))

        assert len(slices) == 0

    def test_iter_slices_lazyframe_single_slice(self):
        """Test iter_slices with LazyFrame that fits in one slice."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lazy_df = df.lazy()
        slices = list(iter_slices(lazy_df, n_rows=5))

        assert len(slices) == 1
        assert slices[0].equals(df)

    def test_iter_slices_lazyframe_multiple_slices(self):
        """Test iter_slices with LazyFrame that requires multiple slices."""
        df = pl.DataFrame({"a": list(range(7)), "b": list(range(7, 14))})
        lazy_df = df.lazy()
        slices = list(iter_slices(lazy_df, n_rows=3))

        assert len(slices) == 3  # 7 rows / 3 = 2.33, so 3 slices

        # Check first slice
        expected_first = pl.DataFrame({"a": [0, 1, 2], "b": [7, 8, 9]})
        assert slices[0].equals(expected_first)

        # Check last slice
        expected_last = pl.DataFrame({"a": [6], "b": [13]})
        assert slices[2].equals(expected_last)

    def test_iter_slices_exact_division(self):
        """Test iter_slices when rows divide evenly into slices."""
        df = pl.DataFrame({"a": list(range(6)), "b": list(range(6, 12))})
        slices = list(iter_slices(df, n_rows=2))

        assert len(slices) == 3

        # All slices should have exactly 2 rows
        for slice_df in slices:
            assert slice_df.height == 2

    def test_iter_slices_default_n_rows(self):
        """Test iter_slices with default n_rows parameter."""
        df = pl.DataFrame({"a": list(range(5)), "b": list(range(5, 10))})
        slices = list(iter_slices(df))  # Uses default n_rows=10_000

        assert len(slices) == 1
        assert slices[0].equals(df)


class TestCacheFunc:
    """Tests for cache_func decorator."""

    @pytest.fixture
    def dummy_func(self):
        """Dummy function that returns a simple DataFrame."""

        def _func(request: RawDataRequest):
            return pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        return _func

    @pytest.fixture
    def dummy_func_none(self):
        """Dummy function that returns None."""

        def _func(request: RawDataRequest):
            return None

        return _func

    def test_cache_func_saves_and_loads(self, dummy_func):
        """Test that cache_func saves data on first call and loads from cache on second call."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cached_func = cache_func(dummy_func, cache_root)
            req = RawDataRequest(
                data_type="test", symbol="BTC", exchange="binance", date=date(2025, 1, 1)
            )

            # First call: should save to cache
            df1, from_cache1 = cached_func(req)
            assert not from_cache1
            assert (cache_root / req.get_path()).exists()

            # Second call: should load from cache
            df2, from_cache2 = cached_func(req)
            assert from_cache2
            assert df2.equals(df1)

    def test_cache_func_skip_load(self, dummy_func):
        """Test cache_func with load_cached_data=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cached_func = cache_func(dummy_func, cache_root)
            req = RawDataRequest(
                data_type="test", symbol="ETH", exchange="binance", date=date(2025, 1, 2)
            )

            # Save to cache first
            df1, _ = cached_func(req)
            assert (cache_root / req.get_path()).exists()

            # Now, use load_cached_data=False
            cached_func_skip = cache_func(dummy_func, cache_root, load_cached_data=False)
            df2, from_cache = cached_func_skip(req)
            assert from_cache
            assert df2 is None

    def test_cache_func_none_result(self, dummy_func_none):
        """Test that None results are not cached."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cached_func = cache_func(dummy_func_none, cache_root)
            req = RawDataRequest(
                data_type="test", symbol="BTC", exchange="binance", date=date(2025, 1, 3)
            )

            df, from_cache = cached_func(req)
            assert df is None
            assert not from_cache
            assert not (cache_root / req.get_path()).exists()

    def test_cache_func_creates_directories(self, dummy_func):
        """Test that cache_func creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cached_func = cache_func(dummy_func, cache_root)
            req = RawDataRequest(
                data_type="nested/test", symbol="BTC", exchange="binance", date=date(2025, 1, 4)
            )

            # Call function to create cache
            df, from_cache = cached_func(req)

            # Check that nested directories were created
            cache_path = cache_root / req.get_path()
            assert cache_path.exists()
            assert cache_path.parent.exists()

    def test_cache_func_with_args_kwargs(self, dummy_func):
        """Test that cache_func passes through additional args and kwargs."""

        def func_with_args(request: RawDataRequest, multiplier=1, **kwargs):
            base_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
            return base_df.with_columns(pl.all() * multiplier)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cached_func = cache_func(func_with_args, cache_root)
            req = RawDataRequest(
                data_type="test", symbol="BTC", exchange="binance", date=date(2025, 1, 5)
            )

            # First call with args
            df1, from_cache1 = cached_func(req, multiplier=2, extra_arg="test")
            assert not from_cache1
            expected_df = pl.DataFrame({"a": [2, 4], "b": [6, 8]})
            assert df1.equals(expected_df)

            # Second call should load from cache (ignoring new args)
            df2, from_cache2 = cached_func(req, multiplier=5, different_arg="ignored")
            assert from_cache2
            assert df2.equals(df1)  # Should be same as first call, not with multiplier=5

    def test_cache_func_different_requests_different_caches(self, dummy_func):
        """Test that different requests create different cache files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            cached_func = cache_func(dummy_func, cache_root)

            req1 = RawDataRequest(
                data_type="test", symbol="BTC", exchange="binance", date=date(2025, 1, 1)
            )
            req2 = RawDataRequest(
                data_type="test", symbol="ETH", exchange="binance", date=date(2025, 1, 1)
            )

            # Both should create separate cache files
            df1, from_cache1 = cached_func(req1)
            df2, from_cache2 = cached_func(req2)

            assert not from_cache1
            assert not from_cache2
            assert (cache_root / req1.get_path()).exists()
            assert (cache_root / req2.get_path()).exists()
            assert req1.get_path() != req2.get_path()
