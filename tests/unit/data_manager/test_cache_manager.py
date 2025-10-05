import tempfile
from pathlib import Path
import polars as pl
from src.data_manager.cache_manager import cache_func


class DummyRequest:
    def __init__(self, name):
        self.name = name

    def get_path(self):
        return Path(f"{self.name}.parquet")


def dummy_func(request: DummyRequest):
    return pl.DataFrame({"a": [1, 2], "b": [3, 4]})


def test_cache_func_saves_and_loads():
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_root = Path(temp_dir)
        cached_func = cache_func(dummy_func, cache_root)
        req = DummyRequest("test1")
        # First call: should save to cache
        df1, from_cache1 = cached_func(req)
        assert not from_cache1
        assert (cache_root / req.get_path()).exists()
        # Second call: should load from cache
        df2, from_cache2 = cached_func(req)
        assert from_cache2
    assert df2.equals(df1)


def test_cache_func_skip_load():
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_root = Path(temp_dir)
        cached_func = cache_func(dummy_func, cache_root)
        req = DummyRequest("test2")
        # Save to cache
        df1, _ = cached_func(req)
        # Now, use load_cached_data=False
        cached_func_skip = cache_func(dummy_func, cache_root, load_cached_data=False)
        df2, from_cache = cached_func_skip(req)
        assert from_cache
        assert df2 is None
