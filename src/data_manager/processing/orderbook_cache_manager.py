"""Cache management for orderbook data."""

from pathlib import Path
from loguru import logger

from core.orderbook import OrderBook
from data_manager.processing.orderbook_request import OrderBookSnapshotRequest
from data_manager.processing.orderbook_processor import OrderBookProcessor


class OrderBookCacheManager:
    """Manages caching for orderbook data with intelligent cache reuse."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.processor = OrderBookProcessor(cache_root)

    def get_cached_orderbook(
        self,
        request: OrderBookSnapshotRequest,
        force_regenerate: bool = False,
    ) -> OrderBook:
        """
        Get orderbook data from cache or generate it.
        Uses higher-level caches when available and prunes lower-level caches.
        """
        cache_dir = self.cache_root / "orderbook_snapshots" / request.exchange / request.symbol
        cache_dir.mkdir(parents=True, exist_ok=True)

        requested_cache_path = self.cache_root / request.get_path()

        if not force_regenerate:
            if requested_cache_path.exists():
                logger.info(f"Loading cached data from {requested_cache_path}")
                return OrderBook.from_parquet(requested_cache_path)

            # Check for higher-level caches
            existing_caches = list(
                cache_dir.glob(f"{request.date.strftime('%Y-%m-%d')}_L*.parquet")
            )
            higher_caches = []
            for path in existing_caches:
                try:
                    level_str = path.stem.split("_L")[-1]
                    existing_level = int(level_str)
                    if existing_level > request.levels:
                        higher_caches.append((existing_level, path))
                except (IndexError, ValueError):
                    continue

            if higher_caches:
                # Use the lowest of the higher caches
                highest_level, highest_cache_path = min(higher_caches, key=lambda x: x[0])
                logger.info(
                    f"Found higher-level cache at {highest_cache_path} (L{highest_level}), "
                    f"using it to generate L{request.levels} data."
                )

                return OrderBook.from_parquet(highest_cache_path).select_levels(request.levels)

        # Generate data from scratch
        data = self.processor.generate_full_event_sampled(request)
        logger.info(f"Saving newly generated data to {requested_cache_path}")
        data.to_parquet(requested_cache_path)

        # Prune lower-level caches
        self._prune_lower_level_caches(request, requested_cache_path)

        return data

    def _prune_lower_level_caches(
        self,
        request: OrderBookSnapshotRequest,
        requested_cache_path: Path,
    ) -> None:
        """Remove lower-level cache files to save space."""
        cache_dir = self.cache_root / "orderbook_snapshots" / request.exchange / request.symbol
        existing_caches = list(cache_dir.glob(f"{request.date.strftime('%Y-%m-%d')}_L*.parquet"))

        for path in existing_caches:
            if path == requested_cache_path:
                continue
            try:
                level_str = path.stem.split("_L")[-1]
                existing_level = int(level_str)
                if existing_level < request.levels:
                    logger.info(f"Pruning lower-level cache: {path}")
                    path.unlink()
            except (IndexError, ValueError):
                continue
