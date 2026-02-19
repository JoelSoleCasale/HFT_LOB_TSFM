"""Tests to ensure no data leakage in feature extraction.

This module contains tests to verify that feature extractors don't use future
information when computing features for a given timestamp. The tests work by:
1. Extracting features from synthetic orderbook data
2. Modifying the data from a cutoff point k onwards
3. Verifying that features remain identical up to the (k-1)th sample
"""

import pytest
import polars as pl
import numpy as np

from core.orderbook import OrderBook
from features.base.feature_extractor import BaseFeatureExtractor
from features.base.input_space import InputSpace
from features.extractors.orderbook_features import (
    MidPriceFeatures,
    SpreadFeatures,
    OrderbookImbalanceFeatures,
    AdvancedOrderbookFeatures,
)


def create_synthetic_orderbook(n_samples: int = 100, levels: int = 5, seed: int = 42) -> OrderBook:
    """Create synthetic orderbook data for testing.

    Args:
        n_samples: Number of snapshots to generate
        levels: Number of orderbook levels
        seed: Random seed for reproducibility

    Returns:
        OrderBook with synthetic data
    """
    np.random.seed(seed)

    # Generate timestamps
    timestamps = np.arange(1000, 1000 + n_samples * 100, 100)

    # Generate base mid price with some trend and noise
    base_price = 50000.0
    price_trend = np.linspace(0, 100, n_samples)
    price_noise = np.random.normal(0, 10, n_samples)
    mid_prices = base_price + price_trend + price_noise

    # Generate orderbook data
    data = {"timestamp": timestamps}

    for level in range(1, levels + 1):
        # Ask prices increase with level
        ask_spread = level * 0.5 + np.random.uniform(0, 0.2, n_samples)
        data[f"ask{level}_price"] = mid_prices + ask_spread
        data[f"ask{level}_qty"] = np.random.uniform(1.0, 5.0, n_samples)

        # Bid prices decrease with level
        bid_spread = level * 0.5 + np.random.uniform(0, 0.2, n_samples)
        data[f"bid{level}_price"] = mid_prices - bid_spread
        data[f"bid{level}_qty"] = np.random.uniform(1.0, 5.0, n_samples)

    df = pl.DataFrame(data)
    return OrderBook(df)


def modify_orderbook_after_cutoff(
    orderbook: OrderBook, cutoff_index: int, seed: int = 123
) -> OrderBook:
    """Modify orderbook data after a cutoff index.

    Args:
        orderbook: Original orderbook
        cutoff_index: Index from which to modify data (inclusive)
        seed: Random seed for modifications

    Returns:
        New OrderBook with modified data after cutoff
    """
    np.random.seed(seed)
    df = orderbook.df

    # Collect if lazy
    if orderbook.is_lazy:
        df = df.collect()

    n_samples = len(df)
    if cutoff_index >= n_samples:
        raise ValueError(f"Cutoff index {cutoff_index} >= number of samples {n_samples}")

    # Get all columns except timestamp
    price_qty_cols = [col for col in df.columns if col != "timestamp"]

    # Create random modifications for each column after cutoff
    modified_df = df.clone()
    for col in price_qty_cols:
        original_values = df[col].to_numpy()
        modified_values = original_values.copy()

        # Modify values from cutoff onwards
        if "price" in col:
            # For prices, add random shifts
            modified_values[cutoff_index:] += np.random.uniform(
                -100, 100, n_samples - cutoff_index
            )
        else:  # qty columns
            # For quantities, multiply by random factors
            modified_values[cutoff_index:] *= np.random.uniform(0.5, 2.0, n_samples - cutoff_index)

        modified_df = modified_df.with_columns(pl.Series(col, modified_values))

    return OrderBook(modified_df)


def assert_no_data_leakage(
    extractor: BaseFeatureExtractor,
    orderbook: OrderBook,
    cutoff_index: int,
    tolerance: float = 1e-6,
) -> None:
    """Generic test for data leakage in a feature extractor.

    This test verifies that features extracted up to index (cutoff_index - 1)
    remain unchanged when data after cutoff_index is modified.

    Args:
        extractor: Feature extractor to test
        orderbook: Original orderbook data
        cutoff_index: Index from which data will be modified
        tolerance: Numerical tolerance for feature comparison

    Raises:
        AssertionError: If features differ before cutoff index
    """
    # Create input space with original data
    input_space_original = InputSpace(orderbook_snapshots=orderbook)

    # Extract features from original data
    features_original = extractor.extract(input_space_original)

    # Collect if lazy (to ensure we can compare)
    if isinstance(features_original, pl.LazyFrame):
        features_original = features_original.collect()

    # Create modified orderbook (data changed after cutoff)
    orderbook_modified = modify_orderbook_after_cutoff(orderbook, cutoff_index)
    input_space_modified = InputSpace(orderbook_snapshots=orderbook_modified)

    # Extract features from modified data
    features_modified = extractor.extract(input_space_modified)
    if isinstance(features_modified, pl.LazyFrame):
        features_modified = features_modified.collect()

    # Verify that features up to cutoff_index - 1 are identical
    n_rows_to_check = cutoff_index

    # Get feature columns (exclude timestamp)
    feature_cols = [col for col in features_original.columns if col != "timestamp"]

    for col in feature_cols:
        original_values = features_original[col].head(n_rows_to_check).to_numpy()
        modified_values = features_modified[col].head(n_rows_to_check).to_numpy()

        # Handle NaN values (they should be in the same positions)
        original_nan_mask = np.isnan(original_values)
        modified_nan_mask = np.isnan(modified_values)

        assert np.array_equal(
            original_nan_mask, modified_nan_mask
        ), f"NaN positions differ for column '{col}' before cutoff index {cutoff_index}"

        # Compare non-NaN values
        non_nan_mask = ~original_nan_mask
        if np.any(non_nan_mask):
            max_diff = np.max(
                np.abs(original_values[non_nan_mask] - modified_values[non_nan_mask])
            )
            assert (
                max_diff < tolerance
            ), f"Feature '{col}' differs before cutoff index {cutoff_index}. Max diff: {max_diff}"


@pytest.fixture(params=[100])
def synthetic_orderbook(request) -> OrderBook:
    """Fixture providing synthetic orderbook with parameterized size."""
    return create_synthetic_orderbook(n_samples=request.param, levels=5, seed=42)


@pytest.fixture(params=[10, 25, 50, 75])
def cutoff_indices(request, synthetic_orderbook) -> int:
    """Fixture providing various cutoff indices for testing."""
    n_samples = len(synthetic_orderbook)
    cutoff = request.param
    if cutoff >= n_samples:
        pytest.skip(f"Cutoff index {cutoff} >= number of samples {n_samples}")
    return cutoff


class TestMidPriceDataLeakage:
    """Test data leakage for MidPriceFeatures extractor."""

    def test_no_leakage(self, synthetic_orderbook, cutoff_indices):
        """Test that mid price features don't leak future data."""
        extractor = MidPriceFeatures()
        assert_no_data_leakage(
            extractor=extractor,
            orderbook=synthetic_orderbook,
            cutoff_index=cutoff_indices,
        )


class TestSpreadDataLeakage:
    """Test data leakage for SpreadFeatures extractor."""

    def test_no_leakage(self, synthetic_orderbook, cutoff_indices):
        """Test that spread features don't leak future data."""
        extractor = SpreadFeatures()
        assert_no_data_leakage(
            extractor=extractor,
            orderbook=synthetic_orderbook,
            cutoff_index=cutoff_indices,
        )


class TestOrderbookImbalanceDataLeakage:
    """Test data leakage for OrderbookImbalanceFeatures extractor."""

    @pytest.mark.parametrize("levels", [[1], [1, 2], [1, 2, 5]])
    def test_no_leakage(self, synthetic_orderbook, cutoff_indices, levels):
        """Test that orderbook imbalance features don't leak future data."""
        extractor = OrderbookImbalanceFeatures(config={"levels": levels})
        assert_no_data_leakage(
            extractor=extractor,
            orderbook=synthetic_orderbook,
            cutoff_index=cutoff_indices,
        )


class TestAdvancedOrderbookDataLeakage:
    """Test data leakage for AdvancedOrderbookFeatures extractor."""

    @pytest.mark.parametrize("levels", [3, 5])
    def test_no_leakage(self, synthetic_orderbook, cutoff_indices, levels):
        """Test that advanced orderbook features don't leak future data."""
        extractor = AdvancedOrderbookFeatures(config={"levels": levels})
        assert_no_data_leakage(
            extractor=extractor,
            orderbook=synthetic_orderbook,
            cutoff_index=cutoff_indices,
        )

    def test_no_leakage_with_lag_features(self, synthetic_orderbook, cutoff_indices):
        """Test that lagged features (log returns) don't leak future data.

        This is especially important for log return features which use shift().
        """
        extractor = AdvancedOrderbookFeatures(config={"levels": 5})

        # For lag features, we need to account for the first row being NaN
        # So we test from cutoff_index + 1 onwards if cutoff_index == 1
        if cutoff_indices <= 1:
            pytest.skip("Cutoff too small for lag feature testing")

        assert_no_data_leakage(
            extractor=extractor,
            orderbook=synthetic_orderbook,
            cutoff_index=cutoff_indices,
        )


class TestDataLeakageEdgeCases:
    """Test edge cases for data leakage detection."""

    def test_cutoff_at_beginning(self, synthetic_orderbook):
        """Test with cutoff at index 1 (minimal valid case)."""
        extractor = MidPriceFeatures()
        assert_no_data_leakage(
            extractor=extractor,
            orderbook=synthetic_orderbook,
            cutoff_index=1,
        )

    def test_cutoff_at_end(self, synthetic_orderbook):
        """Test with cutoff near the end."""
        n_samples = len(synthetic_orderbook)

        extractor = MidPriceFeatures()
        assert_no_data_leakage(
            extractor=extractor,
            orderbook=synthetic_orderbook,
            cutoff_index=n_samples - 2,
        )

    def test_small_orderbook(self):
        """Test with a very small orderbook."""
        small_orderbook = create_synthetic_orderbook(n_samples=10, levels=5, seed=123)
        extractor = MidPriceFeatures()

        assert_no_data_leakage(
            extractor=extractor,
            orderbook=small_orderbook,
            cutoff_index=5,
        )

    def test_different_seeds_produce_different_modifications(self):
        """Verify that our modification function actually changes the data."""
        orderbook = create_synthetic_orderbook(n_samples=50, levels=5, seed=42)

        modified1 = modify_orderbook_after_cutoff(orderbook, cutoff_index=25, seed=1)
        modified2 = modify_orderbook_after_cutoff(orderbook, cutoff_index=25, seed=2)

        # Collect dataframes if lazy
        df_orig = orderbook.df.collect() if orderbook.is_lazy else orderbook.df
        df_mod1 = modified1.df.collect() if modified1.is_lazy else modified1.df
        df_mod2 = modified2.df.collect() if modified2.is_lazy else modified2.df

        # Check that modifications differ from original after cutoff
        assert not df_orig.tail(25).equals(
            df_mod1.tail(25)
        ), "First modification didn't change data"
        assert not df_orig.tail(25).equals(
            df_mod2.tail(25)
        ), "Second modification didn't change data"
        assert not df_mod1.tail(25).equals(
            df_mod2.tail(25)
        ), "Different seeds produced same modifications"

        # Check that data before cutoff is unchanged
        assert df_orig.head(25).equals(df_mod1.head(25)), "Data before cutoff was changed"
        assert df_orig.head(25).equals(df_mod2.head(25)), "Data before cutoff was changed"


class TestAllExtractorsTogether:
    """Test all extractors with the same datasets for consistency."""

    @pytest.mark.parametrize(
        "extractor_class,config",
        [
            (MidPriceFeatures, None),
            (SpreadFeatures, None),
            (OrderbookImbalanceFeatures, {"levels": [1, 2, 5]}),
            (AdvancedOrderbookFeatures, {"levels": 5}),
        ],
    )
    def test_all_extractors_no_leakage(
        self, synthetic_orderbook, cutoff_indices, extractor_class, config
    ):
        """Test all extractors with same data for comprehensive validation."""
        extractor = extractor_class(config=config) if config else extractor_class()
        assert_no_data_leakage(
            extractor=extractor,
            orderbook=synthetic_orderbook,
            cutoff_index=cutoff_indices,
        )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
