"""Feature extractor for DeepLOB model input format"""

from features.base.feature_extractor import BaseFeatureExtractor
from features.registry import FeatureExtractorRegistry
import polars as pl
from features.base.input_space import InputSpace


@FeatureExtractorRegistry.register("deeplob_features")
class DeepLOBFeatures(BaseFeatureExtractor):
    """
    Extract and format orderbook features for DeepLOB model.

    DeepLOB expects 40 features representing 10 levels of the limit order book
    in the following order for each level:
        [ask_price, ask_volume, bid_price, bid_volume]

    This results in the feature order:
        [ask1_price, ask1_qty, bid1_price, bid1_qty,
         ask2_price, ask2_qty, bid2_price, bid2_qty,
         ...
         ask10_price, ask10_qty, bid10_price, bid10_qty]
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.levels = self.config.get("levels", 10)
        self.dependencies = ["orderbook_snapshots"]

        # Define feature names in DeepLOB expected order
        self.feature_names = []
        for level in range(1, self.levels + 1):
            self.feature_names.extend(
                [
                    f"ask{level}_price",
                    f"ask{level}_qty",
                    f"bid{level}_price",
                    f"bid{level}_qty",
                ]
            )

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        """
        Extract orderbook features in DeepLOB format.

        Args:
            input_space: InputSpace containing orderbook_snapshots

        Returns:
            LazyFrame with timestamp and 40 features (for 10 levels) in the correct order
        """
        self.validate_input(input_space)
        df: pl.LazyFrame = input_space.orderbook_snapshots.df

        # Build the column selection in DeepLOB order
        # Start with timestamp
        columns_to_select = [pl.col("timestamp")]

        # Add features for each level in the correct order
        for level in range(1, self.levels + 1):
            columns_to_select.extend(
                [
                    pl.col(f"ask{level}_price"),
                    pl.col(f"ask{level}_qty"),
                    pl.col(f"bid{level}_price"),
                    pl.col(f"bid{level}_qty"),
                ]
            )

        # Select and reorder columns
        return df.select(columns_to_select)

    def get_expected_feature_count(self) -> int:
        """Return the expected number of features (4 per level)"""
        return self.levels * 4
