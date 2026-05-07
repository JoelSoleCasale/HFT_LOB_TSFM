"""Feature extractor for raw LOB model input formats"""

from features.base.feature_extractor import BaseFeatureExtractor
from features.registry import FeatureExtractorRegistry
import polars as pl
from features.base.input_space import InputSpace


class RawLOBFeatures(BaseFeatureExtractor):
    """Extracts raw LOB features (price/qty for each level) in [ask, bid] order per level."""

    REGISTRY_NAME: str  # subclasses define this

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.levels = self.config.get("levels", 10)
        self.dependencies = ["orderbook_snapshots"]
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
        self.validate_input(input_space)
        df: pl.LazyFrame = input_space.orderbook_snapshots.df
        columns = [pl.col("timestamp")]
        for level in range(1, self.levels + 1):
            columns.extend(
                [
                    pl.col(f"ask{level}_price"),
                    pl.col(f"ask{level}_qty"),
                    pl.col(f"bid{level}_price"),
                    pl.col(f"bid{level}_qty"),
                ]
            )
        return df.select(columns)

    def get_expected_feature_count(self) -> int:
        return self.levels * 4


@FeatureExtractorRegistry.register("deeplob_features")
class DeepLOBFeatures(RawLOBFeatures):
    """Raw LOB features for DeepLOB (40 features for 10 levels)."""


@FeatureExtractorRegistry.register("ctabl_features")
class CTABLFeatures(RawLOBFeatures):
    """Raw LOB features for CTABL."""


@FeatureExtractorRegistry.register("deeplob_attention_features")
class DeepLOBAttentionFeatures(RawLOBFeatures):
    """Raw LOB features for DeepLOB-Attention."""


@FeatureExtractorRegistry.register("axial_lob_features")
class AxialLOBFeatures(RawLOBFeatures):
    """Raw LOB features for Axial-LOB."""


@FeatureExtractorRegistry.register("tlob_features")
class TLOBFeatures(RawLOBFeatures):
    """Raw LOB features for TLOB."""
