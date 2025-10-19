"""Pipeline classes for composing extractors"""

import polars as pl
from features.base.feature_extractor import BaseFeatureExtractor
from features.base.input_space import InputSpace


class FeaturePipeline:
    """Compose multiple feature extractors"""

    def __init__(self):
        self.extractors: list[BaseFeatureExtractor] = []

    def add_extractor(self, extractor: BaseFeatureExtractor):
        self.extractors.append(extractor)
        return self

    def extract_all(self, input_space: InputSpace) -> pl.LazyFrame:
        if not self.extractors:
            raise ValueError("No extractors in pipeline")
        features = []
        for extractor in self.extractors:
            try:
                feature_df = extractor.extract(input_space)
                features.append(feature_df)
            except Exception as e:
                print(f"Error in {extractor.__class__.__name__}: {e}")
                raise
        result = features[0]
        for feat in features[1:]:
            result = result.join(feat, on="timestamp", how="left")
        return result

    def get_all_feature_names(self) -> list[str]:
        return [name for ext in self.extractors for name in ext.get_feature_names()]

    def get_all_dependencies(self) -> list[str]:
        deps = set()
        for ext in self.extractors:
            deps.update(ext.get_dependencies())
        return list(deps)
