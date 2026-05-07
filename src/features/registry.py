"""Registry for feature and label extractors"""

from typing import Any, Type
from features.base.feature_extractor import BaseFeatureExtractor
from features.base.label_extractor import BaseLabelExtractor


class FeatureExtractorRegistry:
    """Registry for feature extractors"""

    _extractors: dict[str, Type[BaseFeatureExtractor]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(extractor_cls):
            cls._extractors[name] = extractor_cls
            return extractor_cls

        return wrapper

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseFeatureExtractor:
        if name not in cls._extractors:
            raise ValueError(f"Unknown extractor: {name}")
        return cls._extractors[name](config)

    @classmethod
    def list_extractors(cls) -> list[str]:
        return list(cls._extractors.keys())


class LabelExtractorRegistry:
    """Registry for label extractors"""

    _extractors: dict[str, Type[BaseLabelExtractor]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(extractor_cls):
            cls._extractors[name] = extractor_cls
            return extractor_cls

        return wrapper

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseLabelExtractor:
        if name not in cls._extractors:
            raise ValueError(f"Unknown label extractor: {name}")
        return cls._extractors[name](config)

    @classmethod
    def list_extractors(cls) -> list[str]:
        return list(cls._extractors.keys())
