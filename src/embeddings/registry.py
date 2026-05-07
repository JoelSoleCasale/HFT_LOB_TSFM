"""Registry for embedding generators"""

from typing import Type, Any
from embeddings.base import BaseEmbeddingGenerator


class EmbeddingGeneratorRegistry:
    """Registry for embedding generators"""

    _generators: dict[str, Type[BaseEmbeddingGenerator]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an embedding generator under the given name."""

        def wrapper(generator_cls: Type[BaseEmbeddingGenerator]):
            cls._generators[name] = generator_cls
            return generator_cls

        return wrapper

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseEmbeddingGenerator:
        """Create and return an instance of the named generator."""
        if name not in cls._generators:
            raise ValueError(
                f"Unknown embedding generator: {name}. "
                f"Available generators: {list(cls._generators.keys())}"
            )
        return cls._generators[name](config)

    @classmethod
    def list_generators(cls) -> list[str]:
        """Return all registered generator names."""
        return list(cls._generators.keys())
