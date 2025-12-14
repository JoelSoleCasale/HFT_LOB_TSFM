"""Registry for embedding generators"""

from typing import Type, Any
from embeddings.base import BaseEmbeddingGenerator


class EmbeddingGeneratorRegistry:
    """Registry for embedding generators"""

    _generators: dict[str, Type[BaseEmbeddingGenerator]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an embedding generator.

        Args:
            name: Name to register the generator under

        Example:
            @EmbeddingGeneratorRegistry.register("chronos")
            class ChronosEmbeddingGenerator(BaseEmbeddingGenerator):
                ...
        """

        def wrapper(generator_cls: Type[BaseEmbeddingGenerator]):
            cls._generators[name] = generator_cls
            return generator_cls

        return wrapper

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseEmbeddingGenerator:
        """
        Create an embedding generator instance.

        Args:
            name: Name of the generator to create
            config: Configuration dictionary for the generator

        Returns:
            Instance of the embedding generator

        Raises:
            ValueError: If the generator name is not registered
        """
        if name not in cls._generators:
            raise ValueError(
                f"Unknown embedding generator: {name}. "
                f"Available generators: {list(cls._generators.keys())}"
            )
        return cls._generators[name](config)

    @classmethod
    def list_generators(cls) -> list[str]:
        """
        List all registered embedding generator names.

        Returns:
            List of registered generator names
        """
        return list(cls._generators.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a generator is registered.

        Args:
            name: Name of the generator to check

        Returns:
            True if the generator is registered, False otherwise
        """
        return name in cls._generators
