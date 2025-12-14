"""
Script to generate reference embeddings for integration tests.

This script generates the reference embeddings that will be used to validate
the consistency of the embedding generation process. Run this script once to
generate the reference files, then commit them to the repository.

Usage:
    uv run python tests/integration/embeddings/generate_references.py
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from embeddings.registry import EmbeddingGeneratorRegistry

# Import from local test_fixtures module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_fixtures import (
    generate_sample_time_series,
    generate_sample_time_series_batch,
    SINGLE_CONFIGS,
    BATCH_CONFIGS,
)


def generate_embeddings(configs, data, context_length, output_dir, is_batch=False):
    """
    Generate embeddings for a list of configurations.

    Args:
        configs: List of configuration dictionaries
        data: Input data (single or batch)
        context_length: Context length for embeddings
        output_dir: Directory to save embeddings
        is_batch: Whether this is batch processing
    """
    for item in configs:
        name = item["name"]
        config = item["config"]
        output_path = output_dir / f"{name}.npy"

        print(f"Generating: {name}")
        print(f"  Config: {config}")

        try:
            # Create generator
            generator = EmbeddingGeneratorRegistry.create("chronos", config=config)

            if is_batch:
                # Generate embeddings for each item in the batch
                batch_embeddings = []
                for i in range(data.shape[0]):
                    embedding = generator.generate_embedding(
                        data[i], context_length=context_length
                    )
                    batch_embeddings.append(embedding)
                result = np.array(batch_embeddings)
            else:
                # Generate single embedding
                result = generator.generate_embedding(data, context_length=context_length)

            # Save to file
            np.save(output_path, result)

            print(f"  ✓ Saved to: {output_path}")
            print(f"  Shape: {result.shape}")
            print(f"  Dtype: {result.dtype}")
            print(f"  Range: [{result.min():.6f}, {result.max():.6f}]")
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()
            continue


def main():
    """Generate all reference embeddings."""
    # Setup
    output_dir = Path(__file__).parent.parent.parent / "data_sample" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_data = generate_sample_time_series(seed=42)
    sample_batch = generate_sample_time_series_batch(seed=42)
    context_length = 128

    print("=" * 80)
    print("Generating Reference Embeddings for Integration Tests")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample batch shape: {sample_batch.shape}")
    print(f"Context length: {context_length}")
    print()

    # Generate single embeddings
    print("=" * 80)
    print("SINGLE EMBEDDINGS")
    print("=" * 80)
    print()
    generate_embeddings(SINGLE_CONFIGS, sample_data, context_length, output_dir, is_batch=False)

    # Generate batch embeddings
    print("=" * 80)
    print("BATCH EMBEDDINGS")
    print("=" * 80)
    print()
    generate_embeddings(BATCH_CONFIGS, sample_batch, context_length, output_dir, is_batch=True)

    print("=" * 80)
    print("Reference generation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the generated files in:", output_dir)
    print("2. Commit the reference files to the repository")
    print("3. Run the integration tests: uv run poe test-integration")


if __name__ == "__main__":
    main()
