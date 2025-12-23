# Setup

1. Install [uv](https://docs.astral.sh/uv/)
2. Install dependencies: `uv sync` (use `uv sync --extra cu126`, `uv sync --extra cu128`, or `uv sync --extra cu130` to include dev dependencies for specific CUDA versions)
3. Set up pre-commit hooks: `uv run poe hooks`
4. To download the data, an API key must be set in a `.env` file with the name `CRYPTOHFTDATA_API_KEY`.

## Testing

To run the tests, you can use the following commands:

- `uv run poe install-test`: Install test dependencies.
- `uv run poe test`: Run all tests (excluding slow tests).
- `uv run poe test-full`: Run all tests including slow tests (embedding tests).
- `uv run poe test-unit`: Run unit tests.
- `uv run poe test-integration`: Run integration tests (excluding slow tests).
- `uv run poe test-integration-full`: Run all integration tests including slow tests.
- `uv run poe test-cov`: Run tests with coverage report (excluding slow tests).
- `uv run poe clean-test`: Clean test artifacts.

**Note**: Slow tests (marked with `@pytest.mark.slow`) include embedding generation tests that can take several minutes to run.
