# Setup

1. Install [uv](https://docs.astral.sh/uv/)
2. Install dependencies: `uv sync`
3. Set up pre-commit hooks: `uv run poe hooks`
4. To download the data, an API key must be set in a `.env` file with the name `CRYPTOHFTDATA_API_KEY`.

## Testing

To run the tests, you can use the following commands:

- `uv run poe install-test`: Install test dependencies.
- `uv run poe test`: Run all tests.
- `uv run poe test-unit`: Run unit tests.
- `uv run poe test-integration`: Run integration tests.
- `uv run poe test-cov`: Run tests with coverage report.
- `uv run poe clean-test`: Clean test artifacts.
