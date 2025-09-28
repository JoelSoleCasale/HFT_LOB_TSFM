"""Simple test runner script for TFG project."""

import sys
import subprocess
from definitions import ROOT_DIR


def run_command(cmd, description, cwd):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"Working Directory: {cwd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True, cwd=cwd
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Main test runner."""
    print("TFG Project Test Runner")
    print(f"Project Root: {ROOT_DIR}")
    print("======================")

    # Check if we're in the right directory
    if not (ROOT_DIR / "src").exists() or not (ROOT_DIR / "tests").exists():
        print(f"Error: Project structure not found at ROOT_DIR: {ROOT_DIR}")
        sys.exit(1)

    # Install test dependencies
    if not run_command(
        "uv sync --extra test", "Installing test dependencies", cwd=ROOT_DIR
    ):
        print("Failed to install test dependencies")
        sys.exit(1)

    # Run unit tests
    if not run_command(
        "pytest tests/unit/ -v -m unit", "Running unit tests", cwd=ROOT_DIR
    ):
        print("Unit tests failed")
        sys.exit(1)

    # Run integration tests (optional, can be skipped if slow)
    run_integration = input("\nRun integration tests? (y/N): ").lower().startswith("y")
    if run_integration:
        if not run_command(
            "pytest tests/integration/ -v -m integration",
            "Running integration tests",
            cwd=ROOT_DIR,
        ):
            print("Integration tests failed")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
