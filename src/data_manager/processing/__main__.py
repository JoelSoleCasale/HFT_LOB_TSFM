import yaml
from loguru import logger
from datetime import date
import argparse
import sys

from definitions import ROOT_DIR
from data_manager.processing.incremental_OB_sampler import IncrementalOBSampler


def load_config():
    """Loads configuration from YAML files."""
    config_path = ROOT_DIR / "config" / "config.yaml"
    processing_config_path = (
        ROOT_DIR / "config" / "data_manager" / "processing" / "default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(processing_config_path, "r") as f:
        processing_config = yaml.safe_load(f)
    config["data_manager_processing"] = processing_config
    return config


def parse_args(default_cfg):
    """
    Parse command-line arguments, using defaults from the configuration.
    """
    parser = argparse.ArgumentParser(
        description="Precomputes incremental order book snapshots to the IncrementalOBSampler cache."
    )

    # Set parser defaults from the loaded configuration
    processing_cfg = default_cfg["data_manager_processing"]
    parser.set_defaults(
        log_level=default_cfg["log_level"],
        cache_root=str(ROOT_DIR / default_cfg["data_folder"]),
        exchange=processing_cfg["exchange"],
        symbol=processing_cfg["symbol"],
        levels=processing_cfg["levels"],
        force_regenerate=processing_cfg["force_regenerate"],
        start_date=processing_cfg["dates"]["start_date"],
        end_date=processing_cfg["dates"]["end_date"],
    )

    parser.add_argument("--log_level", type=str, help="Logging level.")
    parser.add_argument(
        "--cache_root", type=str, help="Root folder for data and cache."
    )
    parser.add_argument("--exchange", type=str, nargs="+", help="Exchange(s).")
    parser.add_argument("--symbol", type=str, nargs="+", help="Trading symbol(s).")
    parser.add_argument("--levels", type=int, help="Number of order book levels.")
    parser.add_argument(
        "--force_regenerate",
        action=argparse.BooleanOptionalAction,
        help="If true, force regeneration of cached data.",
    )
    parser.add_argument(
        "--start_date", type=str, help="Start date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date in YYYY-MM-DD format. Defaults to today if not provided.",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main function to run the data sampler with YAML configuration and command-line overrides.
    """
    default_cfg = load_config()
    args = parse_args(default_cfg)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level, colorize=True)

    logger.debug("Final Configuration:\n" + yaml.dump(vars(args)))

    sampler = IncrementalOBSampler(cache_root=args.cache_root)

    # Handle dates
    start_date = date.fromisoformat(args.start_date)
    end_date = (
        date.today() if args.end_date is None else date.fromisoformat(args.end_date)
    )

    sampler.precompute_full_snapshots(
        exchange=args.exchange,
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        levels=args.levels,
        force_regenerate=args.force_regenerate,
    )


if __name__ == "__main__":
    main()
