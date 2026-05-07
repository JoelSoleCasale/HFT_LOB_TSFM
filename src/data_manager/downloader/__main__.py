import yaml
from loguru import logger
from datetime import date, timedelta
import argparse

from data_manager.downloader.data_downloader import DataDownloader

from definitions import ROOT_DIR
from utils import setup_logging


def load_config() -> dict:
    """Loads configuration from YAML files."""
    config_path = ROOT_DIR / "config" / "config.yaml"
    data_downloader_config_path = (
        ROOT_DIR / "config" / "data_manager" / "downloader" / "default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(data_downloader_config_path, "r") as f:
        data_manager_config = yaml.safe_load(f)
    config["data_manager"] = data_manager_config
    return config


def parse_args(default_cfg) -> argparse.Namespace:
    """
    Parse command-line arguments, using defaults from the configuration.
    """
    parser = argparse.ArgumentParser(
        description="Download high-frequency crypto data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Set parser defaults from the loaded configuration
    parser.set_defaults(
        log_level=default_cfg["log_level"],
        data_folder=str(ROOT_DIR / default_cfg["data_folder"]),
        data_type=default_cfg["data_manager"]["data_type"],
        symbol=default_cfg["data_manager"]["symbol"],
        exchange=default_cfg["data_manager"]["exchange"],
        relevant_features_path=default_cfg["data_manager"]["relevant_features_path"],
        reference_ts=default_cfg["data_manager"]["reference_ts"],
        start_date=default_cfg["data_manager"]["dates"]["start_date"],
        end_date=default_cfg["data_manager"]["dates"]["end_date"],
    )

    parser.add_argument("--log_level", type=str, help="Logging level.")
    parser.add_argument("--data_folder", type=str, help="Base folder for data.")
    parser.add_argument("--data_type", type=str, nargs="+", help="Data types to download.")
    parser.add_argument("--symbol", type=str, help="Trading symbol.")
    parser.add_argument("--exchange", type=str, help="Exchange.")
    parser.add_argument(
        "--relevant_features_path",
        type=str,
        help="Path to YAML file with relevant features to keep.",
    )
    parser.add_argument(
        "--reference_ts",
        type=str,
        choices=["received_time", "event_time"],
        help="Timestamp reference to use for sorting.",
    )
    parser.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD format.")
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date in YYYY-MM-DD format. Defaults to today if not provided.",
    )
    parser.add_argument(
        "--overwrite_existing",
        action=argparse.BooleanOptionalAction,
        help="If true, overwrite files that already exist.",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main function to run the data downloader with YAML configuration and command-line overrides.
    """
    default_cfg = load_config()
    args = parse_args(default_cfg)

    setup_logging(level=args.log_level)

    logger.debug("Final Configuration:\n" + yaml.dump(vars(args)))

    downloader = DataDownloader(
        base_folder=args.data_folder, relevant_features_path=args.relevant_features_path
    )

    # Handle dates
    start_date_str = args.start_date
    end_date_str = args.end_date

    start_date = date.fromisoformat(start_date_str)
    end_date = date.today() if end_date_str is None else date.fromisoformat(end_date_str)

    days = [
        (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end_date - start_date).days + 1)
    ]

    downloader.download_data(
        data_type=args.data_type,
        symbol=args.symbol,
        exchange=args.exchange,
        date_val=days,
        skip_existing=not args.overwrite_existing,
        reference_ts=args.reference_ts,
    )


if __name__ == "__main__":
    main()
