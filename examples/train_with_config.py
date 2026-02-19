from definitions import ROOT_DIR
from model_training import run_all_experiments
from utils import setup_logging
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run model training from configuration files")
    parser.add_argument(
        "--global-config",
        type=str,
        default="config/model_training/global_default.yaml",
        help="Path to global configuration file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_training/models/mlp_default.yaml",
        help="Path to model-specific configuration file",
    )

    args = parser.parse_args()

    setup_logging(level="INFO")

    global_config_path = ROOT_DIR / args.global_config
    model_config_path = ROOT_DIR / args.model_config

    run_all_experiments(global_config_path, model_config_path)


if __name__ == "__main__":
    main()
