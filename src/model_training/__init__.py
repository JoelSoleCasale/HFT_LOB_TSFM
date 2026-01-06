from .config import load_and_generate_experiments
from .runner import run_experiment
from loguru import logger


def run_all_experiments(global_config_path: str, model_config_path: str):
    experiments = load_and_generate_experiments(global_config_path, model_config_path)
    logger.info(f"Generated {len(experiments)} experiments.")

    for i, exp_config in enumerate(experiments):
        logger.info(f"Running experiment {i+1}/{len(experiments)}")
        try:
            run_experiment(exp_config)
        except Exception as e:
            logger.exception(f"Experiment {i+1} failed: {e}")
