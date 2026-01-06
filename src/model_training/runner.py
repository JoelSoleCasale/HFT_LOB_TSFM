from typing import Dict, Any
from models import (
    ModelConfig,
    MLPConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
    train_model,
)
from models.architectures.lob import DeepLOBConfig, CTABLConfig
from model_training.data_loader import prepare_data
from model_training.config import parse_list_arg
from loguru import logger


def get_architecture_config(
    model_type: str, params: Dict[str, Any], input_size: int, output_size: int
) -> Any:
    """Factory for architecture config."""
    model_type = model_type.lower()
    if model_type == "mlp":
        hidden_sizes = parse_list_arg(params.get("hidden_sizes", [64]), type=int)
        return MLPConfig(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=params.get("dropout", 0.0),
            sequence_length=params.get("sequence_length", 1),
        )
    elif model_type == "deeplob":
        return DeepLOBConfig(
            input_size=input_size,
            output_size=output_size,
            conv_filters=params.get("conv_filters", 16),
            inception_filters=params.get("inception_filters", 32),
            lstm_hidden_size=params.get("lstm_hidden_size", 32),
            dropout=params.get("dropout", 0.1),
        )
    elif model_type == "ctabl":
        return CTABLConfig(
            input_size=input_size,
            output_size=output_size,
            time_steps=params.get("sequence_length", 10),
            # Default dims as per paper/example
            hidden_dims=((120, 5),),
            output_dims=(3, 1),
            activation=params.get("activation", "relu"),
            dropout=params.get("dropout", 0.1),
        )
    # Add other models here
    raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(config: Dict[str, Any]):
    """
    Run a single training experiment based on config dictionary.
    """

    # 1. Prepare Data
    # The config passed here is the merged config (global + model specific variant)
    features, labels, metadata = prepare_data(config)

    # Extract configs
    data_conf = config.get("data", {})
    train_conf = config.get("training", {})
    log_conf = config.get("logging", {})
    model_conf = config.get("model", {})
    label_conf = config.get("labels", {})

    # 2. Build Model Config objects

    # DataConfig
    dc = DataConfig(
        sequence_length=model_conf.get("sequence_length", 1),
        stride=data_conf.get("stride", 5),
        train_split=data_conf.get("train_split", 0.8),
        val_split=data_conf.get("val_split", 0.1),
        test_split=data_conf.get("test_split", 0.1),
        batch_size=data_conf.get("batch_size", 1024),
        device=data_conf.get("device", "cuda"),
        # num_classes...
    )

    # TrainingConfig
    loss_params = train_conf.get("loss_params", {})
    # Inject class weights if focal/weighted loss
    if config.get("use_class_weights", True):
        # We might want to deep copy to not mutate the passed config
        loss_params = loss_params.copy()
        loss_params["alpha"] = metadata["class_weights"]

    tc = TrainingConfig(
        learning_rate=train_conf.get("learning_rate", 0.001),
        num_epochs=train_conf.get("num_epochs", 100),
        early_stopping_patience=train_conf.get("early_stopping_patience", 10),
        optimizer=train_conf.get("optimizer", "adamw"),
        scheduler=train_conf.get("scheduler", "cosine"),
        loss_function=train_conf.get("loss_function", "focal"),
        loss_params=loss_params,
        mixed_precision=train_conf.get("mixed_precision", False),
    )

    # Architecture
    arch_config = get_architecture_config(
        model_type=model_conf.get("type", "mlp"),
        params=model_conf,
        input_size=metadata["input_size"],
        output_size=3,  # Assuming 3 classes for now
    )

    # Logging
    # Construct experiment name dynamically if not provided or to ensure uniqueness?
    # For now take from config
    exp_name = log_conf.get("experiment_name_prefix", "exp")

    # Add key parameters to experiment name for clarity
    if "hidden_sizes" in model_conf:
        exp_name += f"_mlp{model_conf['hidden_sizes']}"
    if model_conf.get("type") == "deeplob":
        exp_name += "_deeplob"
    if model_conf.get("type") == "ctabl":
        exp_name += "_ctabl"

    if "pca_components" in data_conf and data_conf["pca_components"]:
        exp_name += f"_pca{data_conf['pca_components']}"

    lc = LoggingConfig(
        project_name=log_conf.get("project_name", "financial-models"),
        experiment_name=exp_name,
        wandb_enabled=log_conf.get("wandb_enabled", True),
        log_confusion_matrix=log_conf.get("log_confusion_matrix", True),
        log_trade_accuracy_vs_threshold=log_conf.get("log_trade_accuracy_vs_threshold", True),
        lambda_value=float(label_conf.get("threshold", 5e-4)),  # using threshold as lambda usually
        theta_values=parse_list_arg(log_conf.get("theta_values", "0.0,1e-4,4e-4"), type=float),
    )

    # Compile ModelConfig
    final_config = ModelConfig(
        architecture=arch_config,
        data=dc,
        training=tc,
        logging=lc,
        other=config,  # Save full flat config as other info
    )

    logger.info(f"Starting training for experiment: {exp_name}")

    trainer, results = train_model(features=features, labels=labels, config=final_config)

    return results
