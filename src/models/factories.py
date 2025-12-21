"""
Registry and factory system for model components.
"""

import torch.nn as nn
import torch.optim as optim

from .losses import FocalLoss, ExpectedReturnLoss
from .architectures.base import ModelArchitectureConfig
from .architectures.mlp import MLPConfig
from .architectures.lstm import LSTMConfig
from .architectures.transformer import TransformerConfig


# Optimizer registry
OPTIMIZER_REGISTRY = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


# Scheduler registry
SCHEDULER_REGISTRY = {
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "step": optim.lr_scheduler.StepLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
}


# Loss function registry
CRITERION_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "focal": FocalLoss,
    "expected_return": ExpectedReturnLoss,
}


# Architecture registry
ARCHITECTURE_REGISTRY = {
    "lstm": LSTMConfig,
    "transformer": TransformerConfig,
    "mlp": MLPConfig,
}


def create_optimizer(
    optimizer_name: str,
    model_parameters,
    learning_rate: float,
    weight_decay: float = 0.0,
    **kwargs,
) -> optim.Optimizer:
    """
    Create an optimizer from the registry.

    Args:
        optimizer_name: Name of the optimizer
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance
    """
    optimizer_name_lower = optimizer_name.lower()
    if optimizer_name_lower not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available optimizers: {list(OPTIMIZER_REGISTRY.keys())}"
        )

    optimizer_class = OPTIMIZER_REGISTRY[optimizer_name_lower]

    # Build optimizer kwargs
    optimizer_kwargs = {"lr": learning_rate, "weight_decay": weight_decay}

    # Add optimizer-specific parameters
    if optimizer_name_lower == "sgd":
        optimizer_kwargs["momentum"] = kwargs.get("momentum", 0.9)

    # Add any additional kwargs
    optimizer_kwargs.update(kwargs)

    return optimizer_class(model_parameters, **optimizer_kwargs)


def create_scheduler(
    scheduler_name: str | None, optimizer: optim.Optimizer, scheduler_params: dict, **kwargs
) -> optim.lr_scheduler._LRScheduler | None:
    """
    Create a learning rate scheduler from the registry.

    Args:
        scheduler_name: Name of the scheduler (None for no scheduler)
        optimizer: The optimizer to schedule
        scheduler_params: Scheduler-specific parameters
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Scheduler instance or None
    """
    if scheduler_name is None:
        return None

    scheduler_name_lower = scheduler_name.lower()
    if scheduler_name_lower not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Available schedulers: {list(SCHEDULER_REGISTRY.keys())}"
        )

    scheduler_class = SCHEDULER_REGISTRY[scheduler_name_lower]

    # Build scheduler kwargs based on type
    scheduler_kwargs = {**scheduler_params, **kwargs}

    # Handle scheduler-specific parameters
    if scheduler_name_lower == "cosine":
        # CosineAnnealingLR expects T_max
        if "T_max" not in scheduler_kwargs and "num_epochs" in kwargs:
            scheduler_kwargs["T_max"] = kwargs["num_epochs"]
            del scheduler_kwargs["num_epochs"]
    elif scheduler_name_lower == "step":
        # StepLR expects step_size and gamma
        scheduler_kwargs.setdefault("step_size", 30)
        scheduler_kwargs.setdefault("gamma", 0.1)
    elif scheduler_name_lower == "plateau":
        # ReduceLROnPlateau expects mode, patience, factor
        scheduler_kwargs.setdefault("mode", "min")
        scheduler_kwargs.setdefault("patience", 10)
        scheduler_kwargs.setdefault("factor", 0.5)

    return scheduler_class(optimizer, **scheduler_kwargs)


def create_criterion(loss_name: str, loss_params: dict = None, **kwargs) -> nn.Module:
    """
    Create a loss function from the registry.

    Args:
        loss_name: Name of the loss function
        loss_params: Loss-specific parameters
        **kwargs: Additional loss-specific arguments

    Returns:
        Loss function instance
    """
    loss_name_lower = loss_name.lower()
    if loss_name_lower not in CRITERION_REGISTRY:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available losses: {list(CRITERION_REGISTRY.keys())}"
        )

    criterion_class = CRITERION_REGISTRY[loss_name_lower]

    # Build loss kwargs
    loss_kwargs = {**(loss_params or {}), **kwargs}

    return criterion_class(**loss_kwargs)


def create_architecture_config(model_type: str, arch_dict: dict) -> ModelArchitectureConfig:
    """
    Create an architecture configuration from the registry.

    Args:
        model_type: Type of model architecture
        arch_dict: Dictionary containing architecture parameters

    Returns:
        Architecture configuration instance
    """
    model_type_lower = model_type.lower()
    if model_type_lower not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available architectures: {list(ARCHITECTURE_REGISTRY.keys())}"
        )

    config_class = ARCHITECTURE_REGISTRY[model_type_lower]
    return config_class(**arch_dict)
