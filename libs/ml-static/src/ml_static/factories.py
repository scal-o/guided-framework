"""
Factory functions for creating PyTorch objects from configuration.

These functions keep domain object creation separate from configuration
data structures, improving testability and reducing coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from ml_static.config import OptimizerConfig, SchedulerConfig


def create_optimizer(config: OptimizerConfig, params: Iterator[nn.Parameter]) -> optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        config: Optimizer configuration.
        params: Model parameters to optimize.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer type is not supported.
    """
    if config.type == "adam":
        return optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
            fused=True,
        )
    elif config.type == "adamw":
        return optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    elif config.type == "sgd":
        return optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.type}")


def create_scheduler(
    config: SchedulerConfig,
    optimizer: optim.Optimizer,
    runtime_params: dict[str, Any] | None = None,
) -> LRScheduler:
    """
    Create learning rate scheduler from configuration.

    Args:
        config: Scheduler configuration.
        optimizer: Optimizer to schedule.
        runtime_params: Optional runtime parameters (e.g., 'epochs', 'steps_per_epoch').
            These override config values if provided.

    Returns:
        Configured scheduler instance.

    Raises:
        ValueError: If scheduler type is not supported or required parameters are missing.

    Example:
        >>> scheduler = create_scheduler(
        ...     config.scheduler,
        ...     optimizer,
        ...     runtime_params={'epochs': 100, 'steps_per_epoch': len(train_loader)}
        ... )
    """
    runtime_params = runtime_params or {}

    if config.type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience,
        )
    elif config.type == "one_cycle":
        # use runtime parameters if provided, otherwise use config values
        final_epochs = runtime_params.get("epochs", config.epochs)
        final_steps = runtime_params.get("steps_per_epoch", config.steps_per_epoch)

        if final_epochs is None or final_steps is None:
            raise ValueError(
                "OneCycleLR requires 'epochs' and 'steps_per_epoch' to be specified "
                "either in config or as runtime parameters"
            )
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=final_epochs,
            steps_per_epoch=final_steps,
            pct_start=config.pct_start,
            anneal_strategy=config.anneal_strategy,
        )
    elif config.type == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.type}")
