"""
A declarative, factory-based approach for model creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from ml_static.config import Config


## ======================
## === model registry ===
MODEL_REGISTRY = {}


def register_model(name: str):
    """
    A decorator to register a new model class in the factory's registry.
    """

    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


## ==============================
## === model factory function ===
def model_factory(config: Config) -> nn.Module:
    """
    Creates a model instance using the factory pattern.

    Args:
        config: Configuration object containing model type and architecture.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If model type is not registered.
    """
    model_type = config.model.type

    if model_type not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. Available models: [{available_models}]"
        )

    model_class = MODEL_REGISTRY[model_type]

    # call the model's from_config method
    return model_class.from_config(config)
