from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_static.models.base import BaseConfig

# =============================================================================
# Configuration for Linear Layers
# =============================================================================


@dataclass(frozen=True)
class LinearLayerConfig(BaseConfig):
    """Configuration for a linear layer in MLP.

    Attributes:
        hidden_channels: Output dimension for this layer.
    """

    hidden_channels: int

    def validate(self) -> None:
        """Validate layer configuration."""
        if self.hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {self.hidden_channels}")


# =============================================================================
# Configuration for Edge Predictor
# =============================================================================


@dataclass(frozen=True)
class MLPConfig(BaseConfig):
    """Configuration for general MLP.

    Attributes:
        input_channels: Input feature dimension.
        output_channels: Final output dimension.
        activation: Final layer activation function.
        layers: Tuple of hidden layer configurations.
    """

    input_channels: int
    output_channels: int
    layers: tuple[LinearLayerConfig, ...]
    activation: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse MLPConfig from dictionary."""
        cls._validate_dict(data)

        # Parse layers
        layers_data = data.get("layers", [])
        layers = [LinearLayerConfig.from_dict(d) for d in layers_data]

        # Create a new dict for instantiation to avoid modifying input
        init_data = data.copy()
        init_data["layers"] = tuple(layers)

        return cls(**init_data)

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """
        Validate that required keys are present in dict.
        Overrides BaseConfig because 'layers' and 'activation' are optional/have defaults logic.
        """
        required = {"input_channels", "output_channels"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in MLP config: {missing}")

    def validate(self) -> None:
        """Validate predictor configuration."""
        if self.input_channels < 1:
            raise ValueError(f"input_channels must be >= 1, got {self.input_channels}")
        if self.output_channels < 1:
            raise ValueError(f"output_channels must be >= 1, got {self.output_channels}")


class MLP(nn.Module):
    """
    MLP.

    Supports configurable multi-layer architecture.
    """

    VALID_ACTIVATIONS = {
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "softplus": F.softplus,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        None: lambda x: x,
    }

    def __init__(self, layers: list[nn.Module], final_activation: str | None) -> None:
        """Initialize MLP with a list of linear layers.

        Args:
            layers: List of nn.Linear layers for the MLP stack.
            final_activation: Activation function name for the final layer.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

        if final_activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Invalid final activation '{final_activation}'. "
                f"Supported: {list(self.VALID_ACTIVATIONS.keys())}"
            )

        self.final_activation: Callable = self.VALID_ACTIVATIONS[final_activation]

    @classmethod
    def from_config(cls, config: MLPConfig) -> Self:
        """Build MLP from typed config.

        Args:
            config: MLPConfig instance.

        Returns:
            Configured MLP instance.
        """
        config.validate()

        # start with input channels
        input_dim = config.input_channels

        linear_layers = []

        for layer_config in config.layers:
            linear_layers.append(nn.Linear(input_dim, layer_config.hidden_channels))
            input_dim = layer_config.hidden_channels

        # Final output layer
        linear_layers.append(nn.Linear(input_dim, config.output_channels))

        final_activation = config.activation

        return cls(linear_layers, final_activation)

    def _tensor_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for generic tensor input."""

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all layers except the last
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x)

        x = self.final_activation(x)
        return x

    def forward(
        self,
        x: torch.Tensor | None,
        data: Any = None,
        type: Any = None,
    ) -> torch.Tensor:
        """
        Forward pass implementation for graphs.

        Subclasses should override this method.
        """
        if x is None:
            raise ValueError("Input `x` must be a tensor. This can be overridden in subclasses.")

        return self._tensor_forward(x)
