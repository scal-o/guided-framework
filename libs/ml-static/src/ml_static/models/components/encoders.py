from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ml_static.models.base import BaseConfig
from ml_static.models.components import (
    AttentionLayerConfig,
    RealDependentAttentionLayer,
    VirtualDependentAttentionLayer,
)

# =============================================================================
# Configuration for Encoders
# =============================================================================


@dataclass(frozen=True)
class EncoderConfig(BaseConfig):
    """Configuration for a graph encoder.

    Attributes:
        type: Encoder type ('virtual' or 'real').
        input_channels: Input feature dimension.
        layers: Tuple of layer configurations.
    """

    type: Literal["virtual", "real"]
    input_channels: int
    layers: tuple[AttentionLayerConfig, ...]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse EncoderConfig from dictionary."""
        cls._validate_dict(data)

        # Parse layers
        layers = [AttentionLayerConfig.from_dict(d) for d in data["layers"]]
        layers = tuple(layers)

        return cls(
            type=data["type"],
            input_channels=data["input_channels"],
            layers=layers,
        )

    def validate(self) -> None:
        """Validate encoder configuration."""
        if self.type not in {"virtual", "real"}:
            raise ValueError(f"Encoder 'type' must be 'virtual' or 'real', got '{self.type}'")
        if not self.layers:
            raise ValueError("Encoder must contain at least one layer")

        # Validate all layers
        for layer in self.layers:
            layer.validate()


# =============================================================================
# Encoder Implementation
# =============================================================================


class EncoderBase(nn.Module):
    """
    Base class for graph encoders.
    Accepts both real and virtual edge types.
    """

    SUPPORTED_LAYERS = {
        "RealDependentAttentionLayer": RealDependentAttentionLayer,
        "VirtualDependentAttentionLayer": VirtualDependentAttentionLayer,
    }

    SUPPORTED_STRING = ", ".join(SUPPORTED_LAYERS.keys())

    def __init__(
        self, encoder_layers: list[nn.Module], encoder_type: Literal["virtual", "real"]
    ) -> None:
        super().__init__()

        # set edge type based on encoder type
        edge_types = {
            "virtual": ("nodes", "virtual", "nodes"),
            "real": ("nodes", "real", "nodes"),
        }

        self.edge_type: tuple = edge_types[encoder_type]

        # create encoder layers list
        self.encoder_layers = nn.ModuleList(encoder_layers)

    @classmethod
    def from_config(cls, config: EncoderConfig) -> Self:
        """Instantiate encoder from typed config.

        Args:
            config: EncoderConfig instance.

        Returns:
            Configured EncoderBase instance.
        """
        config.validate()

        encoder_layers = []
        in_channels = config.input_channels

        for layer_config in config.layers:
            layer_type = layer_config.type

            if layer_type not in cls.SUPPORTED_LAYERS:
                raise ValueError(
                    f"Unsupported layer type: {layer_type}. Supported types: {cls.SUPPORTED_STRING}"
                )

            layer_class = cls.SUPPORTED_LAYERS[layer_type]

            # pass edge_embedding_dim if specified (only used by VirtualDependentAttentionLayer)
            if layer_config.edge_embedding_dim is not None:
                layer = layer_class(
                    in_channels,
                    layer_config.hidden_channels,
                    layer_config.num_heads,
                    layer_config.edge_embedding_dim,
                )
            else:
                layer = layer_class(
                    in_channels,
                    layer_config.hidden_channels,
                    layer_config.num_heads,
                )

            encoder_layers.append(layer)
            in_channels = layer_config.hidden_channels

        return cls(encoder_layers, config.type)

    def forward(
        self,
        x: torch.Tensor,
        data: HeteroData,
    ) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            data: Heterogeneous graph data containing edge information.

        Returns:
            Updated node features ``[num_nodes, hidden_dim]``.
        """

        for layer in self.encoder_layers:
            x = layer(x, data, self.edge_type)

        return x