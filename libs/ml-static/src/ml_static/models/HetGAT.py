from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ml_static.config import ConfigLoader
from ml_static.models import register_model
from ml_static.models.base import BaseConfig, BaseGNNModel
from ml_static.models.components import encoders, initializers, predictors

if TYPE_CHECKING:
    from ml_static.config import Config


# =============================================================================
# Configuration
# =============================================================================


@ConfigLoader.register_config_class("HetGAT")
@dataclass(frozen=True)
class HetGATArchitectureConfig(BaseConfig):
    """
    Unified configuration for HetGAT models.

    Attributes:
        encoders: Tuple of encoder configurations.
        predictor: Edge predictor configuration.
        node_initialization: Configuration for how node features are initialized.
                             Defaults to 'raw' if not specified.
    """

    encoders: tuple[encoders.EncoderConfig, ...]
    predictor: predictors.PredictorConfig
    node_initialization: initializers.NodeInitializerConfig

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse configuration from dictionary."""
        cls._validate_dict(data)

        # Parse encoders
        encoders_list = [encoders.EncoderConfig.from_dict(d) for d in data["encoders"]]
        encoders_list = tuple(encoders_list)

        # Parse predictor
        predictor = predictors.PredictorConfig.from_dict(data["predictor"])

        # Parse node initialization configuration
        init_data = data.get("node_initialization", {"type": "raw"})
        node_init = initializers.NodeInitializerConfig.from_dict(init_data)

        return cls(
            encoders=encoders_list,
            predictor=predictor,
            node_initialization=node_init,
        )

    def validate(self) -> None:
        """Validate configuration."""
        if len(self.encoders) < 1:
            raise ValueError("HetGAT requires at least one encoder")

        for encoder in self.encoders:
            encoder.validate()

        self.predictor.validate()
        self.node_initialization.validate()


# =============================================================================
# Unified Model Implementation
# =============================================================================


@register_model("HetGAT")
class HetGAT(BaseGNNModel):
    """
    Unified Heterogeneous Graph Attention Network (HetGAT) model.

    This class consolidates the logic of:
    - HetGAT_no_preproc (Raw initialization)
    - HetGAT_preproc (Preprocessed initialization)
    - HetGAT_OD_init (Demand-based initialization)

    The behavior is determined by the `node_initializer` component.
    """

    _MODEL_TYPE = "HetGAT"

    def __init__(
        self,
        node_initializer: nn.Module,
        encoders: list[nn.Module],
        predictor: nn.Module,
    ):
        super().__init__()
        self.node_initializer = node_initializer
        self.encoders = nn.ModuleList(encoders)
        self.predictor = predictor

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Create model from configuration."""
        arch = config.model.architecture

        if not isinstance(arch, HetGATArchitectureConfig):
            raise ValueError(f"Expected HetGATArchitectureConfig, got {type(arch).__name__}. ")

        arch.validate()

        # Build Node Initializer
        init_config = arch.node_initialization
        node_init = initializers.NodeFeaturesInitializer.from_config(init_config)

        # Build Encoders
        encoders_list = []
        for encoder_config in arch.encoders:
            encoder = encoders.EncoderBase.from_config(encoder_config)
            encoders_list.append(encoder)

        # Build Predictor
        predictor = predictors.EdgePredictor.from_config(arch.predictor)

        return cls(node_init, encoders_list, predictor)

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """Forward pass."""

        # 0. Initialize Node Features
        g = self.node_initializer(graph)

        # 1. Validate Input
        self._validate_input(graph)

        # 2. Apply Encoders
        for encoder in self.encoders:
            g = encoder(g, graph)

        # 3. Predict
        z = self.predictor(g, graph)

        return z
