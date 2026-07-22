from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch
from torch_geometric.data import HeteroData

from ml_static.models.base import BaseConfig
from ml_static.models.components.mlp import MLP, MLPConfig
from ml_static.utils import validate_edge_attribute


@dataclass(frozen=True)
class PredictorConfig(MLPConfig, BaseConfig):
    """Configuration for edge predictor.

    Attributes:
        input_channels: Input feature dimension.
        output_channels: Final output dimension.
        layers: Tuple of hidden layer configurations.
        activation: Optional last layer activation function.
    """

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse PredictorConfig from dictionary."""
        cls._validate_dict(data)

        # Work on a copy to avoid side effects
        data = data.copy()

        # Calculate input channels dynamically from component dims
        input_channels = data.pop("node_feature_dim") * 2 + data.pop("edge_feature_dim")
        data["input_channels"] = input_channels

        # Delegate to MLPConfig's parsing logic
        # This works because PredictorConfig inherits all fields from MLPConfig
        # and MLPConfig.from_dict uses 'cls' to instantiate.
        return super().from_dict(data)

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """Validate that required keys are present in dict."""

        # node_feature_dim and edge_feature_dim can be absent if input_channels is specified
        if "input_channels" in data:
            if "node_feature_dim" in data or "edge_feature_dim" in data:
                raise KeyError(
                    "Error in predictor config: "
                    "either input_channels or [node_, edge_]feature_dim must be specified."
                )

            required = {"output_channels"}
        else:
            required = {"output_channels", "node_feature_dim", "edge_feature_dim"}

        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in predictor config: {missing}")


# =============================================================================
# Edge Predictor Implementation
# =============================================================================


class EdgePredictor(MLP):
    """
    MLP-based edge predictor that combines origin and destination node features
    with edge features to predict a target value for each edge.

    Supports configurable multi-layer architecture.
    """

    def forward(
        self,
        x: torch.Tensor | None,
        data: HeteroData | None = None,
        type: tuple | str = ("nodes", "real", "nodes"),
    ) -> torch.Tensor:
        """
        Forward pass to predict edge values.

        Args:
            x: Node features tensor of shape [num_nodes, input_dim].
            data: Heterogeneous graph data containing edge information.
            type: Edge type identifier in the HeteroData object.

        Returns:
            Predicted edge values tensor of shape [num_edges].
        """

        if x is None:
            raise ValueError("Missing x argument.")
        if data is None:
            raise ValueError("Missing data argument.")

        # extract edge index and features
        validate_edge_attribute(data, type, "edge_index", expected_ndim=2)
        edge_index = data[type].edge_index
        validate_edge_attribute(data, type, "edge_features", expected_ndim=2)
        edge_features = data[type].edge_features
        origin, destination = edge_index

        # Concatenate origin and destination node features with edge features
        z = torch.cat([x[origin], x[destination], edge_features], dim=1)

        return self._tensor_forward(z).view(-1)
