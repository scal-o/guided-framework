from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData

from ml_static.models.base import BaseConfig
from ml_static.models.components.mlp import MLP, MLPConfig
from ml_static.utils import validate_node_attribute


@dataclass(frozen=True)
class PreprocessorConfig(MLPConfig, BaseConfig):
    """Configuration for node features preprocessor.

    Attributes:
        input_channels: Input feature dimension.
        output_channels: Final output dimension.
        layers: Tuple of hidden layer configurations.
        activation: Optional last layer activation function.
    """


# =============================================================================
# Node preprocessor Implementation
# =============================================================================


class NodePreprocessor(MLP):
    """
    MLP-based node preprocessor that transforms (encodes) input node features
    into a desired output dimension.

    Supports configurable multi-layer architecture.
    """

    def forward(
        self,
        x: torch.Tensor | None,
        data: HeteroData | None = None,
        type: str = "nodes",
    ) -> torch.Tensor:
        """
        Forward pass to encode node features.

        Args:
            x: MUST be None. Node features are extracted from `data`.
            data: Heterogeneous graph data containing node features.
            type: Node type identifier in the HeteroData object.

        Returns:
            Processed node features of shape [num_nodes, output_dim].
        """

        if x is not None:
            raise ValueError("Input `x` must be None. Node features are extracted from `data`.")
        if data is None:
            raise ValueError("Missing data argument.")

        # extract node features
        validate_node_attribute(data, type, "x", expected_ndim=2)
        x = data[type].x

        return self._tensor_forward(x)