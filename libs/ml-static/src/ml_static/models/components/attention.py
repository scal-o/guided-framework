from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Self

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax

from ml_static.models.components.mlp import MLP
from ml_static.utils.validation import validate_edge_attribute

# =============================================================================
# Configuration for Attention Layers
# =============================================================================


@dataclass(frozen=True)
class AttentionLayerConfig:
    """Configuration for an attention layer.

    Attributes:
        type: Layer class name (e.g., 'RealDependentAttentionLayer').
        hidden_channels: Hidden dimension for this layer.
        num_heads: Number of attention heads.
        edge_embedding_dim: Optional edge embedding dimension (for VirtualDependentAttentionLayer).
    """

    type: str
    hidden_channels: int
    num_heads: int
    edge_embedding_dim: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse AttentionLayerConfig from dictionary.

        Args:
            data: Dict with keys: type, hidden_channels, num_heads, edge_embedding_dim (optional).

        Returns:
            AttentionLayerConfig instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If values are invalid.
        """
        cls._validate_dict(data)

        return cls(
            type=data["type"],
            hidden_channels=data["hidden_channels"],
            num_heads=data["num_heads"],
            edge_embedding_dim=data.get("edge_embedding_dim"),
        )

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """Validate that required keys are present in dict.

        Args:
            data: Dictionary to validate.

        Raises:
            KeyError: If required keys are missing.
        """
        required = {"type", "hidden_channels", "num_heads"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in attention layer config: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        return asdict(self)

    def validate(self) -> None:
        """Validate layer configuration."""
        if self.hidden_channels % self.num_heads != 0:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must be divisible "
                f"by num_heads ({self.num_heads})"
            )


# =============================================================================
# Attention Layer Implementations
# =============================================================================


class BaseDependentAttentionLayer(nn.Module):
    """
    Base node-feature focused AttentionLayer class.
    The attention mechanism uses key, query, and value projections computed
    on the graph node features.
    These projections are combined with edge-dependent weights to compute
    attention scores over sparse graph edges.

    The compute_edge_weights method is left unimplemented and should be
    overridden by subclasses.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            msg = "hidden_dim must be divisible by num_heads"
            raise ValueError(msg)

        self.layer_norm = nn.LayerNorm(input_dim)

        self.hidden_dim: int = hidden_dim
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads

        self.lin_q = nn.Linear(input_dim, hidden_dim)
        self.lin_k = nn.Linear(input_dim, hidden_dim)
        self.lin_v = nn.Linear(input_dim, hidden_dim)

        self.scale = self.head_dim**-0.5

        # linear module
        layer1 = nn.Linear(hidden_dim, hidden_dim)
        layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = MLP([layer1, layer2], None)

    def compute_edge_weights(
        self, x: torch.Tensor, data: HeteroData, edge_type: tuple | str, **kwargs
    ) -> torch.Tensor:
        """Return attention modifiers for each edge and head.

        Args:
            x: Node features with shape ``[num_nodes, input_dim]``.
            edge_type: Edge type as a tuple or string.
            data: HeteroData object containing the graph data.
            **kwargs: Additional tensors required by concrete subclasses.

        Returns:
            Tensor of shape ``[num_edges, num_heads]`` containing edge weights.
        """

        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, data: HeteroData, edge_type: tuple | str, **kwargs
    ) -> torch.Tensor:
        """Apply dependent attention to sparse edges.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_type: Edge type as a tuple or string.
            data: HeteroData object containing the graph data.
            **kwargs: Extra tensors consumed by subclasses.

        Returns:
            Updated node features ``[num_nodes, hidden_dim]``.
        """

        num_nodes = x.size(0)
        validate_edge_attribute(data, edge_type, "edge_index", expected_ndim=2)
        edge_index = data[edge_type].edge_index
        origin, destination = edge_index

        # pre layer norm
        x_norm = self.layer_norm(x)

        q = self.lin_q(x_norm).view(num_nodes, self.num_heads, self.head_dim)
        k = self.lin_k(x_norm).view(num_nodes, self.num_heads, self.head_dim)
        v = self.lin_v(x_norm).view(num_nodes, self.num_heads, self.head_dim)

        q_edges = q[origin]
        k_edges = k[destination]
        scores = (q_edges * k_edges).sum(dim=-1) * self.scale

        edge_weights = self.compute_edge_weights(x_norm, data, edge_type, **kwargs)
        weighted_scores = scores * edge_weights

        attention_weights = softmax(weighted_scores, origin, num_nodes=num_nodes)

        v_edges = v[destination]
        weighted_values = attention_weights.unsqueeze(-1) * v_edges

        values = torch.zeros(
            num_nodes,
            self.num_heads,
            self.head_dim,
            device=x.device,
            dtype=weighted_values.dtype,
        )
        values.index_add_(0, origin, weighted_values)

        values = values.reshape(num_nodes, self.hidden_dim)
        values = self.lin_out(values)

        return x + values


## === adaptive weights submodules ===
class RealAdaptiveWeights(nn.Module):
    """Learns adaptive attention weights for real OD edges."""

    def __init__(self, edge_dim: int, num_heads: int) -> None:
        super().__init__()
        self.edge_dim: int = edge_dim
        self.num_heads: int = num_heads

        # linear module
        self.linear = nn.Linear(self.edge_dim, self.num_heads)

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on edge features.

        Args:
            edge_features: Edge attributes ``[num_edges, edge_dim]``.

        Returns:
            Adaptive weights ``[num_edges, num_heads]`` for each head.
        """

        return self.linear(edge_features)


class VirtualAdaptiveWeights(nn.Module):
    """Learns adaptive attention weights for virtual OD edges.

    Supports two modes:
    1. Node features only: concatenates origin and destination node features
    2. With edge embeddings: concatenates node features + edge embeddings
    """

    def __init__(
        self, input_dim: int, num_heads: int, edge_embedding_dim: int | None = None
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim * 2
        self.num_heads: int = num_heads
        self.edge_embedding_dim: int | None = edge_embedding_dim

        # determine total feature dimension
        if edge_embedding_dim is not None:
            total_dim = self.input_dim + edge_embedding_dim
        else:
            total_dim = self.input_dim

        # linear module
        layer1 = nn.Linear(total_dim, total_dim)
        layer2 = nn.Linear(total_dim, self.num_heads)
        self.lin_v_edges = MLP([layer1, layer2], None)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute adaptive weights based on origin/destination features.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Virtual edge connectivity ``[2, num_edges]``.
            edge_embedding: Optional edge embeddings ``[num_edges, edge_embedding_dim]``.

        Returns:
            Adaptive weights ``[num_edges, num_heads]`` for each head.
        """
        origin, destination = edge_index
        pair_features = torch.cat([x[origin], x[destination]], dim=1)

        # concatenate edge embeddings if available
        if edge_embedding is not None:
            pair_features = torch.cat([pair_features, edge_embedding], dim=1)

        return self.lin_v_edges(pair_features)


## === attention layer implementations ===
class RealDependentAttentionLayer(BaseDependentAttentionLayer):
    """Dependent attention variant that consumes real edge features."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__(input_dim, hidden_dim, num_heads)
        # TODO: make edge_dim configurable
        self.adaptive_layer = RealAdaptiveWeights(2, num_heads)

    def compute_edge_weights(
        self,
        x: torch.Tensor,
        data: HeteroData,
        edge_type: tuple | str = ("nodes", "real", "nodes"),
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate edge features and broadcast across heads.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            edge_index: Edge connectivity ``[2, num_edges]`` (unused).
            edge_features: Edge attributes ``[num_edges, edge_dim]``.

        Returns:
            Tensor with shape ``[num_edges, num_heads]`` containing weights.
        """
        validate_edge_attribute(data, edge_type, "edge_features", expected_ndim=2)
        edge_features = data[edge_type].edge_features

        return self.adaptive_layer(edge_features)


class VirtualDependentAttentionLayer(BaseDependentAttentionLayer):
    """Dependent attention variant that adapts weights per virtual edge.

    Optionally incorporates edge embeddings if available in the data.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_heads: int, edge_embedding_dim: int | None = None
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_heads)
        self.adaptive_layer = VirtualAdaptiveWeights(input_dim, num_heads, edge_embedding_dim)

    def compute_edge_weights(
        self,
        x: torch.Tensor,
        data: HeteroData,
        edge_type: tuple | str = ("nodes", "virtual", "nodes"),
        **kwargs,
    ) -> torch.Tensor:
        """Use learned adaptive weights for each virtual edge.

        Optionally uses edge embeddings if available in data.

        Args:
            x: Node features ``[num_nodes, input_dim]``.
            data: HeteroData object containing edge information.
            edge_type: Edge type identifier.
            **kwargs: Ignored keyword arguments.

        Returns:
            Adaptive weights ``[num_edges, num_heads]``.
        """
        validate_edge_attribute(data, edge_type, "edge_index", expected_ndim=2)
        edge_index = data[edge_type].edge_index

        # check if edge embeddings are available
        edge_embedding = None
        if hasattr(data[edge_type], "edge_embedding"):
            edge_embedding = data[edge_type].edge_embedding

        return self.adaptive_layer(x, edge_index, edge_embedding)
