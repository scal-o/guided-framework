"""
Edge embedding processors.

Transforms scalar edge attributes into learned embeddings via:
- Linear MLP expansion
- Gaussian RBF expansion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ml_static.models.components.mlp import MLP, LinearLayerConfig, MLPConfig
from ml_static.utils.validation import validate_edge_attribute

# =============================================================================
# Base Class
# =============================================================================


class EdgeEmbeddingProcessor(nn.Module):
    """Base class for processing scalar edge attributes into embeddings.

    Follows the component pattern:
    - Extracts edge_attr from data[edge_type]
    - Processes it into embeddings
    - Writes embeddings to data[edge_type].edge_embedding
    - Returns updated data
    """

    def _process_edge_attr(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Process edge attributes into embeddings.

        This method should be implemented by subclasses.

        Args:
            edge_attr: Scalar edge attributes [num_edges] or [num_edges, 1].

        Returns:
            Edge embeddings [num_edges, output_dim].
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor | None,
        data: HeteroData,
        edge_type: tuple | str = ("nodes", "virtual", "nodes"),
    ) -> HeteroData:
        """Transform scalar edge attributes to embeddings and update data.

        Args:
            x: MUST be None. Edge attributes are extracted from `data`.
            data: Heterogeneous graph data containing edge attributes.
            edge_type: Edge type identifier in the HeteroData object.

        Returns:
            Updated HeteroData with edge_embedding attribute.
        """
        if x is not None:
            raise ValueError("Input `x` must be None. Edge attributes are extracted from `data`.")

        # validate and extract edge_attr
        validate_edge_attribute(data, edge_type, "edge_demand", expected_ndim=None)
        edge_demand = data[edge_type].edge_demand

        # process edge attributes into embeddings
        edge_embedding = self._process_edge_attr(edge_demand)

        # write embeddings back to data
        data[edge_type].edge_embedding = edge_embedding

        return data


# =============================================================================
# LinearEdgeProcessor Configuration
# =============================================================================


@dataclass(frozen=True)
class LinearEdgeProcessorConfig:
    """Configuration for Linear edge embedding processor.

    Attributes:
        hidden_dim: Intermediate MLP dimension.
        output_dim: Final embedding dimension.
    """

    hidden_dim: int
    output_dim: int

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse LinearEdgeProcessorConfig from dictionary.

        Args:
            data: Dict with keys: hidden_dim, output_dim.

        Returns:
            LinearEdgeProcessorConfig instance.

        Raises:
            KeyError: If required keys are missing.
        """
        cls._validate_dict(data)
        return cls(
            hidden_dim=data["hidden_dim"],
            output_dim=data["output_dim"],
        )

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """Validate required keys."""
        required = {"hidden_dim", "output_dim"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in LinearEdgeProcessor config: {missing}")

    def validate(self) -> None:
        """Validate configuration values."""
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {self.output_dim}")


# =============================================================================
# LinearEdgeProcessor Implementation
# =============================================================================


class LinearEdgeProcessor(EdgeEmbeddingProcessor):
    """
    MLP-based edge embedding processor.

    Transforms scalar edge attributes into embeddings via a 2-layer MLP:
        Linear(1 → hidden_dim) + LeakyReLU
        Linear(hidden_dim → output_dim)
    """

    def __init__(self, mlp: MLP, output_dim: int):
        """Initialize LinearEdgeProcessor.

        Args:
            mlp: MLP component for transforming scalar to embeddings.
            output_dim: Final edge embedding dimension.
        """
        super().__init__()
        self.output_dim = output_dim
        self.mlp = mlp

    @classmethod
    def from_config(cls, config: LinearEdgeProcessorConfig) -> Self:
        """Build LinearEdgeProcessor from config.

        Args:
            config: LinearEdgeProcessorConfig instance.

        Returns:
            Configured LinearEdgeProcessor.
        """
        config.validate()

        # construct MLP component
        mlp_config = MLPConfig(
            input_channels=1,
            output_channels=config.output_dim,
            activation=None,
            layers=(LinearLayerConfig(hidden_channels=config.hidden_dim),),
        )
        mlp = MLP.from_config(mlp_config)

        return cls(mlp, config.output_dim)

    def _process_edge_attr(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Transform scalar edge attributes to embeddings.

        Args:
            edge_attr: Scalar edge attributes [num_edges] or [num_edges, 1].

        Returns:
            Edge embeddings [num_edges, output_dim].
        """
        # ensure edge_attr is [num_edges, 1]
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        # transform via MLP
        edge_emb = self.mlp(x=edge_attr)  # [num_edges, output_dim]

        return edge_emb


# =============================================================================
# RbfEdgeProcessor Configuration
# =============================================================================


@dataclass(frozen=True)
class RbfEdgeProcessorConfig:
    """Configuration for RBF-based edge embedding processor.

    Attributes:
        num_centers: Number of Gaussian basis functions (K).
        demand_min: Minimum value for center placement.
        demand_max: Maximum value for center placement.
        sigma: Gaussian width parameter (can be learnable).
        mlp: MLP configuration for mixing RBF features.
        learnable_sigma: Whether sigma should be a learnable parameter.
    """

    num_centers: int
    demand_min: float
    demand_max: float
    sigma: float
    mlp: MLPConfig
    learnable_sigma: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Parse RbfEdgeProcessorConfig from dictionary.

        Args:
            data: Dict with keys: num_centers, demand_min, demand_max,
                  sigma, mlp, learnable_sigma (optional).

        Returns:
            RbfEdgeProcessorConfig instance.

        Raises:
            KeyError: If required keys are missing.
        """
        cls._validate_dict(data)

        # parse MLP config
        mlp_config = MLPConfig.from_dict(data["mlp"])

        return cls(
            num_centers=data["num_centers"],
            demand_min=data["demand_min"],
            demand_max=data["demand_max"],
            sigma=data["sigma"],
            mlp=mlp_config,
            learnable_sigma=data.get("learnable_sigma", False),
        )

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """Validate required keys."""
        required = {"num_centers", "demand_min", "demand_max", "sigma", "mlp"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in RbfEdgeProcessor config: {missing}")

    def validate(self) -> None:
        """Validate configuration values."""
        if self.num_centers < 1:
            raise ValueError(f"num_centers must be >= 1, got {self.num_centers}")
        if self.demand_min >= self.demand_max:
            raise ValueError(
                f"demand_min ({self.demand_min}) must be < demand_max ({self.demand_max})"
            )
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

        # validate MLP config
        self.mlp.validate()


# =============================================================================
# RbfEdgeProcessor Implementation
# =============================================================================


class RbfEdgeProcessor(EdgeEmbeddingProcessor):
    """
    RBF-based edge embedding processor.

    Transforms scalar edge attributes via Gaussian Basis Expansion:
    1. Gaussian Basis Expansion: scalar → RBF features [num_centers]
    2. MLP mixing: RBF features → embeddings [output_dim]

    RBF Formula:
        phi_k(x) = exp(-(x - mu_k)^2 / sigma^2)

    where mu_k are uniformly spaced centers in [demand_min, demand_max].
    """

    def __init__(
        self,
        mlp: MLP,
        centers: torch.Tensor,
        log_sigma: torch.Tensor | nn.Parameter,
    ):
        """Initialize RbfEdgeProcessor.

        Args:
            mlp: MLP component for mixing RBF features.
            centers: Gaussian basis function centers [num_centers].
            log_sigma: Log of Gaussian width parameter (tensor or parameter).
        """
        super().__init__()
        self.mlp = mlp
        # extract output_dim from MLP's final layer
        self.output_dim = mlp.layers[-1].out_features

        # register centers as buffer (not learnable)
        self.register_buffer("centers", centers)

        # sigma can be learnable or fixed
        if isinstance(log_sigma, nn.Parameter):
            self.log_sigma = log_sigma
        else:
            self.register_buffer("log_sigma", log_sigma)

    @classmethod
    def from_config(cls, config: RbfEdgeProcessorConfig) -> Self:
        """Build RbfEdgeProcessor from config.

        Args:
            config: RbfEdgeProcessorConfig instance.

        Returns:
            Configured RbfEdgeProcessor.
        """
        config.validate()

        # validate that MLP input matches num_centers
        if config.mlp.input_channels != config.num_centers:
            raise ValueError(
                f"MLP input_channels ({config.mlp.input_channels}) must match "
                f"num_centers ({config.num_centers})"
            )

        # construct MLP component from config
        mlp = MLP.from_config(config.mlp)

        # construct centers tensor
        centers = torch.linspace(config.demand_min, config.demand_max, config.num_centers)

        # construct log_sigma (parameter or tensor)
        log_sigma_value = torch.tensor(config.sigma).log()
        if config.learnable_sigma:
            log_sigma = nn.Parameter(log_sigma_value)
        else:
            log_sigma = log_sigma_value

        return cls(mlp, centers, log_sigma)

    def _gaussian_expansion(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian basis expansion to scalar input.

        Args:
            x: Scalar values [num_edges, 1].

        Returns:
            RBF features [num_edges, num_centers].
        """
        # x: [num_edges, 1], centers: [num_centers]
        # compute (x - mu_k)^2 for all k
        diff = x - self.centers.view(1, -1)  # [num_edges, num_centers]
        sigma = self.log_sigma.exp()
        rbf_features = torch.exp(-diff.pow(2) / (sigma**2))
        return rbf_features

    def _process_edge_attr(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Transform scalar edge attributes to embeddings via RBF.

        Args:
            edge_attr: Scalar edge attributes [num_edges] or [num_edges, 1].

        Returns:
            Edge embeddings [num_edges, output_dim].
        """
        # ensure edge_attr is [num_edges, 1]
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        # Gaussian expansion
        rbf_features = self._gaussian_expansion(edge_attr)  # [num_edges, num_centers]

        # mix RBF features → edge embeddings
        edge_emb = self.mlp(x=rbf_features)  # [num_edges, output_dim]

        return edge_emb
