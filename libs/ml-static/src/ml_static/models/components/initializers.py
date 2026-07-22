"""
Node initializers for HetGAT models.

These components encapsulate the logic for generating the initial node embeddings 'x'
before they are passed to the graph encoders. This allows the main HetGAT model
to be agnostic to whether the features come from raw data, preprocessing, or
complex OD-aggregation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ml_static.models.base import BaseConfig
from ml_static.models.components.edge_processors import (
    LinearEdgeProcessor,
    LinearEdgeProcessorConfig,
    RbfEdgeProcessor,
    RbfEdgeProcessorConfig,
)
from ml_static.models.components.od_initializers import ODNodeInitializer, ODNodeInitializerConfig
from ml_static.models.components.preprocessors import NodePreprocessor, PreprocessorConfig
from ml_static.utils.validation import validate_node_attribute

# =============================================================================
# Configurations
# =============================================================================


@dataclass(frozen=True)
class NodeInitializerConfig(BaseConfig):
    """
    Configuration for node feature initialization.

    Supports two paper strategies:
    - 'preprocessed': Direct MLP projection of node features (Baseline HetGAT).
    - 'from_demand': GUIDED virtual edge demand expansion and aggregation.
    """

    type: Literal["preprocessed", "from_demand"] = "from_demand"
    preprocessor: PreprocessorConfig | None = None
    edge_processor: LinearEdgeProcessorConfig | RbfEdgeProcessorConfig | None = None
    od_initializer: ODNodeInitializerConfig | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Factory method to create node initializer config."""
        init_type = data.get("type", "from_demand")

        preprocessor = None
        if "preprocessor" in data:
            preprocessor = PreprocessorConfig.from_dict(data["preprocessor"])

        edge_processor = None
        if "edge_processor" in data:
            ep_data = data["edge_processor"]
            ep_type = ep_data.get("type", "linear")
            if ep_type == "linear":
                edge_processor = LinearEdgeProcessorConfig.from_dict(ep_data)
            elif ep_type == "rbf":
                edge_processor = RbfEdgeProcessorConfig.from_dict(ep_data)
            else:
                raise ValueError(f"Unknown edge processor type: {ep_type}")

        od_initializer = None
        if "od_initializer" in data:
            od_initializer = ODNodeInitializerConfig.from_dict(data["od_initializer"])

        return cls(
            type=init_type,
            preprocessor=preprocessor,
            edge_processor=edge_processor,
            od_initializer=od_initializer,
        )

    def validate(self) -> None:
        if self.type == "preprocessed":
            if self.preprocessor is None:
                raise ValueError("Strategy 'preprocessed' requires 'preprocessor' config")
            if self.edge_processor is not None:
                raise ValueError("Strategy 'preprocessed' does not require 'edge_processor' config")
            elif self.od_initializer is not None:
                raise ValueError("Strategy 'preprocessed' does not require 'od_initializer' config")

            self.preprocessor.validate()

        elif self.type == "from_demand":
            if self.edge_processor is None:
                raise ValueError("Strategy 'from_demand' requires 'edge_processor' config")
            if self.od_initializer is None:
                raise ValueError("Strategy 'from_demand' requires 'od_initializer' config")

            if self.preprocessor is not None:
                self.preprocessor.validate()
            self.edge_processor.validate()
            self.od_initializer.validate()
        else:
            raise ValueError(f"Unsupported node initialization strategy: '{self.type}'")


# =============================================================================
# Modules
# =============================================================================


class NodeFeaturesInitializer(nn.Module):
    def __init__(
        self,
        preprocessor: nn.Module | None = None,
        edge_processor: nn.Module | None = None,
        od_initializer: ODNodeInitializer | None = None,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.edge_processor = edge_processor
        self.od_initializer = od_initializer

    def forward(self, graph: HeteroData) -> torch.Tensor:
        if self.edge_processor is not None:
            virtual_edge_type = ("nodes", "virtual", "nodes")
            graph = self.edge_processor(x=None, data=graph, edge_type=virtual_edge_type)
            if self.od_initializer is not None:
                graph = self.od_initializer(x=None, data=graph, edge_type=virtual_edge_type)

        if self.preprocessor is not None:
            g = self.preprocessor(x=None, data=graph, type="nodes")
        else:
            validate_node_attribute(graph, "nodes", "x", expected_ndim=2)
            g = graph["nodes"].x

        return g

    @classmethod
    def from_config(cls, config: NodeInitializerConfig) -> Self:
        config.validate()
        preprocessor, edge_processor, od_initializer = None, None, None

        if config.type == "preprocessed":
            if config.preprocessor is not None:
                preprocessor = NodePreprocessor.from_config(config.preprocessor)
        elif config.type == "from_demand":
            if isinstance(config.edge_processor, LinearEdgeProcessorConfig):
                edge_processor = LinearEdgeProcessor.from_config(config.edge_processor)
            elif isinstance(config.edge_processor, RbfEdgeProcessorConfig):
                edge_processor = RbfEdgeProcessor.from_config(config.edge_processor)
            else:
                raise ValueError(f"Unknown edge processor config: {type(config.edge_processor)}")

            if config.preprocessor is not None:
                preprocessor = NodePreprocessor.from_config(config.preprocessor)

            assert config.od_initializer is not None, (
                "od_initializer must be defined for from_demand strategy"
            )
            od_initializer = ODNodeInitializer.from_config(config.od_initializer)

        return cls(
            preprocessor=preprocessor, edge_processor=edge_processor, od_initializer=od_initializer
        )
