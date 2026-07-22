"""
Feature builders for PyTorch Geometric graph construction.

Each builder function is a modular, stateless graph transform that extracts attributes
from raw scenario data (`data["_raw"]`) and constructs node, link, or target tensors in
the heterogeneous PyG graph representation (`HeteroData`).

Registry Architecture:
- Builder functions decorate themselves with `@register_builder("name")`.
- `BuilderTransform` wraps builder functions into standard PyG `BaseTransform` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch_geometric.transforms import BaseTransform

from ml_static.utils import validate_edge_attribute, validate_node_attribute

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import BuilderTransformConfig
    from ml_static.data import STADataset


# === Builder Registry ===
BUILDERS_REGISTRY: dict[str, Callable[[HeteroData], HeteroData]] = {}


def register_builder(name: str):
    """
    Decorator to register a graph feature builder function in `BUILDERS_REGISTRY`.

    Args:
        name: Unique string key used to reference the builder in YAML configs.

    Returns:
        Decorator function wrapping the builder implementation.
    """

    def decorator(func):
        if name in BUILDERS_REGISTRY:
            raise ValueError(f"Builder '{name}' is already registered.")
        BUILDERS_REGISTRY[name] = func
        return func

    return decorator


# =============================================================================
# Node Feature Builders
# =============================================================================

@register_builder("nodes_add_demand")
def nodes_add_demand(data: HeteroData) -> HeteroData:
    """
    Extracts the full origin-destination (OD) demand matrix from `_raw` data
    and stores it in `data["nodes"].demand`.
    """
    data["nodes"].demand = data["_raw"].demand
    return data


@register_builder("nodes_add_demand_as_x")
def nodes_add_demand_as_x(data: HeteroData) -> HeteroData:
    """
    Sets `data["nodes"].x = data["nodes"].demand` for transductive baseline models
    where node features consist of raw or scaled demand matrix rows.
    """
    validate_node_attribute(data, "nodes", "demand")
    data["nodes"].x = data["nodes"].demand
    return data


@register_builder("nodes_add_padding")
def nodes_add_padding(data: HeteroData) -> HeteroData:
    """
    Pads `data["nodes"].x` from current node count N to `max_nodes` (N_max).

    Why this is used:
    - Required for transductive baseline models (zero-padded node feature matrices of shape [N_max, N_max])
      to homogenize batch dimensions across different networks.
    - Also creates a `node_mask` boolean vector [N_max] marking valid physical nodes vs zero-padded dummy nodes.
    """
    validate_node_attribute(data, "nodes", "x")
    validate_node_attribute(data, "nodes", "max_nodes")

    max_nodes = data["nodes"].max_nodes
    n_curr = data["nodes"].num_nodes
    x_curr = data["nodes"].x

    if max_nodes < n_curr:
        raise ValueError(f"max_nodes ({max_nodes}) must be >= current num nodes ({n_curr})")

    # Construct zero-padded feature matrix of shape [N_max, N_max]
    feat_dim = max_nodes
    x_padded = torch.zeros((max_nodes, feat_dim), device=x_curr.device, dtype=x_curr.dtype)
    x_padded[:n_curr, :n_curr] = x_curr

    data["nodes"].x = x_padded

    # Create node_mask vector: True for physical nodes, False for dummy padded nodes
    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    node_mask[:n_curr] = True
    data["nodes"].node_mask = node_mask

    return data


@register_builder("nodes_add_net_demand")
def nodes_add_net_demand_raw(data: HeteroData) -> HeteroData:
    """
    Calculates net demand for each node v: net_demand_v = inflow_v - outflow_v.

    Why this is used:
    - Essential for calculating the physics-based node flow conservation loss L_c.
    - Net demand represents expected net vehicle flow accumulation at each physical node.
    """
    demand = data["_raw"].demand

    # Inflow is sum across origins (dim 0), Outflow is sum across destinations (dim 1)
    inflow = torch.sum(demand, dim=0)
    outflow = torch.sum(demand, dim=1)

    # Net node demand vector [N]
    data["nodes"].net_demand = inflow - outflow
    return data


# =============================================================================
# Physical (Real) Link Feature Builders
# =============================================================================

@register_builder("real_edges_add_index")
def real_edges_add_index(data: HeteroData) -> HeteroData:
    """
    Assigns physical link connectivity graph indices [2, E_real] to data[("nodes", "real", "nodes")].edge_index.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_index = data["_raw"].real_index
    del data["_raw"]["real_index"]
    return data


@register_builder("real_edges_add_capacity")
def real_edges_add_capacity(data: HeteroData) -> HeteroData:
    """
    Extracts physical link capacities (PCE/hour) and stores them in data[("nodes", "real", "nodes")].edge_capacity.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_capacity = data["_raw"].edge_capacity
    return data


@register_builder("real_edges_add_free_flow_time")
def real_edges_add_free_flow_time(data: HeteroData) -> HeteroData:
    """
    Extracts uncongested free-flow travel times and stores them in data[("nodes", "real", "nodes")].edge_free_flow_time.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_free_flow_time = data["_raw"].edge_free_flow_time
    return data


@register_builder("real_edges_add_vcr")
def real_edges_add_vcr(data: HeteroData) -> HeteroData:
    """
    Extracts ground-truth Volume-to-Capacity Ratios (V/C) for physical links.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_vcr = data["_raw"].edge_vcr
    return data


@register_builder("real_edges_add_flow")
def real_edges_add_flow(data: HeteroData) -> HeteroData:
    """
    Extracts ground-truth link flow volumes (PCE/hour) for physical links.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_flow = data["_raw"].edge_flow
    return data


@register_builder("real_edges_stack_capacity_free_flow_raw")
def real_edges_stack_capacity_free_flow(data: HeteroData) -> HeteroData:
    """
    Stacks unscaled capacity and free-flow travel time into edge_features [E_real, 2].
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_features = torch.stack(
        [data["_raw"].edge_capacity, data["_raw"].edge_free_flow_time], dim=1
    )
    return data


@register_builder("real_edges_stack_capacity_free_flow_scaled")
def real_edges_stack_capacity_free_flow_scaled(data: HeteroData) -> HeteroData:
    """
    Stacks min-max scaled capacity and free-flow travel time into edge_features [E_real, 2].
    This 2-dimensional feature tensor serves as link input for the EdgePredictor module.
    """
    edge_type = ("nodes", "real", "nodes")
    data[edge_type].edge_features = torch.stack(
        [data[edge_type].edge_capacity, data[edge_type].edge_free_flow_time], dim=1
    )
    return data


# =============================================================================
# Target Variable Builders
# =============================================================================

@register_builder("target_flow")
def target_flow(data: HeteroData) -> HeteroData:
    """
    Sets prediction target data.y to link flows (PCE/hour).
    """
    data.target_var = "flow"
    data.y = data["_raw"].edge_flow
    return data


@register_builder("target_vcr")
def target_vcr(data: HeteroData) -> HeteroData:
    """
    Sets prediction target data.y to Volume-to-Capacity Ratio (V/C).
    This is the default target specification for GUIDED models.
    """
    data.target_var = "vcr"
    data.y = data["_raw"].edge_vcr
    return data


# =============================================================================
# Virtual (OD Pair) Link Builders
# =============================================================================

@register_builder("virtual_edges_add_index")
def virtual_edges_add_index(data: HeteroData) -> HeteroData:
    """
    Assigns virtual OD pair link indices [2, E_virtual] to data[("nodes", "virtual", "nodes")].edge_index.
    """
    edge_type = ("nodes", "virtual", "nodes")
    data[edge_type].edge_index = data["_raw"].virtual_index
    del data["_raw"]["virtual_index"]
    return data


@register_builder("virtual_edges_add_demand")
def virtual_edges_add_demand(data: HeteroData) -> HeteroData:
    """
    Extracts scalar travel demand d_q for each active OD pair (u, v)
    and assigns it to data[("nodes", "virtual", "nodes")].edge_demand [E_virtual].

    Why this is used:
    - Serves as the scalar input for GUIDED EdgeProcessor (Linear or RBF)
      to project demand into 32-dim virtual edge embeddings.
    """
    virtual_edge_type = ("nodes", "virtual", "nodes")

    validate_edge_attribute(data, virtual_edge_type, "edge_index", expected_ndim=2)
    validate_node_attribute(data, "_raw", "demand", expected_ndim=2)

    edge_index = data[virtual_edge_type].edge_index
    demand_matrix = data["_raw"].demand

    sources = edge_index[0]
    targets = edge_index[1]
    demand_values = demand_matrix[sources, targets]

    data[virtual_edge_type].edge_demand = demand_values
    return data


# =============================================================================
# Memory Cleanup Builders
# =============================================================================

@register_builder("clean_raw_data")
def clean_raw_data(data: HeteroData) -> HeteroData:
    """
    Deletes temporary `_raw` namespace from `HeteroData` after feature construction
    to optimize GPU memory usage.
    """
    if "_raw" in data.node_types:
        del data["_raw"]
    return data


@register_builder("nodes_clean_demand")
def nodes_clean_demand(data: HeteroData) -> HeteroData:
    """
    Deletes intermediate `data["nodes"].demand` tensor after feature construction.
    """
    validate_node_attribute(data, "nodes", "demand")
    del data["nodes"].demand
    return data


@register_builder("real_edges_clean_capacity")
def real_edges_clean_capacity(data: HeteroData) -> HeteroData:
    """
    Deletes intermediate unstacked `edge_capacity` tensor after stacking into `edge_features`.
    """
    edge_type = ("nodes", "real", "nodes")
    validate_edge_attribute(data, edge_type, "edge_capacity")
    del data[edge_type].edge_capacity
    return data


@register_builder("real_edges_clean_free_flow_time")
def real_edges_clean_free_flow_time(data: HeteroData) -> HeteroData:
    """
    Deletes intermediate unstacked `edge_free_flow_time` tensor after stacking into `edge_features`.
    """
    edge_type = ("nodes", "real", "nodes")
    validate_edge_attribute(data, edge_type, "edge_free_flow_time")
    del data[edge_type].edge_free_flow_time
    return data


def get_builder(name: str) -> Callable[[HeteroData], HeteroData]:
    """
    Retrieves a registered builder function by its string key.

    Args:
        name: Name of registered builder function.

    Returns:
        Callable builder function.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in BUILDERS_REGISTRY:
        available = ", ".join(f"'{k}'" for k in BUILDERS_REGISTRY.keys())
        raise KeyError(f"Unknown builder '{name}'. Available builders: {available}")
    return BUILDERS_REGISTRY[name]


class BuilderTransform(BaseTransform):
    """
    PyG Transform wrapper encapsulating a builder function.
    """

    def __init__(self, builder_name: str):
        """
        Args:
            builder_name: Registered name of builder function.
        """
        self.builder_name = builder_name
        self.builder = get_builder(builder_name)

    @classmethod
    def from_config(cls, config: BuilderTransformConfig) -> BuilderTransform:
        """Instantiates BuilderTransform from configuration dataclass."""
        return cls(builder_name=config.builder)

    def forward(self, data: HeteroData) -> HeteroData:
        """Applies builder function to graph data object."""
        return self.builder(data)

    def fit_dataset(self, dataset: STADataset) -> None:
        """No-op (builders are stateless)."""
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(builder_name="{self.builder_name}")'
