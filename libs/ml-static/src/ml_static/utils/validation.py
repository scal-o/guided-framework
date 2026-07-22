"""Validation utilities for heterogeneous graph data."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData


def validate_edge_attribute(
    data: HeteroData,
    edge_type: tuple | str,
    attribute: str,
    expected_ndim: int | None = None,
) -> None:
    """Validate that an edge type has the required attribute.

    Args:
        data: Heterogeneous graph data.
        edge_type: Edge type to validate (e.g., ("nodes", "real", "nodes")).
        attribute: Attribute name to check (e.g., 'edge_features', 'edge_index').
        expected_ndim: Optional expected number of dimensions for the tensor.
            If provided, validates that the attribute tensor has this many dimensions.

    Raises:
        ValueError: If edge type doesn't exist, attribute is missing,
            or dimension validation fails.

    Example:
        >>> validate_edge_attribute(data, ("nodes", "real", "nodes"), "edge_features")
        >>> validate_edge_attribute(data, ("nodes", "real", "nodes"), "edge_index", expected_ndim=2)
    """
    if edge_type not in data.edge_types:
        raise ValueError(f"Edge type {edge_type} not found in data")

    if attribute not in data[edge_type]:
        raise ValueError(f"Edge type {edge_type} does not have '{attribute}'")

    # optional dimension validation
    if expected_ndim is not None:
        tensor = data[edge_type][attribute]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Edge type {edge_type}.{attribute} is not a tensor, got {type(tensor)}"
            )

        actual_ndim = tensor.dim()
        if actual_ndim != expected_ndim:
            raise ValueError(
                f"Edge type {edge_type}.{attribute} has {actual_ndim} dimensions, "
                f"expected {expected_ndim}"
            )


def validate_node_attribute(
    data: HeteroData,
    node_type: str,
    attribute: str,
    expected_ndim: int | None = None,
) -> None:
    """Validate that a node type has the required attribute.

    Args:
        data: Heterogeneous graph data.
        node_type: Node type to validate (e.g., "nodes").
        attribute: Attribute name to check (e.g., 'x', 'features').
        expected_ndim: Optional expected number of dimensions for the tensor.
            If provided, validates that the attribute tensor has this many dimensions.

    Raises:
        ValueError: If node type doesn't exist, attribute is missing,
            or dimension validation fails.

    Example:
        >>> validate_node_attribute(data, "nodes", "x")
        >>> validate_node_attribute(data, "nodes", "x", expected_ndim=2)
    """
    if node_type not in data.node_types:
        raise ValueError(f"Node type '{node_type}' not found in data")

    if attribute not in data[node_type]:
        raise ValueError(f"Node type '{node_type}' does not have '{attribute}'")

    # optional dimension validation
    if expected_ndim is not None:
        tensor = data[node_type][attribute]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Node type {node_type}.{attribute} is not a tensor, got {type(tensor)}"
            )

        actual_ndim = tensor.dim()
        if actual_ndim != expected_ndim:
            raise ValueError(
                f"Node type {node_type}.{attribute} has {actual_ndim} dimensions, "
                f"expected {expected_ndim}"
            )
