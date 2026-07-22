"""
Base classes for models and configurations.

This module provides:
1. BaseConfig: Reduces boilerplate for configuration dataclasses.
2. BaseGNNModel: Provides standard checkpointing and input validation for GNNs.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ml_static.config import Config
from ml_static.utils.validation import validate_node_attribute

# =============================================================================
# Base Configuration
# =============================================================================


@dataclass(frozen=True)
class BaseConfig:
    """
    Base class for configuration objects.
    Provides generic dictionary parsing and validation.
    """

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """
        Parse config from dictionary.
        Automatically validates keys against dataclass fields.
        """
        cls._validate_dict(data)
        # Note: Subclasses with nested configs should override this
        # if they need to parse sub-objects (which is common).
        # This default implementation works for flat configs.
        return cls(**data)

    @classmethod
    def _validate_dict(cls, data: dict) -> None:
        """
        Validate that required keys are present in dict.
        Infers required keys from the dataclass fields.
        """
        required = {f.name for f in fields(cls)}
        # Optional fields (with defaults) could be excluded here if we inspect defaults,
        # but typically we want explicit configuration in this project.
        # If strict validation is too rigid, subclasses can override.

        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in {cls.__name__}: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        return asdict(self)

    def validate(self) -> None:
        """
        Validate configuration constraints.
        Subclasses should override this to add specific logic.
        """
        pass


# =============================================================================
# Base Model
# =============================================================================


class BaseGNNModel(nn.Module):
    """
    Base class for GNN models.
    Provides standardized checkpointing and input validation.
    """

    # Subclasses should define this string
    _MODEL_TYPE: str = "BaseGNN"

    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Create model from Config object. Must be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """Forward pass. Must be implemented by subclasses."""
        raise NotImplementedError

    def extract_checkpoint(self) -> dict:
        """
        Extract model checkpoint for saving.

        Returns:
            Dictionary containing state_dict and model metadata.
        """
        return {
            "state_dict": copy.deepcopy(self.state_dict()),
            "model_type": self._MODEL_TYPE,
        }

    @classmethod
    def from_checkpoint(cls, config: Config, checkpoint_path: Path | str) -> Self:
        """
        Load model from checkpoint file.

        Args:
            config: Configuration object for model architecture.
            checkpoint_path: Path to checkpoint file (.pt).

        Returns:
            Loaded model instance.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        ckpt_type = checkpoint.get("model_type")

        # Verify model type
        if ckpt_type != cls._MODEL_TYPE:
            raise ValueError(
                f"Checkpoint model type '{ckpt_type}' is not compatible with {cls._MODEL_TYPE}"
            )

        # Create model from config
        model = cls.from_config(config)

        # Load state dict
        # Subclasses can override _load_state_dict to handle migration
        model._load_state_dict(checkpoint["state_dict"])

        return model

    def _load_state_dict(self, state_dict: dict) -> None:
        """Load state dict into model (hook for migration logic)."""
        self.load_state_dict(state_dict)

    def _validate_input(self, graph: HeteroData) -> None:
        """
        Standard validation for input graphs.
        Checks for basic attributes expected by most GNNs.
        """
        validate_node_attribute(graph, "nodes", "x", expected_ndim=2)
