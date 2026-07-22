from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from ml_static.transforms import SequentialTransform
from ml_static.utils import validate_edge_attribute, validate_node_attribute

if TYPE_CHECKING:
    from ml_static.config import Config

# Loss component names (order matters for learnable weights indexing)
LOSS_COMPONENT_NAMES = ("vcr", "flow", "conservation")

# Keys for fixed-weight physics-informed loss components
LOSS_WEIGHT_KEYS = ("w_vcr", "w_flow", "w_conservation")


class LossWrapper(nn.Module):
    """
    Wrapper for possible loss functions.
    """

    def __init__(self, loss_type: str, dataset_path: Path, **kwargs) -> None:
        super().__init__()

        VALID_LOSSES = {
            "l1": self._l1_loss,
            "l2": self._l2_loss,
            "mse": self._l2_loss,
            "rmsn": self._rmsn_loss,
            "custom": self._custom_loss,
        }

        INIT_FUNCTIONS = {
            "custom": self._init_custom_params,
        }

        MASK_FUNCTIONS = {
            "random": self._get_fixed_mask,
            "predefined": self._get_predefined_mask,
            "full": self._get_full_mask,
            None: self._get_full_mask,
        }

        if loss_type not in VALID_LOSSES:
            valid_list = ", ".join(f"'{t}'" for t in sorted(VALID_LOSSES))
            raise ValueError(f"Invalid loss type '{loss_type}'. Valid options are: {valid_list}.")

        self.loss_type: str = loss_type
        self.loss_fn: Callable = VALID_LOSSES[self.loss_type]

        if loss_type in INIT_FUNCTIONS:
            INIT_FUNCTIONS[loss_type](**kwargs)

        self.mask_type: str | None = kwargs.get("mask_type", None)
        self.mask_evaluation: str = kwargs.get("mask_evaluation", "full")
        self.mask_ratio: torch.Tensor = torch.as_tensor(kwargs.get("mask_ratio", 0.1))

        print(
            f"Initialized LossWrapper with loss_type='{self.loss_type}', mask_type='{self.mask_type}', mask_ratio={self.mask_ratio.item()}"
        )

        self.mask_fn: Callable = MASK_FUNCTIONS[self.mask_type]

        self.dataset_path: Path = dataset_path

        self._validated: bool = False

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """
        Create loss function from configuration object.
        """
        return cls(config.loss.type, config.dataset.full_path, **config.loss.kwargs)

    def forward(self, pred, data, evaluate=False) -> tuple[torch.Tensor, dict]:
        if evaluate:
            if self.mask_evaluation == "full":
                mask = self._get_full_mask(data)
            elif self.mask_evaluation == "unobserved":
                mask = self.mask_fn(data)
                mask = ~mask  # invert mask to evaluate on unobserved portion
            else:
                raise ValueError(
                    f"Invalid mask_evaluation mode '{self.mask_evaluation}'. "
                    "Valid options are 'full' or 'unobserved'."
                )
        else:
            mask = self.mask_fn(data)

        return self.loss_fn(pred, data, mask)

    def register_transform(self, transform: SequentialTransform) -> None:
        """
        Register a transform pipeline with the loss function.
        """
        self.transform: SequentialTransform = transform

    def set_weights(self, **weights: float) -> None:
        """Update fixed loss weights dynamically (e.g., for curriculum learning).

        Args:
            **weights: Weight values keyed by names like 'w_vcr', 'w_flow', 'w_conservation'.

        Raises:
            RuntimeError: If called for a loss type that does not use fixed weights.
            KeyError: If an unknown weight key is provided.
        """
        if self.loss_type not in ("custom", "learnable"):
            raise RuntimeError(
                f"set_weights() is only supported for physics-informed losses; got loss_type='{self.loss_type}'."
            )

        if not hasattr(self, "weight_vars"):
            raise RuntimeError(
                "Loss has no 'weight_vars' initialized; cannot set weights dynamically."
            )

        for key, value in weights.items():
            if key not in LOSS_WEIGHT_KEYS:
                valid = ", ".join(LOSS_WEIGHT_KEYS)
                raise KeyError(f"Unknown loss weight key '{key}'. Valid keys: {valid}.")
            self.weight_vars[key] = float(value)

    def _init_custom_params(self, **kwargs) -> None:
        """
        Initializes parameters for the fixed-weight custom loss.
        """
        defaults = {"w_vcr": 1.0, "w_flow": 0.003, "w_conservation": 0.003}
        self.weight_vars: dict[str, float] = {}
        for key, default_val in defaults.items():
            val = kwargs.get(key, None)
            if val is None:
                print(f"Warning: Using default weight for {key}: {default_val}")
                val = default_val
            self.weight_vars[key] = val

    def _validate_custom_data(self, data: HeteroData) -> None:
        """
        Validate that the data object contains the necessary attributes
        for computing physics-informed losses.
        """
        if self._validated:
            return

        if not hasattr(self, "transform"):
            raise RuntimeError(
                "Physics-informed losses require a transform. "
                "Call register_transform() before using this loss."
            )

        real_edge = ("nodes", "real", "nodes")
        validate_node_attribute(data, "nodes", "net_demand")
        validate_edge_attribute(data, real_edge, "edge_capacity")
        validate_edge_attribute(data, real_edge, "edge_flow")
        validate_edge_attribute(data, real_edge, "edge_index")

        if not hasattr(data, "target_var"):
            raise ValueError("data must have attribute 'target_var' for custom loss.")
        elif data.target_var != "vcr" and all(var != "vcr" for var in data.target_var):
            raise ValueError("Custom loss can only be used when target_var is 'vcr'.")

        self._validated = True

    def _get_full_mask(self, data: HeteroData) -> torch.Tensor:
        """
        Returns a mask of all True values (no masking).
        """
        return torch.ones(data.y.shape[0], dtype=torch.bool, device=data.y.device)

    def _get_fixed_mask(self, data: HeteroData) -> torch.Tensor:
        """
        Returns a (random) fixed mask based on the pre-defined mask ratio.
        """

        # check if we have a batch graph or not
        batch_size = getattr(data, "num_graphs", 1)

        # if the _mask attr has not been set yet, create it based on the mask ratio and batch size
        if not hasattr(self, "_mask"):
            mask_size = data.y.shape[0] // batch_size
            self._mask = torch.rand(mask_size, device=data.y.device) < self.mask_ratio

        # repeat the mask for each graph in the batch
        # if batch_size == 1, this will just return the original mask
        return self._mask.repeat(batch_size)

    def _get_predefined_mask(self, data: HeteroData) -> torch.Tensor:
        """
        Returns the predefined mask for the current dataset based on the mask ratio.
        """

        # check if we have a batch graph or not
        batch_size = getattr(data, "num_graphs", 1)

        # if the _mask attr has not been set yet, create it by loading the predefined mask
        if not hasattr(self, "_mask"):
            mask_path = self.dataset_path / "link_mask.pt"

            if not mask_path.exists():
                raise FileNotFoundError(f"Predefined mask file not found at {mask_path}")

            mask = torch.load(mask_path, weights_only=False)
            mask = mask < self.mask_ratio  # convert to boolean mask based on ratio
            self._mask = mask.to(device=data.y.device)

        # repeat the mask for each graph in the batch
        # if batch_size == 1, this will just return the original mask
        return self._mask.repeat(batch_size)

    def _calculate_physics_losses(
        self, pred: torch.Tensor, data: HeteroData, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the individual physics-based loss components (VCR, flow, conservation).
        This is a shared helper method for physics-informed losses.
        """
        self._validate_custom_data(data)

        real_edge = ("nodes", "real", "nodes")

        ## prepare masks - non-blocking
        # mask_float is used to multiply the error matrix to zero out unmasked entries
        mask_float = mask.float()
        if mask_float.dim() < pred.dim():
            # if mask is 1D and pred is 2D, unsqueeze mask to match dimensions for broadcasting
            mask_float = mask_float.unsqueeze(1)

        # mask_sum is used to compute the number of valid entries for averaging the loss
        mask_sum = mask_float.sum().clamp_min(1e-3)

        # 1. VCR loss
        vcr_loss = F.l1_loss(pred, data.y, reduction="none")
        vcr_loss = vcr_loss * mask_float
        vcr_loss = vcr_loss.sum() / mask_sum

        # 2. Flow loss
        real_capacity = self.transform.inverse_transform(
            data[real_edge].edge_capacity, feature="edge_capacity"
        )
        real_vcr = self.transform.inverse_transform(pred, feature="target")
        true_flow = self.transform.inverse_transform(data[real_edge].edge_flow, feature="edge_flow")
        pred_flow = real_vcr * real_capacity

        flow_loss = F.l1_loss(pred_flow, true_flow, reduction="none")
        flow_loss = flow_loss * mask_float
        flow_loss = flow_loss.sum() / mask_sum

        # 3. Conservation loss
        edge_index = data[real_edge].edge_index
        num_nodes = data["nodes"].net_demand.shape[0]
        inflow = scatter(pred_flow, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        outflow = scatter(pred_flow, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        pred_demand = inflow - outflow
        net_demand = data["nodes"].net_demand
        conservation_loss = F.l1_loss(pred_demand, net_demand)

        return vcr_loss, flow_loss, conservation_loss

    def _l1_loss(self, pred, data, mask) -> tuple[torch.Tensor, dict]:
        target = data.y
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        return F.l1_loss(pred, target), {}

    def _l2_loss(self, pred, data, mask) -> tuple[torch.Tensor, dict]:
        target = data.y
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        return F.mse_loss(pred, target), {}

    def _rmsn_loss(self, pred, data, mask) -> tuple[torch.Tensor, dict]:
        """
        Normalized Root Mean Square Error (RMSN).

        Computes the RMSE (sqrt of mean squared error) between `pred` and `data.y`
        and normalizes it by the mean absolute value of the true target to avoid
        sign issues. If a transform has been registered (via `register_transform`),
        both `pred` and `target` are inverse-transformed back to the original data
        scale before computing the metric.

        Returns:
            Tuple[loss_tensor, dict] where `loss_tensor` is the scalar NRMSE
            (suitable for backprop) and the dict contains diagnostic entries.
        """
        target = data.y
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        # if available, compute on the original scale by inverse transforming.
        if hasattr(self, "transform"):
            pred_true = self.transform.inverse_transform(pred, feature="target")
            target_true = self.transform.inverse_transform(target, feature="target")
        else:
            pred_true = pred
            target_true = target

        # compute RMSE
        mse = F.mse_loss(pred_true, target_true)
        rmse = torch.sqrt(mse)

        # normalize by mean absolute true value; clamp to avoid division by zero.
        eps = torch.tensor(1e-8, device=rmse.device, dtype=rmse.dtype)
        denom = torch.mean(torch.abs(target_true))
        denom_safe = denom.clamp_min(eps)

        rmsn = rmse / denom_safe

        losses_log = {
            "rmse": rmse.detach(),
            "nrmse": rmsn.detach(),
            "nrmse_denom": denom.detach(),
        }

        return rmsn, losses_log

    def _custom_loss(self, pred, data, mask) -> tuple[torch.Tensor, dict]:
        """
        Physics-informed loss with fixed weights.
        """
        vcr_loss, flow_loss, conservation_loss = self._calculate_physics_losses(pred, data, mask)

        # Components in order matching LOSS_COMPONENT_NAMES
        loss_components = [vcr_loss, flow_loss, conservation_loss]
        weights = [
            self.weight_vars["w_vcr"],
            self.weight_vars["w_flow"],
            self.weight_vars["w_conservation"],
        ]

        total_loss = torch.zeros((), device=pred.device)
        losses_log = {}

        for name, loss, weight in zip(LOSS_COMPONENT_NAMES, loss_components, weights):
            weighted = weight * loss
            total_loss = total_loss + weighted

            losses_log[f"unweighted_{name}_loss"] = loss.detach()
            losses_log[f"weighted_{name}_loss"] = weighted.detach()

        return total_loss, losses_log
