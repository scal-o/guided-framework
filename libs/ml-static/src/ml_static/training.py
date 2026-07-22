from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import autocast
from torch.amp import GradScaler

from ml_static.config import TrainingFreezeConfig

if TYPE_CHECKING:
    from ml_static.losses import LossWrapper

# Lazy AMP gradient scaler for 16-bit mixed precision training on CUDA GPUs
_SCALER: GradScaler | None = None


def get_scaler() -> GradScaler:
    """Lazy initializer for AMP GradScaler."""
    global _SCALER
    if _SCALER is None:
        _SCALER = GradScaler()
    return _SCALER


def freeze_parameters(
    model: nn.Module,
    *,
    enabled: bool,
    exclude: list[str] | None = None,
    keep: list[str] | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """
    Freeze specific model parameters by substring matching on parameter names.

    Semantics:
      - Any parameter containing a substring in `exclude` is marked frozen (requires_grad = False).
      - Parameters matching a substring in `keep` remain trainable even if they match `exclude`.

    Args:
        model: Model whose parameters may be frozen.
        enabled: If False, all parameters remain trainable.
        exclude: List of parameter name substrings to freeze.
        keep: List of parameter name substrings to keep trainable.
        keywords: Deprecated alias for `exclude`.

    Returns:
        Dict reporting frozen and trainable parameter counts and details.
    """
    if exclude is None:
        exclude = []
    if keep is None:
        keep = []

    params_rows: list[dict] = []
    frozen_count = 0
    trainable_count = 0

    # If freezing is disabled, return report with all parameters trainable
    if not enabled:
        params_rows = [
            {"parameter_name": name, "frozen": False} for name, _ in model.named_parameters()
        ]
        return {
            "exclude": list(exclude),
            "keep": list(keep),
            "params": params_rows,
            "frozen_count": 0,
            "trainable_count": len(params_rows),
        }

    if len(exclude) == 0:
        raise ValueError("'exclude' must be non-empty when freezing is enabled")

    for name, param in model.named_parameters():
        matches_exclude = any(k in name for k in exclude)
        matches_keep = any(k in name for k in keep)
        should_freeze = matches_exclude and not matches_keep

        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1

        params_rows.append({"parameter_name": name, "frozen": bool(should_freeze)})

    return {
        "exclude": list(exclude),
        "keep": list(keep),
        "params": params_rows,
        "frozen_count": frozen_count,
        "trainable_count": trainable_count,
    }


def freeze_parameters_from_config(model: nn.Module, cfg) -> dict:
    """
    Apply parameter freezing using a parsed TrainingFreezeConfig object.

    Args:
        model: PyTorch model whose parameters will be frozen.
        cfg: TrainingFreezeConfig object containing enabled, exclude, and keep lists.

    Returns:
        Summary dict containing freezing results and parameter counts.
    """
    if cfg is None:
        raise ValueError("freeze config must be provided to freeze_parameters_from_config")

    if not isinstance(cfg, TrainingFreezeConfig):
        raise TypeError("cfg must be an instance of TrainingFreezeConfig")

    return freeze_parameters(model, enabled=cfg.enabled, exclude=cfg.exclude, keep=cfg.keep)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: LossWrapper,
    graph,
    device: torch.device,
) -> tuple[float, dict]:
    """
    Perform a single training step on a batch of graph data.

    Uses AMP (float16) on CUDA GPUs, and falls back to standard float32 precision on CPU.
    """
    use_amp = (device.type == "cuda")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        pred = model(graph)
        loss, loss_components = criterion(pred, graph)

    if use_amp:
        scaler = get_scaler()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.detach(), loss_components

    return loss.detach(), loss_components


def validate(
    model: nn.Module,
    criterion: LossWrapper,
    graph,
    device: torch.device,
) -> tuple[float, dict]:
    """
    Perform validation evaluation on a batch of graph data without updating gradients.
    """
    use_amp = (device.type == "cuda")

    model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(graph)
            loss, loss_components = criterion(pred, graph, evaluate=True)

    return loss.detach(), loss_components


def run_epoch(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss: LossWrapper,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, dict, dict]:
    """
    Run a full epoch of training and validation over the respective dataloaders.

    Returns:
        Tuple of (average train loss, average val loss, train components dict, val components dict).
    """
    e_train_loss = 0.0
    e_val_loss = 0.0
    train_batches = 0
    val_batches = 0

    train_components_acc: dict[str, float] = {}
    val_components_acc: dict[str, float] = {}

    # 1. Training Phase
    train_len = len(train_loader)
    for step, data in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        train_loss, train_components = train(model, optimizer, loss, data, device)

        e_train_loss += train_loss
        train_batches += 1

        for key, value in train_components.items():
            train_components_acc[key] = train_components_acc.get(key, 0.0) + value

        if step + 1 == train_len:
            break

    # 2. Validation Phase
    val_len = len(val_loader)
    for step, data in enumerate(val_loader):
        data = data.to(device, non_blocking=True)
        val_loss, val_components = validate(model, loss, data, device)

        e_val_loss += val_loss
        val_batches += 1

        for key, value in val_components.items():
            val_components_acc[key] = val_components_acc.get(key, 0.0) + value

        if step + 1 == val_len:
            break

    # Compute averages across batches
    e_train_loss /= train_batches
    e_val_loss /= val_batches

    for key in train_components_acc:
        train_components_acc[key] /= train_batches

    for key in val_components_acc:
        val_components_acc[key] /= val_batches

    return e_train_loss, e_val_loss, train_components_acc, val_components_acc


def run_test(
    model: nn.Module,
    test_loader,
    loss: LossWrapper,
    device: torch.device,
) -> float:
    """
    Evaluate model on the test dataset split and return average test loss.
    """
    test_loss = 0.0
    test_batches = 0

    for data in test_loader:
        data = data.to(device)
        loss_out, _ = validate(model, loss, data, device)
        test_loss += loss_out
        test_batches += 1

    test_loss /= test_batches
    return test_loss
