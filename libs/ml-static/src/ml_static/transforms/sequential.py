"""
Base class for sequential transform pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform

from ml_static.config import BuilderTransformConfig, ScalerTransformConfig
from ml_static.transforms.builders import BuilderTransform
from ml_static.transforms.scalers import ScalerTransform

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from ml_static.config import Config
    from ml_static.data import STADataset


class SequentialTransform(BaseTransform):
    """
    Base class for sequential transform pipelines.

    Applies a list of transforms in sequence, with a unified interface
    for both forward() and fit_dataset() operations.
    """

    def __init__(self, transforms: list) -> None:
        """
        Initialize with a list of transforms.

        Args:
            transforms: List of BaseTransform instances to apply in sequence.
        """
        self.transforms = transforms
        self.scalers = [t for t in transforms if isinstance(t, ScalerTransform)]
        self.builders = [t for t in transforms if isinstance(t, BuilderTransform)]

    @classmethod
    def from_config(cls, config: Config, stage: str | None = None) -> Self:
        """
        Create SequentialTransform from configuration.

        Args:
            config: Root configuration object.
            stage: Stage of the pipeline ('pre' or 'post').

        Returns:
            SequentialTransform instance.
        """
        if stage is None:
            raise ValueError("stage parameter is required for SequentialTransform")

        transforms = []

        # select transform configurations based on stage
        if stage == "pre":
            iterator = config.transforms.pre
        elif stage == "post":
            iterator = config.transforms.post
        else:
            raise ValueError(f"Invalid transform stage: {stage}. Must be 'pre' or 'post'.")

        # build transforms from configurations using cascaded from_config
        for transform_cfg in iterator:
            if isinstance(transform_cfg, BuilderTransformConfig):
                transforms.append(BuilderTransform.from_config(transform_cfg))
            elif isinstance(transform_cfg, ScalerTransformConfig):
                transforms.append(ScalerTransform.from_config(transform_cfg))
            else:
                raise ValueError(f"Unknown transform config type: {type(transform_cfg)}")

        return cls(transforms=transforms)

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Apply all transforms sequentially.

        Args:
            data: Input HeteroData object.

        Returns:
            Transformed HeteroData with all transforms applied.
        """
        for transform in self.transforms:
            data = transform(data)
        return data

    def fit_dataset(self, dataset: STADataset) -> None:
        """
        Fit all transforms on the dataset.

        Only transforms that need fitting (e.g., scalers) will perform operations.
        Stateless transforms (e.g., builders) will be skipped.

        Args:
            dataset: Dataset to fit transforms on.
        """
        for transform in self.transforms:
            transform.fit_dataset(dataset)

    def inverse_transform(
        self, x: torch.Tensor | np.ndarray, feature: str
    ) -> torch.Tensor | np.ndarray:
        """
        Apply inverse transformations in reverse order.

        Args:
            data: Input HeteroData object.

        Returns:
            HeteroData after applying inverse transformations.
        """

        target_scaler = [t for t in self.scalers if t.is_label]
        feature_scalers = [t for t in self.scalers if not t.is_label]

        transform = None
        if feature == "target":
            transform = target_scaler[0] if len(target_scaler) > 0 else None
        else:
            for scaler in feature_scalers:
                spec, scaler_feature = scaler.target
                if feature == scaler_feature:
                    transform = scaler
                    break

        if transform is None:
            return x

        return transform.inverse_transform(x)

    def to(self, device: torch.device | str) -> Self:
        """
        Move all transforms to the specified device.

        Args:
            device: Target device.

        Returns:
            Self with all transforms moved to the target device.
        """
        for scaler in self.scalers:
            scaler.to(device)
        return self

    def __repr__(self) -> str:
        transforms_str = ",\n  ".join([str(t) for t in self.transforms])
        return f"{self.__class__.__name__}([\n  {transforms_str}\n])"
