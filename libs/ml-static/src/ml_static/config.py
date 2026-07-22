"""
Configuration management for ml-static.

This module provides a clean separation between:
1. Configuration schema (dataclasses) - pure data with validation
2. Configuration loading - YAML parsing and schema construction
3. Factory methods remain on domain classes as thin wrappers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

# =============================================================================
# Model Architecture Protocol
# =============================================================================


class ModelArchitectureConfig(Protocol):
    """
    Protocol that all model architecture configs must satisfy.

    This defines the minimal interface without forcing inheritance.
    Each model can define its own dataclass structure in its module.
    """

    @classmethod
    def from_dict(cls, data: dict) -> ModelArchitectureConfig:
        """Parse config from dictionary.

        Args:
            data: Dictionary containing architecture configuration.

        Returns:
            Parsed architecture config instance.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for backward compatibility."""
        ...

    def validate(self) -> None:
        """Validate architecture-specific constraints."""
        ...


# =============================================================================
# Configuration Schema (Pure Data Structures)
# =============================================================================


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer."""

    type: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = 0.001

    # SGD parameters
    momentum: float = 0.9

    # Adam/AdamW parameters
    weight_decay: float = 0.0
    amsgrad: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.learning_rate}")


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: Literal["reduce_on_plateau", "one_cycle", "cosine_annealing"] = "reduce_on_plateau"

    # ReduceLROnPlateau parameters
    factor: float = 0.5
    patience: int = 30

    # Warmup: skip scheduler stepping for the first N epochs
    warmup_epochs: int = 0

    # OneCycleLR parameters
    max_lr: float = 0.01
    epochs: int | None = None
    steps_per_epoch: int | None = None
    pct_start: float = 0.3
    anneal_strategy: Literal["cos", "linear"] = "cos"

    # CosineAnnealingWarmRestarts parameters
    T_0: int = 20
    T_mult: int = 2
    eta_min: float = 1e-6


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    enabled: bool = False
    patience: int = 50
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"

    # Warmup: ignore early stopping checks for the first N epochs
    warmup_epochs: int = 0


@dataclass(frozen=True)
class LossSchedulePhaseConfig:
    """Configuration for a single loss curriculum phase.

    This is intentionally a lightweight schema container. Detailed validation and
    runtime behavior (matching epochs, computing values) are implemented in the
    curriculum module that consumes this config.

    Supported kinds (validated by the consumer):
        - constant
        - linear_ramp

    Epoch conventions:
        - Epochs are 1-based in the training loop.
        - Ranges are inclusive.
    """

    kind: Literal["constant", "linear_ramp"]

    # Which component to schedule (currently only "conservation" is supported at runtime)
    component: Literal["conservation"] = "conservation"

    # constant
    value: float | None = None
    from_epoch: int | None = None
    until_epoch: int | None = None

    # linear_ramp
    start_value: float | None = None
    end_value: float | None = None
    start_epoch: int | None = None
    end_epoch: int | None = None


@dataclass(frozen=True)
class LossCurriculumConfig:
    """Configuration for loss curriculum scheduling."""

    enabled: bool = False
    schedule: tuple[LossSchedulePhaseConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LossConfig:
    """Configuration for loss function."""

    type: Literal["mse", "l1", "l2", "rmsn", "custom"] = "mse"
    kwargs: dict[str, Any] = field(default_factory=dict)
    curriculum: LossCurriculumConfig = field(default_factory=LossCurriculumConfig)
    mask_mode: Literal[None, "fixed", "random"] = None
    mask_ratio: float | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset."""

    name: str = "anaheim_a"
    path: Path = field(default_factory=lambda: Path("data"))
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    max_nodes: int | None = None

    def __post_init__(self) -> None:
        if abs(sum(self.split) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(self.split)}")

    @property
    def full_path(self) -> Path:
        """Get full path to dataset directory."""
        return self.path / self.name


@dataclass(frozen=True)
class TrainingInitConfig:
    """Configuration for model initialization in fine-tuning mode.

    Notes:
        For finetuning, we support initializing model weights from:
        - an MLflow run (default workflow; if run_id is omitted, uses last run)
        - a local checkpoint path (future-proofing; not required by current workflow)
    """

    source: Literal["mlflow_run", "checkpoint_path"] = "mlflow_run"

    # MLflow run id to load; if None, defaults to last run (RunContext behavior)
    run_id: str | None = None
    download_path: Path = field(default_factory=lambda: Path("downloaded_models"))
    force_download: bool = False

    # local checkpoint path for future use / experiments
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class TrainingFreezeConfig:
    """Configuration for freezing model parameters during fine-tuning.

    Notes:
        - If `enabled` is True, `exclude` must be non-empty.
        - Any parameter whose name contains a substring listed in `exclude` will be
          frozen, unless its name also contains a substring listed in `keep`. The
          `keep` list takes precedence over `exclude`.
    """

    enabled: bool = False
    exclude: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training."""

    mode: Literal["run", "finetune"] = "run"
    init: TrainingInitConfig | None = None

    # Optional fine-tuning parameter freezing policy
    freeze: TrainingFreezeConfig | None = None

    epochs: int = 100
    batch_size: int = 32
    seed: int | None = None  # None means random seed


@dataclass
class ModelConfigWithArchitecture:
    """Model configuration with typed architecture."""

    type: str = "HetGAT"

    # Architecture loaded dynamically based on type
    architecture: ModelArchitectureConfig | None = None


@dataclass(frozen=True)
class BuilderTransformConfig:
    """Configuration for builder transform in pipeline."""

    builder: str


@dataclass(frozen=True)
class ScalerTransformConfig:
    """
    Unified configuration for scaler transforms.

    Handles both target scalers and feature scalers:
    - Target scalers: specify target="y"
    - Feature scalers: specify type="nodes" (or edge type) and feature="demand"
    """

    transform: str  # Required: 'log', 'norm', 'minmax'

    # For target scalers (labels): specify target="y"
    target: str | None = None

    # For feature scalers: specify type and feature
    type: str | list[str] | None = None  # node type or edge type (list for edges)
    feature: str | None = None

    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that only one of the two configurations is specified."""
        # check that exactly one configuration is used
        has_target = self.target is not None
        has_type_feature = self.type is not None and self.feature is not None

        if has_target and has_type_feature:
            raise ValueError(
                "Cannot specify both 'target' and 'type'+'feature'. "
                "Use 'target' for labels or 'type'+'feature' for node/edge features."
            )

        if not has_target and not has_type_feature:
            raise ValueError(
                "Must specify either 'target' (for labels) or both 'type' and 'feature' (for node/edge features)."
            )

        # validate target format (if specified)
        if has_target:
            if not isinstance(self.target, str):
                raise TypeError(f"Target must be str. Got {type(self.target).__name__}")
            if self.target != "y":
                raise ValueError(f"Target must be 'y' for labels. Got: '{self.target}'")

        # validate type + feature format (if specified)
        if has_type_feature:
            if self.type is None or self.feature is None:
                raise ValueError("Both 'type' and 'feature' must be specified together.")

            if not isinstance(self.feature, str):
                raise TypeError(f"Feature must be str. Got {type(self.feature).__name__}")

    @property
    def scaler_target(self) -> str | tuple:
        """Get target in format expected by Scaler class.

        Normalizes type specifications for PyTorch Geometric compatibility:
        - Single-element lists/tuples become strings (node types)
        - Multi-element lists become tuples (edge types)
        - Strings remain as-is
        """
        # return target directly if specified
        if self.target is not None:
            return self.target

        # convert type + feature to (type_spec, feature) tuple
        assert self.type is not None and self.feature is not None

        # Normalize type_spec for PyG HeteroData indexing
        if isinstance(self.type, list):
            # Single-element list -> string (node type)
            if len(self.type) == 1:
                type_spec = self.type[0]
            # Multi-element list -> tuple (edge type)
            else:
                type_spec = tuple(self.type)
        else:
            # Already a string
            type_spec = self.type

        return (type_spec, self.feature)


@dataclass(frozen=True)
class TransformsConfig:
    """Configuration for all transforms (pre and post)."""

    pre: tuple[BuilderTransformConfig, ...] = field(default_factory=tuple)
    post: tuple[BuilderTransformConfig | ScalerTransformConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ConfigOrigin:
    """Provenance information for a loaded configuration.

    Stores the exact YAML file paths used to construct a `Config`.
    """

    run_config_path: Path
    model_config_path: Path


@dataclass
class Config:
    """
    Root configuration object.

    This is the main entry point for all configuration. It holds
    all sub-configurations as typed, validated dataclasses.
    """

    model: ModelConfigWithArchitecture = field(default_factory=ModelConfigWithArchitecture)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)

    # keep raw config for MLflow logging
    _raw: dict = field(default_factory=dict, repr=False)
    _raw_model: dict = field(default_factory=dict, repr=False)

    @property
    def raw_config(self) -> dict:
        """Get raw configuration dictionary for logging."""
        return self._raw

    @property
    def raw_model(self) -> dict:
        """Get raw model configuration dictionary for logging."""
        return self._raw_model


# =============================================================================
# Configuration Loader
# =============================================================================


class ConfigLoader:
    """
    Loads configuration from YAML files into Config schema.

    Handles the complexity of:
    - Loading main run config
    - Loading model-specific config with transforms
    - Merging default transforms with overrides
    - Dynamic loading of model-specific architecture configs
    """

    # Registry mapping model types to their config classes
    MODEL_CONFIG_CLASSES: dict[str, type[ModelArchitectureConfig]] = {}

    @classmethod
    def register_config_class(cls, model_type: str):
        """Decorator to register a model architecture config class.

        Args:
            model_type: Model type identifier (e.g., 'HetGAT').

        Returns:
            Decorator function.

        Example:
            @ConfigLoader.register_config_class('HetGAT')
            @dataclass(frozen=True)
            class HetGATArchitectureConfig:
                ...
        """

        def decorator(config_class):
            if model_type in cls.MODEL_CONFIG_CLASSES:
                raise ValueError(f"Config class for '{model_type}' is already registered.")
            cls.MODEL_CONFIG_CLASSES[model_type] = config_class
            return config_class

        return decorator

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        model_config_path: Path | None = None,
    ) -> Config:
        """
        Load configuration from YAML file(s).

        Args:
            config_path: Path to the main run configuration file.
            model_config_path: Optional path to model configuration.
                If None, auto-discovers based on model name.

        Returns:
            Fully constructed Config object.

        Raises:
            FileNotFoundError: If required config files don't exist.
            yaml.YAMLError: If YAML is malformed.
            ValueError: If configuration values are invalid.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # parse model name (single source of truth for model identification)
        model_name = raw_config.get("model", {}).get("name", "HetGAT")

        # resolve and load model config path
        if model_config_path is None:
            model_config_path = config_path.parent / f"conf_model_{model_name}.yaml"

        if not model_config_path.exists():
            raise FileNotFoundError(
                f"Model config file not found: {model_config_path}. "
                f"Expected config for model '{model_name}'."
            )

        with open(model_config_path) as f:
            model_raw_config = yaml.safe_load(f)

        # parse main config sections
        training = cls._parse_training_config(raw_config.get("training", {}))
        optimizer = cls._parse_optimizer_config(raw_config.get("optimizer", {}))
        scheduler = cls._parse_scheduler_config(raw_config.get("scheduler", {}))
        early_stopping = cls._parse_early_stopping_config(raw_config.get("early_stopping", {}))
        loss = cls._parse_loss_config(raw_config.get("loss", {}))
        dataset = cls._parse_dataset_config(raw_config.get("dataset", {}))
        model = cls._parse_model_config(
            model_raw_config.get("type", ""), model_raw_config.get("architecture", {})
        )

        # parse transforms
        model_pre, model_post = cls._parse_transforms_config(model_raw_config.get("transforms", {}))
        target_pre, target_post = cls._parse_transforms_config(
            raw_config.get("training", {}).get("target", {})
        )

        # combine target transforms with model transforms
        transforms = TransformsConfig(
            pre=model_pre + target_pre,
            post=model_post + target_post,
        )

        return Config(
            model=model,
            training=training,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            loss=loss,
            dataset=dataset,
            transforms=transforms,
            _raw=raw_config,
            _raw_model=model_raw_config,
        )

    @classmethod
    def _parse_model_config(
        cls,
        model_type: str,
        architecture_data: dict,
    ) -> ModelConfigWithArchitecture:
        """Parse model architecture configuration.

        Args:
            model_type: Model type identifier.
            architecture_data: Architecture configuration dictionary.

        Returns:
            ModelConfigWithArchitecture instance.

        Raises:
            ValueError: If model type not registered or architecture invalid.
        """
        if model_type not in cls.MODEL_CONFIG_CLASSES:
            raise ValueError(
                f"No config class registered for model type '{model_type}'. "
                f"Available types: {list(cls.MODEL_CONFIG_CLASSES.keys())}"
            )

        config_class = cls.MODEL_CONFIG_CLASSES[model_type]
        try:
            architecture = config_class.from_dict(architecture_data)
            if hasattr(architecture, "validate"):
                architecture.validate()
        except Exception as e:
            raise ValueError(
                f"Failed to load architecture for model type '{model_type}': {e}"
            ) from e

        return ModelConfigWithArchitecture(
            type=model_type,
            architecture=architecture,
        )

    @classmethod
    def _parse_transforms_config(
        cls, transforms_data: dict
    ) -> tuple[
        tuple[BuilderTransformConfig, ...],
        tuple[BuilderTransformConfig | ScalerTransformConfig, ...],
    ]:
        """Parse transforms configuration (pre and post).

        Args:
            transforms_data: Transforms configuration dictionary.

        Returns:
            Tuple of (pre_transforms, post_transforms).
        """
        pre = cls._parse_transform_list(transforms_data.get("pre") or [])
        post = cls._parse_transform_list(transforms_data.get("post") or [])
        return pre, post

    @classmethod
    def _parse_training_config(cls, data: dict) -> TrainingConfig:
        """Parse training configuration section."""
        mode = (data.get("mode") or "run").lower()

        init_cfg = None
        init_data = data.get("init")
        if init_data is not None:
            if not isinstance(init_data, dict):
                raise TypeError(
                    f"'training.init' must be a dict if provided, got {type(init_data).__name__}."
                )

            source = (init_data.get("source") or "mlflow_run").lower()

            download_path_raw = init_data.get("download_path", "downloaded_models")
            checkpoint_path_raw = init_data.get("checkpoint_path")

            init_cfg = TrainingInitConfig(
                source=source,
                run_id=init_data.get("run_id"),
                download_path=Path(download_path_raw),
                force_download=bool(init_data.get("force_download", False)),
                checkpoint_path=Path(checkpoint_path_raw) if checkpoint_path_raw else None,
            )

        if mode == "finetune" and init_cfg is None:
            # Allow finetune configs to omit `init` entirely so the engine can default
            # to "last run" behavior (see RunContext). This is convenient for batch suites.
            init_cfg = TrainingInitConfig(source="mlflow_run", run_id=None)

        freeze_cfg = None
        freeze_data = data.get("freeze")
        if freeze_data is not None:
            if not isinstance(freeze_data, dict):
                raise TypeError(
                    f"'training.freeze' must be a dict if provided, got {type(freeze_data).__name__}."
                )

            enabled = bool(freeze_data.get("enabled", False))

            exclude_raw = freeze_data.get("exclude", [])
            if exclude_raw is None:
                exclude_raw = []
            if not isinstance(exclude_raw, list) or not all(
                isinstance(k, str) for k in exclude_raw
            ):
                raise TypeError("'training.freeze.exclude' must be a list of strings.")

            keep_raw = freeze_data.get("keep", [])
            if keep_raw is None:
                keep_raw = []
            if not isinstance(keep_raw, list) or not all(isinstance(k, str) for k in keep_raw):
                raise TypeError("'training.freeze.keep' must be a list of strings.")

            freeze_cfg = TrainingFreezeConfig(enabled=enabled, exclude=exclude_raw, keep=keep_raw)

            if freeze_cfg.enabled and len(freeze_cfg.exclude) == 0:
                raise ValueError(
                    "'training.freeze.exclude' must be non-empty when 'training.freeze.enabled' is true."
                )

        return TrainingConfig(
            mode=mode,
            init=init_cfg,
            freeze=freeze_cfg,
            epochs=int(data.get("epochs", 100)),
            batch_size=int(data.get("batch_size", 32)),
            seed=data.get("seed"),
        )

    @classmethod
    def _parse_optimizer_config(cls, data: dict) -> OptimizerConfig:
        """Parse optimizer configuration section."""
        return OptimizerConfig(
            type=data.get("type", "adam").lower(),
            learning_rate=data.get("learning_rate", 0.001),
            momentum=data.get("momentum", 0.9),
            weight_decay=data.get("weight_decay", 0.0),
            amsgrad=data.get("amsgrad", False),
        )

    @classmethod
    def _parse_scheduler_config(cls, data: dict) -> SchedulerConfig:
        """Parse scheduler configuration section."""
        return SchedulerConfig(
            type=data.get("type", "reduce_on_plateau").lower(),
            # ReduceLROnPlateau
            factor=data.get("factor", 0.1),
            patience=data.get("patience", 10),
            warmup_epochs=int(data.get("warmup_epochs", 0) or 0),
            # OneCycleLR
            max_lr=data.get("max_lr", 0.01),
            epochs=data.get("epochs"),
            steps_per_epoch=data.get("steps_per_epoch"),
            pct_start=data.get("pct_start", 0.3),
            anneal_strategy=data.get("anneal_strategy", "cos"),
            # CosineAnnealingWarmRestarts
            T_0=data.get("T_0", 10),
            T_mult=data.get("T_mult", 1),
            eta_min=data.get("eta_min", 0.0),
        )

    @classmethod
    def _parse_early_stopping_config(cls, data: dict) -> EarlyStoppingConfig:
        """Parse early stopping configuration section."""
        return EarlyStoppingConfig(
            enabled=data.get("enabled", False),
            patience=data.get("patience", 50),
            min_delta=data.get("min_delta", 0.0),
            mode=data.get("mode", "min"),
            warmup_epochs=int(data.get("warmup_epochs", 0) or 0),
        )

    @classmethod
    def _parse_loss_config(cls, data: dict) -> LossConfig:
        """Parse loss configuration section."""
        loss_type = data.get("type", "mse")

        curriculum_cfg = LossCurriculumConfig()
        curriculum_data = data.get("curriculum")
        if curriculum_data is not None:
            if not isinstance(curriculum_data, dict):
                raise TypeError(
                    f"'loss.curriculum' must be a dict if provided, got {type(curriculum_data).__name__}."
                )

            enabled = bool(curriculum_data.get("enabled", False))
            schedule_items = curriculum_data.get("schedule") or []
            if not isinstance(schedule_items, list):
                raise TypeError(
                    f"'loss.curriculum.schedule' must be a list, got {type(schedule_items).__name__}."
                )

            phases: list[LossSchedulePhaseConfig] = []
            for item in schedule_items:
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Each curriculum schedule item must be a dict, got {type(item).__name__}."
                    )
                if "kind" not in item:
                    raise ValueError("Each curriculum schedule item must include 'kind'.")

                phases.append(
                    LossSchedulePhaseConfig(
                        kind=(item.get("kind") or "").lower(),
                        component=(item.get("component") or "conservation").lower(),
                        value=item.get("value"),
                        from_epoch=item.get("from_epoch"),
                        until_epoch=item.get("until_epoch"),
                        start_value=item.get("start_value"),
                        end_value=item.get("end_value"),
                        start_epoch=item.get("start_epoch"),
                        end_epoch=item.get("end_epoch"),
                    )
                )

            curriculum_cfg = LossCurriculumConfig(enabled=enabled, schedule=tuple(phases))

        mask_mode = data.get("mask_mode")
        mask_ratio = data.get("mask_ratio")

        # keep kwargs behavior, but exclude reserved keys
        kwargs = {k: v for k, v in data.items() if k not in ("type", "curriculum")}
        return LossConfig(
            type=loss_type,
            kwargs=kwargs,
            curriculum=curriculum_cfg,
            mask_mode=mask_mode,
            mask_ratio=mask_ratio,
        )

    @classmethod
    def _parse_dataset_config(cls, data: dict) -> DatasetConfig:
        """Parse dataset configuration section."""
        split_dict = data.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
        split = tuple(split_dict.values()) if isinstance(split_dict, dict) else tuple(split_dict)

        return DatasetConfig(
            name=data.get("name", "anaheim_a"),
            path=Path(data.get("path", "data")),
            split=split,
            max_nodes=data.get("max_nodes"),
        )

    @classmethod
    def _parse_transform_list(
        cls, items: list[dict]
    ) -> tuple[BuilderTransformConfig | ScalerTransformConfig, ...]:
        """Parse a list of transforms (builders or scalers).

        Args:
            items: List of transform dictionaries.

        Returns:
            Tuple of parsed transform configs.

        Raises:
            ValueError: If transform structure is invalid.
        """
        configs: list[BuilderTransformConfig | ScalerTransformConfig] = []

        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Transform must be dict, got {type(item).__name__}")

            if "builder" in item:
                configs.append(BuilderTransformConfig(builder=item["builder"]))
            elif "scaler" in item:
                scaler_data = item["scaler"]
                if not isinstance(scaler_data, dict):
                    raise ValueError(f"Scaler data must be dict, got {type(scaler_data).__name__}")

                transform_type = scaler_data.get("transform")
                if not transform_type:
                    raise ValueError("Scaler must have 'transform' key")

                # extract target/type/feature
                target_value = scaler_data.get("target")
                type_value = scaler_data.get("type")
                feature_value = scaler_data.get("feature")

                # extract kwargs (exclude reserved keys)
                kwargs = {
                    k: v
                    for k, v in scaler_data.items()
                    if k not in ("transform", "target", "type", "feature")
                }

                configs.append(
                    ScalerTransformConfig(
                        transform=transform_type,
                        target=target_value,
                        type=type_value,
                        feature=feature_value,
                        kwargs=kwargs,
                    )
                )
            else:
                raise ValueError(f"Unknown transform type: {list(item.keys())}")

        return tuple(configs)


# =============================================================================
# Convenience function for loading
# =============================================================================


def load_config(
    config_path: Path,
    model_config_path: Path | None = None,
) -> Config:
    """
    Load configuration from YAML file(s).

    This is the main entry point for loading configuration.

    Args:
        config_path: Path to the main run configuration file.
        model_config_path: Optional path to model configuration.

    Returns:
        Fully constructed Config object.

    Example:
        >>> config = load_config(Path("conf_run.yaml"))
        >>> config.training.epochs
        100
        >>> config.optimizer.learning_rate
        0.001
    """
    return ConfigLoader.from_yaml(config_path, model_config_path)


def load_config_with_origin(
    config_path: Path,
    model_config_path: Path | None = None,
) -> tuple[Config, ConfigOrigin]:
    """Load configuration and return both the typed config and its source paths.

    Args:
        config_path: Path to the main run configuration file.
        model_config_path: Optional path to model configuration. If None, it is
            resolved based on the `model.name` found in the run config.

    Returns:
        Tuple of (Config, ConfigOrigin) where ConfigOrigin contains the exact YAML
        files used to construct the Config.
    """
    # If model_config_path is not provided, resolve it the same way as ConfigLoader.
    if model_config_path is None:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            raw_config = yaml.safe_load(f) or {}

        model_name = (raw_config.get("model") or {}).get("name", "HetGAT")
        model_config_path = config_path.parent / f"conf_model_{model_name}.yaml"

    config = ConfigLoader.from_yaml(config_path, model_config_path)
    origin = ConfigOrigin(run_config_path=config_path, model_config_path=model_config_path)
    return config, origin
