"""
Loss curriculum scheduling utilities.

Currently supports scheduling the *conservation* component weight for the
physics-informed loss.

Design goals:
- Small, dependency-free, and easy to extend.
- Declarative + reproducible: behavior driven by config dataclasses.
- Encapsulated runtime behavior: schedule phases implement matching/value logic.

Epoch conventions:
- The training loop in this project uses 1-based epoch indexing.
- All schedule ranges are inclusive.

Extending to other components:
- Change `LossSchedulePhaseConfig.component` to allow more names.
- Update `_SUPPORTED_COMPONENTS` and `weights_for_epoch` accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from ml_static.config import LossCurriculumConfig, LossSchedulePhaseConfig

_SUPPORTED_COMPONENTS: Final[set[str]] = {"conservation"}
_SUPPORTED_KINDS: Final[set[str]] = {"constant", "linear_ramp", "sigmoid_ramp"}


@dataclass(frozen=True)
class LossWeightsUpdate:
    """Computed weight updates for a given epoch.

    Attributes:
        weights: Mapping of loss weight key -> scheduled float value.
        phase_kind: The schedule phase kind that matched (e.g., 'constant', 'linear_ramp').
        phase_component: The component being scheduled (currently only 'conservation').
    """

    weights: dict[str, float]
    phase_kind: str | None = None
    phase_component: str | None = None


@dataclass(frozen=True)
class LossSchedulePhase:
    """Runtime schedule phase built from `LossSchedulePhaseConfig`.

    This class encapsulates:
    - validation of the phase structure
    - determining whether a phase applies at a given epoch
    - computing the scheduled value for a given epoch
    """

    kind: str
    component: str

    # constant
    value: float | None
    from_epoch: int | None
    until_epoch: int | None

    # linear_ramp and sigmoid_ramp
    start_value: float | None
    end_value: float | None
    start_epoch: int | None
    end_epoch: int | None

    @classmethod
    def from_config(cls, cfg: LossSchedulePhaseConfig) -> LossSchedulePhase:
        """Build a runtime phase from the config dataclass."""
        phase = cls(
            kind=str(cfg.kind),
            component=str(cfg.component),
            value=cfg.value,
            from_epoch=cfg.from_epoch,
            until_epoch=cfg.until_epoch,
            start_value=cfg.start_value,
            end_value=cfg.end_value,
            start_epoch=cfg.start_epoch,
            end_epoch=cfg.end_epoch,
        )
        phase.validate()
        return phase

    def validate(self) -> None:
        """Validate phase structure and value ranges."""
        if self.component not in _SUPPORTED_COMPONENTS:
            supported = ", ".join(sorted(_SUPPORTED_COMPONENTS))
            raise ValueError(
                f"Unsupported scheduled component '{self.component}'. Supported: {supported}."
            )

        if self.kind not in _SUPPORTED_KINDS:
            supported = ", ".join(sorted(_SUPPORTED_KINDS))
            raise ValueError(
                f"Unsupported schedule phase kind '{self.kind}'. Supported: {supported}."
            )

        if self.kind == "constant":
            if self.value is None:
                raise ValueError("Schedule phase kind='constant' requires 'value'.")
            if self.from_epoch is None and self.until_epoch is None:
                raise ValueError(
                    "Schedule phase kind='constant' requires at least one of 'from_epoch' or 'until_epoch'."
                )
            if self.from_epoch is not None and self.from_epoch < 1:
                raise ValueError(f"'from_epoch' must be >= 1, got {self.from_epoch}.")
            if self.until_epoch is not None and self.until_epoch < 1:
                raise ValueError(f"'until_epoch' must be >= 1, got {self.until_epoch}.")
            if (
                self.from_epoch is not None
                and self.until_epoch is not None
                and self.until_epoch < self.from_epoch
            ):
                raise ValueError(
                    f"Invalid constant phase range: from_epoch={self.from_epoch}, until_epoch={self.until_epoch}."
                )

        if self.kind in ("linear_ramp", "sigmoid_ramp"):
            missing = [
                name
                for name in (
                    "start_value",
                    "end_value",
                    "start_epoch",
                    "end_epoch",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"Schedule phase kind='{self.kind}' is missing required field(s): "
                    + ", ".join(missing)
                )

            assert self.start_epoch is not None and self.end_epoch is not None
            if self.start_epoch < 1 or self.end_epoch < 1:
                raise ValueError(
                    f"'start_epoch' and 'end_epoch' must be >= 1, got start_epoch={self.start_epoch}, end_epoch={self.end_epoch}."
                )
            if self.end_epoch < self.start_epoch:
                raise ValueError(
                    f"Invalid {self.kind} epoch range: start_epoch={self.start_epoch}, end_epoch={self.end_epoch}."
                )

    def matches(self, epoch: int) -> bool:
        """Return True if the given epoch falls within this phase's active range."""
        if epoch < 1:
            raise ValueError(f"Epoch must be >= 1 (1-based), got {epoch}.")

        if self.kind == "constant":
            if self.from_epoch is not None and epoch < self.from_epoch:
                return False
            if self.until_epoch is not None and epoch > self.until_epoch:
                return False
            return True

        # linear_ramp
        assert self.start_epoch is not None and self.end_epoch is not None
        return self.start_epoch <= epoch <= self.end_epoch

    def value_for_epoch(self, epoch: int) -> float:
        """Compute the scheduled value for the given epoch under this phase."""
        if not self.matches(epoch):
            raise ValueError(f"Phase does not apply at epoch={epoch} (kind={self.kind}).")

        if self.kind == "constant":
            assert self.value is not None
            return float(self.value)

        # linear_ramp and sigmoid_ramp
        assert (
            self.start_value is not None
            and self.end_value is not None
            and self.start_epoch is not None
            and self.end_epoch is not None
        )

        if self.start_epoch == self.end_epoch:
            # Degenerate ramp: treat as constant at end_value for that single epoch.
            return float(self.end_value)

        # normalize epoch to [0, 1] range
        t = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

        if self.kind == "linear_ramp":
            return float(self.start_value + t * (self.end_value - self.start_value))

        # sigmoid_ramp: use sigmoid function for smooth S-shaped transition
        # maps [0, 1] through sigmoid with fixed steepness for smooth curve
        sigmoid_input = (t - 0.5) * 4.0
        sigmoid_value = 1.0 / (1.0 + math.exp(-sigmoid_input))
        return float(self.start_value + sigmoid_value * (self.end_value - self.start_value))


class LossCurriculumScheduler:
    """Computes scheduled loss weights per epoch based on a curriculum config."""

    def __init__(self, curriculum: LossCurriculumConfig) -> None:
        self.curriculum = curriculum
        self.phases: tuple[LossSchedulePhase, ...] = tuple(
            LossSchedulePhase.from_config(p) for p in curriculum.schedule
        )
        self._validate()

    @classmethod
    def from_config(cls, curriculum: LossCurriculumConfig) -> LossCurriculumScheduler:
        """Create scheduler from LossCurriculumConfig."""
        return cls(curriculum)

    def _validate(self) -> None:
        """Validate that the curriculum is well-formed and supported."""
        if not self.curriculum.enabled:
            return

        # `LossSchedulePhase.from_config()` already validates each phase. Here we
        # only enforce curriculum-level constraints.
        # (Reserved for future checks like overlap warnings, etc.)
        if not self.phases:
            raise ValueError(
                "loss.curriculum.enabled is true, but no schedule phases were provided."
            )

    def weights_for_epoch(self, epoch: int) -> LossWeightsUpdate:
        """Compute weight updates for the given epoch.

        Args:
            epoch: 1-based epoch index.

        Returns:
            LossWeightsUpdate with:
            - weights dict (possibly empty if no update applies)
            - optional phase metadata (kind/component) for logging
        """
        if not self.curriculum.enabled:
            return LossWeightsUpdate(weights={})

        # In case of overlapping phases, we resolve conflicts by "last match wins"
        # to make behavior explicit and easy to override.
        matched_phase: LossSchedulePhase | None = None
        matched_value: float | None = None

        for phase in self.phases:
            if phase.matches(epoch):
                matched_phase = phase
                matched_value = phase.value_for_epoch(epoch)

        if matched_phase is None or matched_value is None:
            return LossWeightsUpdate(weights={})

        # Map scheduled component -> LossWrapper weight key
        # For now, component == "conservation" -> "w_conservation"
        key = f"w_{matched_phase.component}"
        return LossWeightsUpdate(
            weights={key: float(matched_value)},
            phase_kind=matched_phase.kind,
            phase_component=matched_phase.component,
        )
