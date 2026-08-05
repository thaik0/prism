"""Strict configuration and chronological splits for the Milestone 3 predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping


MAX_RANDOM_SEED = 2**32 - 1
PREDICTOR_ALGORITHM_SETTINGS = {
    "activation_algorithm": "sklearn.linear_model.LogisticRegression",
    "activation_penalty": "l2",
    "activation_C": 1.0,
    "activation_solver": "lbfgs",
    "activation_class_weight": None,
    "intensity_algorithm": "sklearn.linear_model.Ridge",
    "intensity_fit_intercept": True,
    "prediction_horizon_windows": 1,
    "required_factor_demand_lags": 3,
}


class PredictorConfigError(ValueError):
    """Raised when predictor configuration or source dimensions are invalid."""


@dataclass(frozen=True)
class SplitBoundaries:
    """Chronological target-window boundaries."""

    num_windows: int
    train_end: int
    validation_end: int

    def split_code_for_target(self, target_window_id: int) -> int:
        if target_window_id < self.train_end:
            return 0
        if target_window_id < self.validation_end:
            return 1
        return 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_windows": self.num_windows,
            "train_windows": [0, self.train_end],
            "training_target_windows": [3, self.train_end],
            "validation_target_windows": [self.train_end, self.validation_end],
            "test_target_windows": [self.validation_end, self.num_windows],
            "train_end": self.train_end,
            "validation_end": self.validation_end,
            "split_codes": {"training": 0, "validation": 1, "test": 2},
        }


@dataclass(frozen=True)
class PredictorConfig:
    """Fully specified configurable inputs for the fixed predictor models."""

    fit_seed: int
    train_fraction: float
    validation_fraction: float
    activation_max_iter: int
    activation_tolerance: float
    intensity_ridge_alpha: float
    calibration_bins: int

    def __post_init__(self) -> None:
        _require_integer("fit_seed", self.fit_seed, minimum=0, maximum=MAX_RANDOM_SEED)
        _require_integer("activation_max_iter", self.activation_max_iter, minimum=1)
        _require_integer("calibration_bins", self.calibration_bins, minimum=2)
        for name in ("train_fraction", "validation_fraction"):
            value = _require_number(name, getattr(self, name), exclusive_minimum=0.0)
            if value >= 1.0:
                raise PredictorConfigError(f"{name} must be less than 1")
            object.__setattr__(self, name, value)
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise PredictorConfigError(
                "train_fraction plus validation_fraction must be less than 1"
            )
        tolerance = _require_number(
            "activation_tolerance", self.activation_tolerance, exclusive_minimum=0.0
        )
        alpha = _require_number(
            "intensity_ridge_alpha", self.intensity_ridge_alpha, minimum=0.0
        )
        object.__setattr__(self, "activation_tolerance", tolerance)
        object.__setattr__(self, "intensity_ridge_alpha", alpha)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PredictorConfig:
        if not isinstance(raw, Mapping):
            raise PredictorConfigError("predictor configuration root must be a JSON object")
        expected = {field.name for field in fields(cls)}
        supplied = set(raw)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing:
            raise PredictorConfigError(
                "missing required predictor configuration fields: " + ", ".join(missing)
            )
        if unknown:
            raise PredictorConfigError(
                "unknown predictor configuration fields: " + ", ".join(unknown)
            )
        return cls(**dict(raw))

    @classmethod
    def from_json(cls, path: str | Path) -> PredictorConfig:
        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise PredictorConfigError(
                f"invalid JSON in {config_path}: {error.msg}"
            ) from error
        return cls.from_dict(raw)

    def split_boundaries(self, num_windows: int) -> SplitBoundaries:
        _require_integer("num_windows", num_windows, minimum=1)
        train_end = math.floor(self.train_fraction * num_windows)
        validation_end = math.floor(
            (self.train_fraction + self.validation_fraction) * num_windows
        )
        if train_end <= 3:
            raise PredictorConfigError(
                "source trace is too short to construct lagged training examples"
            )
        if validation_end <= train_end:
            raise PredictorConfigError(
                "source trace is too short to construct validation examples"
            )
        if validation_end >= num_windows:
            raise PredictorConfigError(
                "source trace is too short to construct test examples"
            )
        return SplitBoundaries(num_windows, train_end, validation_end)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def to_resolved_dict(self) -> dict[str, Any]:
        return {**self.to_dict(), **PREDICTOR_ALGORITHM_SETTINGS}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PredictorConfigError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _require_integer(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredictorConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise PredictorConfigError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise PredictorConfigError(f"{name} must be at most {maximum}")


def _require_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictorConfigError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise PredictorConfigError(f"{name} must be finite")
    if minimum is not None and resolved < minimum:
        raise PredictorConfigError(f"{name} must be at least {minimum}")
    if exclusive_minimum is not None and resolved <= exclusive_minimum:
        raise PredictorConfigError(f"{name} must be greater than {exclusive_minimum}")
    return resolved
