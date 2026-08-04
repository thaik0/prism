"""Strict configuration for the controlled workload generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping


PROBABILITY_SUM_TOLERANCE = 1e-12


class WorkloadConfigError(ValueError):
    """Raised when workload configuration is malformed."""


@dataclass(frozen=True)
class WorkloadConfig:
    """Fully resolved configuration for one deterministic workload run."""

    seed: int
    num_windows: int
    num_records: int
    num_working_sets: int
    num_users: int
    request_types: tuple[str, ...]
    operation_type_probabilities: dict[str, float]
    record_size_min_bytes: int
    record_size_max_bytes: int
    working_set_support_min: int
    working_set_support_max: int
    min_sessions_per_window: int
    max_sessions_per_window: int
    min_requests_per_session: int
    max_requests_per_session: int
    min_accesses_per_request: int
    max_accesses_per_request: int
    spontaneous_activation_probability: float
    precursor_probability_scale: float
    burst_duration_min_windows: int
    burst_duration_max_windows: int
    burst_intensity_min: float
    burst_intensity_max: float
    burst_access_weight: float
    baseline_access_weight: float
    noise_access_weight: float
    burst_intensity_context_weight: float = 0.0
    post_burst_cooldown_windows: int = 0

    def __post_init__(self) -> None:
        _require_integer("seed", self.seed)

        positive_integer_fields = (
            "num_windows",
            "num_records",
            "num_working_sets",
            "num_users",
            "record_size_min_bytes",
            "record_size_max_bytes",
            "working_set_support_min",
            "working_set_support_max",
            "min_sessions_per_window",
            "max_sessions_per_window",
            "min_requests_per_session",
            "max_requests_per_session",
            "min_accesses_per_request",
            "max_accesses_per_request",
            "burst_duration_min_windows",
            "burst_duration_max_windows",
        )
        for name in positive_integer_fields:
            _require_integer(name, getattr(self, name), minimum=1)
        _require_integer(
            "post_burst_cooldown_windows",
            self.post_burst_cooldown_windows,
            minimum=0,
        )

        if self.num_windows < 2:
            raise WorkloadConfigError("num_windows must be at least 2")

        _require_ordered_bounds(
            "record_size_min_bytes",
            self.record_size_min_bytes,
            "record_size_max_bytes",
            self.record_size_max_bytes,
        )
        _require_ordered_bounds(
            "working_set_support_min",
            self.working_set_support_min,
            "working_set_support_max",
            self.working_set_support_max,
        )
        _require_ordered_bounds(
            "min_sessions_per_window",
            self.min_sessions_per_window,
            "max_sessions_per_window",
            self.max_sessions_per_window,
        )
        _require_ordered_bounds(
            "min_requests_per_session",
            self.min_requests_per_session,
            "max_requests_per_session",
            self.max_requests_per_session,
        )
        _require_ordered_bounds(
            "min_accesses_per_request",
            self.min_accesses_per_request,
            "max_accesses_per_request",
            self.max_accesses_per_request,
        )
        _require_ordered_bounds(
            "burst_duration_min_windows",
            self.burst_duration_min_windows,
            "burst_duration_max_windows",
            self.burst_duration_max_windows,
        )
        if self.working_set_support_max > self.num_records:
            raise WorkloadConfigError(
                "working_set_support_max must not exceed num_records"
            )

        request_types = _validated_categories("request_types", self.request_types)
        object.__setattr__(self, "request_types", request_types)

        operation_probabilities = _validated_operation_probabilities(
            self.operation_type_probabilities
        )
        object.__setattr__(
            self, "operation_type_probabilities", operation_probabilities
        )

        probability_fields = (
            "spontaneous_activation_probability",
            "precursor_probability_scale",
            "burst_intensity_context_weight",
        )
        for name in probability_fields:
            value = _require_number(name, getattr(self, name), minimum=0.0, maximum=1.0)
            object.__setattr__(self, name, value)

        positive_number_fields = ("burst_intensity_min", "burst_intensity_max")
        for name in positive_number_fields:
            value = _require_number(name, getattr(self, name), exclusive_minimum=0.0)
            object.__setattr__(self, name, value)

        _require_ordered_bounds(
            "burst_intensity_min",
            self.burst_intensity_min,
            "burst_intensity_max",
            self.burst_intensity_max,
        )

        weight_fields = (
            "burst_access_weight",
            "baseline_access_weight",
            "noise_access_weight",
        )
        for name in weight_fields:
            value = _require_number(name, getattr(self, name), minimum=0.0)
            object.__setattr__(self, name, value)

        if self.baseline_access_weight == 0.0 and self.noise_access_weight == 0.0:
            raise WorkloadConfigError(
                "baseline_access_weight and noise_access_weight cannot both be zero"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> WorkloadConfig:
        """Validate an exact mapping and return a canonical configuration."""

        if not isinstance(raw, Mapping):
            raise WorkloadConfigError("configuration root must be a JSON object")

        expected = {field.name for field in fields(cls)}
        supplied = set(raw)
        optional_defaults = {
            "burst_intensity_context_weight": 0.0,
            "post_burst_cooldown_windows": 0,
        }
        missing = sorted(expected - supplied - set(optional_defaults))
        unknown = sorted(supplied - expected)
        if missing:
            raise WorkloadConfigError(
                f"missing required configuration fields: {', '.join(missing)}"
            )
        if unknown:
            raise WorkloadConfigError(
                f"unknown configuration fields: {', '.join(unknown)}"
            )

        resolved = {**optional_defaults, **dict(raw)}
        return cls(**resolved)

    @classmethod
    def from_json(cls, path: str | Path) -> WorkloadConfig:
        """Load strict JSON, rejecting duplicate object keys."""

        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise WorkloadConfigError(
                f"invalid JSON in {config_path}: {error.msg}"
            ) from error
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy artifact representation.

        A zero cooldown is omitted so every accepted pre-Milestone-5.5 workload
        remains byte-identical. New code that needs an explicit canonical value
        must use :meth:`to_resolved_dict`.
        """

        result = asdict(self)
        if result["post_burst_cooldown_windows"] == 0:
            del result["post_burst_cooldown_windows"]
        result["request_types"] = list(self.request_types)
        result["operation_type_probabilities"] = dict(
            self.operation_type_probabilities
        )
        return result

    def to_resolved_dict(self) -> dict[str, Any]:
        """Return a canonical representation with an explicit cooldown."""

        result = self.to_dict()
        result["post_burst_cooldown_windows"] = self.post_burst_cooldown_windows
        return dict(sorted(result.items()))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkloadConfigError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _require_integer(name: str, value: Any, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkloadConfigError(f"{name} must be an integer, not a boolean")
    if minimum is not None and value < minimum:
        raise WorkloadConfigError(f"{name} must be at least {minimum}")


def _require_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadConfigError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise WorkloadConfigError(f"{name} must be finite")
    if minimum is not None and resolved < minimum:
        raise WorkloadConfigError(f"{name} must be at least {minimum}")
    if maximum is not None and resolved > maximum:
        raise WorkloadConfigError(f"{name} must be at most {maximum}")
    if exclusive_minimum is not None and resolved <= exclusive_minimum:
        raise WorkloadConfigError(f"{name} must be greater than {exclusive_minimum}")
    return resolved


def _require_ordered_bounds(
    minimum_name: str,
    minimum_value: int | float,
    maximum_name: str,
    maximum_value: int | float,
) -> None:
    if minimum_value > maximum_value:
        raise WorkloadConfigError(
            f"{minimum_name} must not exceed {maximum_name}"
        )


def _validated_categories(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise WorkloadConfigError(f"{name} must be a nonempty list")
    categories: list[str] = []
    for category in value:
        if not isinstance(category, str) or not category:
            raise WorkloadConfigError(f"{name} entries must be nonempty strings")
        categories.append(category)
    if len(set(categories)) != len(categories):
        raise WorkloadConfigError(f"{name} entries must be unique")
    return tuple(categories)


def _validated_operation_probabilities(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise WorkloadConfigError(
            "operation_type_probabilities must be a nonempty object"
        )

    probabilities: dict[str, float] = {}
    for operation_type, probability in value.items():
        if not isinstance(operation_type, str) or not operation_type:
            raise WorkloadConfigError(
                "operation_type_probabilities keys must be nonempty strings"
            )
        probabilities[operation_type] = _require_number(
            f"operation_type_probabilities.{operation_type}",
            probability,
            minimum=0.0,
        )

    total = math.fsum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
        raise WorkloadConfigError(
            "operation_type_probabilities must sum to 1 within "
            f"{PROBABILITY_SUM_TOLERANCE:g}; got {total}"
        )
    return dict(sorted(probabilities.items()))
