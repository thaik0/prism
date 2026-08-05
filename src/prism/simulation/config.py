"""Strict configuration for the Milestone 4 controlled simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence


class SimulationConfigError(ValueError):
    """Raised when simulation configuration or source sizes are invalid."""


@dataclass(frozen=True)
class SimulationConfig:
    """The four fixed byte-capacity and cost parameters."""

    fast_capacity_bytes: int
    fast_read_cost: float
    slow_read_cost: float
    promotion_cost_per_byte: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.fast_capacity_bytes, bool)
            or not isinstance(self.fast_capacity_bytes, int)
            or self.fast_capacity_bytes <= 0
        ):
            raise SimulationConfigError(
                "fast_capacity_bytes must be a positive integer"
            )
        for name in (
            "fast_read_cost",
            "slow_read_cost",
            "promotion_cost_per_byte",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SimulationConfigError(f"{name} must be a finite number")
            resolved = float(value)
            if not math.isfinite(resolved) or resolved < 0.0:
                raise SimulationConfigError(
                    f"{name} must be a finite nonnegative number"
                )
            object.__setattr__(self, name, resolved)
        if self.slow_read_cost <= self.fast_read_cost:
            raise SimulationConfigError(
                "slow_read_cost must be greater than fast_read_cost"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SimulationConfig:
        if not isinstance(raw, Mapping):
            raise SimulationConfigError("simulation configuration must be an object")
        expected = {field.name for field in fields(cls)}
        missing = sorted(expected - set(raw))
        unknown = sorted(set(raw) - expected)
        if missing:
            raise SimulationConfigError(
                "missing required simulation configuration fields: "
                + ", ".join(missing)
            )
        if unknown:
            raise SimulationConfigError(
                "unknown simulation configuration fields: " + ", ".join(unknown)
            )
        return cls(**dict(raw))

    @classmethod
    def from_json(cls, path: str | Path) -> SimulationConfig:
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise SimulationConfigError(f"invalid simulation JSON: {error.msg}") from error
        return cls.from_dict(raw)

    def validate_record_sizes(self, record_sizes: Sequence[int]) -> None:
        if not record_sizes:
            raise SimulationConfigError("source must contain at least one record")
        for size in record_sizes:
            if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
                raise SimulationConfigError(
                    "source record sizes must be positive integers"
                )
        if min(record_sizes) > self.fast_capacity_bytes:
            raise SimulationConfigError(
                "fast_capacity_bytes cannot hold any source record"
            )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SimulationConfigError(f"duplicate simulation JSON field: {key}")
        result[key] = value
    return result
