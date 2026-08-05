"""Shared expected-benefit objective with greedy and exact placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


class ControllerError(ValueError):
    """Raised when placement inputs or an exact solve are invalid."""


@dataclass(frozen=True)
class PlacementSelection:
    target_record_ids: tuple[int, ...]
    objective_value: float
    solver_status: str


def record_benefits(
    forecast_accesses: Sequence[float],
    record_ids: Sequence[int],
    record_sizes: Mapping[int, int],
    resident_record_ids: set[int] | frozenset[int],
    fast_read_cost: float,
    slow_read_cost: float,
    promotion_cost_per_byte: float,
) -> dict[int, float]:
    """Calculate the one shared net-benefit formula for every record."""

    forecast = np.asarray(forecast_accesses, dtype=np.float64)
    ids = tuple(int(record_id) for record_id in record_ids)
    if forecast.shape != (len(ids),) or not np.all(np.isfinite(forecast)):
        raise ControllerError("forecast_accesses must be a finite record vector")
    if np.any(forecast < 0.0):
        raise ControllerError("forecast_accesses must be nonnegative")
    savings = slow_read_cost - fast_read_cost
    if savings <= 0.0 or promotion_cost_per_byte < 0.0:
        raise ControllerError("controller costs are invalid")
    benefits: dict[int, float] = {}
    for index, record_id in enumerate(ids):
        if record_id not in record_sizes:
            raise ControllerError(f"missing size for record {record_id}")
        promotion = (
            0.0
            if record_id in resident_record_ids
            else record_sizes[record_id] * promotion_cost_per_byte
        )
        benefits[record_id] = float(forecast[index] * savings - promotion)
    return benefits


def greedy_placement(
    benefits: Mapping[int, float],
    record_sizes: Mapping[int, int],
    capacity_bytes: int,
) -> PlacementSelection:
    """Select positive-benefit records by the specified deterministic density."""

    candidates = _candidates(benefits, record_sizes, capacity_bytes)
    ordered = sorted(
        candidates,
        key=lambda record_id: (
            -benefits[record_id] / record_sizes[record_id],
            -benefits[record_id],
            record_sizes[record_id],
            record_id,
        ),
    )
    selected: list[int] = []
    used = 0
    for record_id in ordered:
        size = record_sizes[record_id]
        if used + size <= capacity_bytes:
            selected.append(record_id)
            used += size
    target = tuple(sorted(selected))
    return PlacementSelection(
        target, sum(benefits[record_id] for record_id in target), "greedy"
    )


def exact_placement(
    benefits: Mapping[int, float],
    record_sizes: Mapping[int, int],
    capacity_bytes: int,
) -> PlacementSelection:
    """Solve the same positive-benefit 0/1 capacity problem to proven optimality."""

    candidates = tuple(sorted(_candidates(benefits, record_sizes, capacity_bytes)))
    if not candidates:
        return PlacementSelection((), 0.0, "optimal_empty")
    objective = -np.asarray([benefits[item] for item in candidates], dtype=np.float64)
    sizes = np.asarray([record_sizes[item] for item in candidates], dtype=np.float64)
    result = milp(
        c=objective,
        integrality=np.ones(len(candidates), dtype=np.int8),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(sizes, -np.inf, float(capacity_bytes)),
        options={"disp": False},
    )
    if not result.success or result.status != 0 or result.x is None:
        raise ControllerError(
            f"exact placement did not prove optimality: status={result.status}, "
            f"message={result.message}"
        )
    selected = tuple(
        record_id
        for record_id, value in zip(candidates, result.x, strict=True)
        if value > 0.5
    )
    if sum(record_sizes[item] for item in selected) > capacity_bytes:
        raise ControllerError("exact placement returned an infeasible selection")
    return PlacementSelection(
        selected,
        sum(benefits[item] for item in selected),
        f"optimal_status_{result.status}",
    )


def _candidates(
    benefits: Mapping[int, float],
    record_sizes: Mapping[int, int],
    capacity_bytes: int,
) -> tuple[int, ...]:
    if isinstance(capacity_bytes, bool) or capacity_bytes <= 0:
        raise ControllerError("capacity_bytes must be positive")
    candidates: list[int] = []
    for record_id, benefit in benefits.items():
        if record_id not in record_sizes or record_sizes[record_id] <= 0:
            raise ControllerError(f"record {record_id} has an invalid size")
        if not np.isfinite(benefit):
            raise ControllerError(f"record {record_id} has non-finite benefit")
        if benefit > 0.0 and record_sizes[record_id] <= capacity_bytes:
            candidates.append(record_id)
    return tuple(candidates)
