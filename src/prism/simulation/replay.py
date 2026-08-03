"""Chronological validation warm-up and identical six-policy test replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from prism.simulation.config import SimulationConfig
from prism.simulation.controllers import (
    PlacementSelection,
    exact_placement,
    greedy_placement,
    record_benefits,
)
from prism.simulation.storage import Migration, StorageState
from prism.workload.models import ObservableEvent


POLICY_NAMES = (
    "lru",
    "lfu",
    "recent_demand_greedy",
    "predictive_greedy",
    "oracle_greedy",
    "oracle_exact",
)
BOUNDARY_POLICIES = frozenset(POLICY_NAMES[2:])
WINDOW_ARRAY_NAMES = (
    "access_count",
    "hit_count",
    "miss_count",
    "access_cost",
    "promotion_cost",
    "combined_cost",
    "promotion_count",
    "eviction_count",
    "bytes_promoted",
    "bytes_evicted",
    "end_occupancy_bytes",
    "end_resident_record_count",
)


class ReplayError(ValueError):
    """Raised when shared replay inputs or policy behavior are invalid."""


@dataclass(frozen=True)
class PolicyReplayResult:
    policy_names: tuple[str, ...]
    test_window_ids: np.ndarray
    test_event_indices: np.ndarray
    per_event_tier_cost: np.ndarray
    per_event_hit: np.ndarray
    per_window: dict[str, np.ndarray]
    final_resident_indicator: np.ndarray
    policy_metrics: dict[str, dict[str, Any]]
    exact_solver_diagnostics: tuple[dict[str, Any], ...]
    capacity_violations: int

    def __post_init__(self) -> None:
        for name in (
            "test_window_ids",
            "test_event_indices",
            "per_event_tier_cost",
            "per_event_hit",
            "final_resident_indicator",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        immutable: dict[str, np.ndarray] = {}
        for name, raw in self.per_window.items():
            value = np.array(raw, copy=True)
            value.setflags(write=False)
            immutable[name] = value
        object.__setattr__(self, "per_window", immutable)


@dataclass
class _WindowRow:
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    access_cost: float = 0.0
    promotion_cost: float = 0.0
    promotion_count: int = 0
    eviction_count: int = 0
    bytes_promoted: int = 0
    bytes_evicted: int = 0
    end_occupancy_bytes: int = 0
    end_resident_record_count: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "access_cost": self.access_cost,
            "promotion_cost": self.promotion_cost,
            "combined_cost": self.access_cost + self.promotion_cost,
            "promotion_count": self.promotion_count,
            "eviction_count": self.eviction_count,
            "bytes_promoted": self.bytes_promoted,
            "bytes_evicted": self.bytes_evicted,
            "end_occupancy_bytes": self.end_occupancy_bytes,
            "end_resident_record_count": self.end_resident_record_count,
        }


@dataclass
class _Runtime:
    state: StorageState
    lru_last_access: dict[int, int]
    lfu_frequency: dict[int, int]
    lfu_last_access: dict[int, int]
    wasted_promotion_count: int = 0
    wasted_promoted_bytes: int = 0


def run_policy_replay(
    observable_events: Sequence[ObservableEvent],
    observable_demand: np.ndarray,
    record_ids: Sequence[int],
    record_sizes: Sequence[int],
    predicted_record_demand: np.ndarray,
    prediction_available: np.ndarray,
    config: SimulationConfig,
    validation_start: int,
    test_start: int,
) -> PolicyReplayResult:
    """Warm every policy independently on validation and measure only test."""

    (
        events,
        demand,
        ids,
        sizes,
        predictive,
        available,
        events_by_window,
    ) = _validate_replay_inputs(
        observable_events,
        observable_demand,
        record_ids,
        record_sizes,
        predicted_record_demand,
        prediction_available,
        config,
        validation_start,
        test_start,
    )
    num_windows = demand.shape[0]
    test_windows = np.arange(test_start, num_windows, dtype=np.int64)
    test_events = tuple(event for event in events if event.window_id >= test_start)
    event_indices = np.asarray(
        [event.event_index for event in test_events], dtype=np.int64
    )
    record_size_map = dict(zip(ids, sizes, strict=True))

    event_cost_rows: list[np.ndarray] = []
    event_hit_rows: list[np.ndarray] = []
    window_rows_by_policy: list[list[dict[str, int | float]]] = []
    final_residency_rows: list[np.ndarray] = []
    metrics: dict[str, dict[str, Any]] = {}
    exact_diagnostics: list[dict[str, Any]] = []

    for policy_name in POLICY_NAMES:
        runtime = _Runtime(
            state=StorageState(record_size_map, config.fast_capacity_bytes),
            lru_last_access={},
            lfu_frequency={record_id: 0 for record_id in ids},
            lfu_last_access={},
        )
        policy_event_costs: list[float] = []
        policy_event_hits: list[bool] = []
        test_rows: list[dict[str, int | float]] = []
        for window_id in range(validation_start, num_windows):
            in_test = window_id >= test_start
            row = _WindowRow()
            if policy_name in BOUNDARY_POLICIES:
                forecast = _boundary_forecast(
                    policy_name, window_id, demand, predictive, available
                )
                selection = _boundary_selection(
                    policy_name,
                    forecast,
                    ids,
                    record_size_map,
                    runtime.state.resident,
                    config,
                )
                if policy_name == "oracle_exact":
                    exact_diagnostics.append(
                        {
                            "window_id": window_id,
                            "period": "test" if in_test else "validation",
                            "solver_status": selection.solver_status,
                            "objective_value": selection.objective_value,
                            "selected_record_count": len(
                                selection.target_record_ids
                            ),
                        }
                    )
                _apply_target(runtime, selection, row, in_test, config)

            for event in events_by_window[window_id]:
                hit = runtime.state.note_access(event.record_id)
                tier_cost = config.fast_read_cost if hit else config.slow_read_cost
                if in_test:
                    row.access_count += 1
                    row.hit_count += int(hit)
                    row.miss_count += int(not hit)
                    row.access_cost += tier_cost
                    policy_event_costs.append(tier_cost)
                    policy_event_hits.append(hit)
                if policy_name == "lru":
                    _handle_lru(runtime, event, hit, row, in_test, config)
                elif policy_name == "lfu":
                    _handle_lfu(runtime, event, hit, row, in_test, config)

            if in_test:
                row.end_occupancy_bytes = runtime.state.resident_bytes
                row.end_resident_record_count = len(runtime.state.resident)
                test_rows.append(row.to_dict())

        for episode in runtime.state.active_test_episodes():
            if episode.resident_accesses == 0:
                runtime.wasted_promotion_count += 1
                runtime.wasted_promoted_bytes += episode.size_bytes

        costs = np.asarray(policy_event_costs, dtype=np.float64)
        hits = np.asarray(policy_event_hits, dtype=np.bool_)
        if len(costs) != len(test_events):
            raise ReplayError(f"policy {policy_name} did not replay every test event")
        event_cost_rows.append(costs)
        event_hit_rows.append(hits)
        window_rows_by_policy.append(test_rows)
        final_residency_rows.append(
            np.asarray(
                [record_id in runtime.state.resident for record_id in ids],
                dtype=np.bool_,
            )
        )
        metrics[policy_name] = _policy_metrics(
            test_rows,
            costs,
            hits,
            runtime,
            config,
        )

    per_window = {
        name: np.asarray(
            [
                [rows[index][name] for index in range(len(test_windows))]
                for rows in window_rows_by_policy
            ]
        )
        for name in WINDOW_ARRAY_NAMES
    }
    return PolicyReplayResult(
        policy_names=POLICY_NAMES,
        test_window_ids=test_windows,
        test_event_indices=event_indices,
        per_event_tier_cost=np.stack(event_cost_rows),
        per_event_hit=np.stack(event_hit_rows),
        per_window=per_window,
        final_resident_indicator=np.stack(final_residency_rows),
        policy_metrics=metrics,
        exact_solver_diagnostics=tuple(exact_diagnostics),
        capacity_violations=0,
    )


def _boundary_forecast(
    policy_name: str,
    window_id: int,
    demand: np.ndarray,
    predictive: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    if policy_name == "recent_demand_greedy":
        return np.asarray(demand[window_id - 1], dtype=np.float64)
    if policy_name == "predictive_greedy":
        if not available[window_id]:
            raise ReplayError(
                f"predictive forecast is unavailable for window {window_id}"
            )
        return predictive[window_id]
    return np.asarray(demand[window_id], dtype=np.float64)


def _boundary_selection(
    policy_name: str,
    forecast: np.ndarray,
    record_ids: tuple[int, ...],
    record_sizes: Mapping[int, int],
    resident: set[int],
    config: SimulationConfig,
) -> PlacementSelection:
    benefits = record_benefits(
        forecast,
        record_ids,
        record_sizes,
        resident,
        config.fast_read_cost,
        config.slow_read_cost,
        config.promotion_cost_per_byte,
    )
    if policy_name == "oracle_exact":
        return exact_placement(benefits, record_sizes, config.fast_capacity_bytes)
    return greedy_placement(benefits, record_sizes, config.fast_capacity_bytes)


def _apply_target(
    runtime: _Runtime,
    selection: PlacementSelection,
    row: _WindowRow,
    in_test: bool,
    config: SimulationConfig,
) -> None:
    target = set(selection.target_record_ids)
    for record_id in sorted(runtime.state.resident - target):
        _record_eviction(runtime, runtime.state.evict(record_id), row, in_test)
    for record_id in sorted(target - runtime.state.resident):
        migration = runtime.state.promote(record_id, began_in_test=in_test)
        _record_promotion(migration, row, in_test, config)
    if runtime.state.resident != target:
        raise ReplayError("boundary controller did not establish exact target residency")


def _handle_lru(
    runtime: _Runtime,
    event: ObservableEvent,
    hit: bool,
    row: _WindowRow,
    in_test: bool,
    config: SimulationConfig,
) -> None:
    if hit:
        runtime.lru_last_access[event.record_id] = event.event_index
        return
    if not runtime.state.fits(event.record_id):
        return
    while not runtime.state.can_admit(event.record_id):
        victim = min(
            runtime.state.resident,
            key=lambda record_id: (
                runtime.lru_last_access[record_id], record_id
            ),
        )
        _record_eviction(runtime, runtime.state.evict(victim), row, in_test)
        runtime.lru_last_access.pop(victim)
    migration = runtime.state.promote(event.record_id, began_in_test=in_test)
    _record_promotion(migration, row, in_test, config)
    runtime.lru_last_access[event.record_id] = event.event_index


def _handle_lfu(
    runtime: _Runtime,
    event: ObservableEvent,
    hit: bool,
    row: _WindowRow,
    in_test: bool,
    config: SimulationConfig,
) -> None:
    runtime.lfu_frequency[event.record_id] += 1
    runtime.lfu_last_access[event.record_id] = event.event_index
    if hit or not runtime.state.fits(event.record_id):
        return
    while not runtime.state.can_admit(event.record_id):
        victim = min(
            runtime.state.resident,
            key=lambda record_id: (
                runtime.lfu_frequency[record_id],
                runtime.lfu_last_access[record_id],
                record_id,
            ),
        )
        _record_eviction(runtime, runtime.state.evict(victim), row, in_test)
        runtime.lfu_last_access.pop(victim)
    migration = runtime.state.promote(event.record_id, began_in_test=in_test)
    _record_promotion(migration, row, in_test, config)


def _record_promotion(
    migration: Migration,
    row: _WindowRow,
    in_test: bool,
    config: SimulationConfig,
) -> None:
    if not in_test:
        return
    row.promotion_count += 1
    row.bytes_promoted += migration.size_bytes
    row.promotion_cost += migration.size_bytes * config.promotion_cost_per_byte


def _record_eviction(
    runtime: _Runtime,
    migration: Migration,
    row: _WindowRow,
    in_test: bool,
) -> None:
    if not in_test:
        return
    row.eviction_count += 1
    row.bytes_evicted += migration.size_bytes
    episode = migration.episode
    if (
        episode is not None
        and episode.began_in_test
        and episode.resident_accesses == 0
    ):
        runtime.wasted_promotion_count += 1
        runtime.wasted_promoted_bytes += episode.size_bytes


def _policy_metrics(
    rows: list[dict[str, int | float]],
    costs: np.ndarray,
    hits: np.ndarray,
    runtime: _Runtime,
    config: SimulationConfig,
) -> dict[str, Any]:
    def total(name: str) -> int | float:
        values = [row[name] for row in rows]
        return math.fsum(values) if any(isinstance(value, float) for value in values) else sum(values)

    event_count = len(costs)
    hit_count = int(np.sum(hits))
    miss_count = event_count - hit_count
    access_cost = float(math.fsum(costs.tolist()))
    promotion_cost = float(total("promotion_cost"))
    occupancies = np.asarray(
        [row["end_occupancy_bytes"] for row in rows], dtype=np.float64
    )
    return {
        "event_count": event_count,
        "fast_hits": hit_count,
        "slow_reads": miss_count,
        "hit_rate": hit_count / event_count if event_count else None,
        "total_fast_read_cost": hit_count * config.fast_read_cost,
        "total_slow_read_cost": miss_count * config.slow_read_cost,
        "total_access_cost": access_cost,
        "promotion_count": int(total("promotion_count")),
        "bytes_promoted": int(total("bytes_promoted")),
        "total_promotion_cost": promotion_cost,
        "eviction_count": int(total("eviction_count")),
        "bytes_evicted": int(total("bytes_evicted")),
        "total_combined_cost": access_cost + promotion_cost,
        "mean_combined_cost_per_access": (
            (access_cost + promotion_cost) / event_count if event_count else None
        ),
        "wasted_promotion_count": runtime.wasted_promotion_count,
        "wasted_promoted_bytes": runtime.wasted_promoted_bytes,
        "wasted_promotion_cost": (
            runtime.wasted_promoted_bytes * config.promotion_cost_per_byte
        ),
        "access_tier_cost_percentiles": {
            "p50": float(np.percentile(costs, 50)) if event_count else None,
            "p95": float(np.percentile(costs, 95)) if event_count else None,
            "p99": float(np.percentile(costs, 99)) if event_count else None,
        },
        "occupancy": {
            "minimum_end_window_bytes": int(np.min(occupancies)),
            "maximum_end_window_bytes": int(np.max(occupancies)),
            "mean_end_window_bytes": float(np.mean(occupancies)),
            "mean_fraction": float(
                np.mean(occupancies / config.fast_capacity_bytes)
            ),
            "capacity_violations": 0,
        },
    }


def _validate_replay_inputs(
    observable_events: Sequence[ObservableEvent],
    observable_demand: np.ndarray,
    record_ids: Sequence[int],
    record_sizes: Sequence[int],
    predicted_record_demand: np.ndarray,
    prediction_available: np.ndarray,
    config: SimulationConfig,
    validation_start: int,
    test_start: int,
) -> tuple[
    tuple[ObservableEvent, ...],
    np.ndarray,
    tuple[int, ...],
    tuple[int, ...],
    np.ndarray,
    np.ndarray,
    tuple[tuple[ObservableEvent, ...], ...],
]:
    events = tuple(observable_events)
    demand = np.asarray(observable_demand)
    ids = tuple(int(value) for value in record_ids)
    raw_sizes = tuple(record_sizes)
    predictive = np.asarray(predicted_record_demand, dtype=np.float64)
    available = np.asarray(prediction_available, dtype=np.bool_)
    if demand.ndim != 2 or demand.shape[1] != len(ids):
        raise ReplayError("observable demand and record IDs are incompatible")
    if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
        raise ReplayError("record IDs must be unique and ascending")
    if len(raw_sizes) != len(ids):
        raise ReplayError("record sizes and IDs are incompatible")
    config.validate_record_sizes(raw_sizes)
    sizes = tuple(int(value) for value in raw_sizes)
    if predictive.shape != demand.shape or available.shape != (demand.shape[0],):
        raise ReplayError("predictive record demand has incompatible dimensions")
    if not 0 < validation_start < test_start < demand.shape[0]:
        raise ReplayError("validation and test boundaries are invalid")
    if validation_start == 0:
        raise ReplayError("recent-demand warm-up requires a preceding window")

    record_index = {record_id: index for index, record_id in enumerate(ids)}
    grouped: list[list[ObservableEvent]] = [[] for _ in range(demand.shape[0])]
    reconstructed = np.zeros_like(demand, dtype=np.int64)
    for expected_event_index, event in enumerate(events):
        if not isinstance(event, ObservableEvent):
            raise ReplayError("observable events must use ObservableEvent")
        if event.event_index != expected_event_index:
            raise ReplayError("observable event indices must be contiguous")
        if event.record_id not in record_index:
            raise ReplayError("observable event references an unknown record")
        if event.record_size_bytes != sizes[record_index[event.record_id]]:
            raise ReplayError("observable event record size is inconsistent")
        if not 0 <= event.window_id < demand.shape[0]:
            raise ReplayError("observable event window is invalid")
        grouped[event.window_id].append(event)
        reconstructed[event.window_id, record_index[event.record_id]] += 1
    if not np.array_equal(reconstructed, demand):
        raise ReplayError("observable events do not match the frozen demand matrix")
    return (
        events,
        demand,
        ids,
        sizes,
        predictive,
        available,
        tuple(tuple(group) for group in grouped),
    )
