"""Exact workload and trace-specific simulation configuration materialization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
from typing import Sequence

from prism.experiments.config import (
    ActionabilityManifest,
    ExperimentManifest,
    RegimeDefinition,
    VariantDefinition,
)
from prism.simulation import SimulationConfig
from prism.workload import WorkloadConfig


@dataclass(frozen=True)
class MaterializedRun:
    experiment_id: str
    variant: VariantDefinition
    seed: int
    workload_config: WorkloadConfig
    simulation_config: SimulationConfig


def resolve_workload_config(
    manifest: ExperimentManifest, variant: VariantDefinition, seed: int
) -> WorkloadConfig:
    """Apply only the variant overrides and explicit experiment seed."""

    if seed not in manifest.seeds:
        raise ValueError(f"seed is not frozen in the manifest: {seed}")
    base = WorkloadConfig.from_json(manifest.base_workload_config)
    raw = base.to_dict()
    raw.update(variant.workload_overrides)
    raw["seed"] = seed
    resolved = WorkloadConfig.from_dict(raw)
    expected_changed = set(variant.workload_overrides) | {"seed"}
    base_raw = base.to_dict()
    changed = {key for key in raw if raw[key] != base_raw[key]}
    if changed != expected_changed - ({"seed"} if seed == base.seed else set()):
        raise ValueError("workload materialization changed unintended fields")
    return resolved


def resolve_simulation_config(
    variant: VariantDefinition,
    record_sizes: Sequence[int],
    *,
    fast_read_cost: float = 1.0,
    slow_read_cost: float = 10.0,
) -> SimulationConfig:
    """Resolve byte capacity and median-size promotion cost from one trace."""

    if not record_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in record_sizes
    ):
        raise ValueError("record_sizes must contain positive integers")
    total = sum(record_sizes)
    capacity = math.floor(variant.capacity_fraction * total)
    if capacity < min(record_sizes):
        raise ValueError("resolved capacity cannot hold any source record")
    median = float(statistics.median(record_sizes))
    promotion = (
        variant.promotion_saved_read_equivalents
        * (slow_read_cost - fast_read_cost)
        / median
    )
    return SimulationConfig(capacity, fast_read_cost, slow_read_cost, promotion)


def materialize_run(
    manifest: ExperimentManifest,
    variant_id: str,
    seed: int,
    record_sizes: Sequence[int],
) -> MaterializedRun:
    variant = manifest.variant(variant_id)
    workload = resolve_workload_config(manifest, variant, seed)
    simulation = resolve_simulation_config(variant, record_sizes)
    return MaterializedRun(
        f"{variant.id}__seed_{seed}", variant, seed, workload, simulation
    )


def resolve_actionability_workload_config(
    manifest: ActionabilityManifest,
    regime: RegimeDefinition,
    seed: int,
) -> WorkloadConfig:
    """Apply only the three frozen regime fields and the explicit seed."""

    if seed not in manifest.seeds:
        raise ValueError(f"seed is not frozen in the manifest: {seed}")
    base = WorkloadConfig.from_json(manifest.base_workload_config)
    raw = base.to_resolved_dict()
    raw.update(regime.workload_overrides)
    raw["seed"] = seed
    resolved = WorkloadConfig.from_dict(raw)
    base_resolved = base.to_resolved_dict()
    changed = {
        key for key, value in resolved.to_resolved_dict().items()
        if value != base_resolved[key]
    }
    expected = {
        key for key, value in regime.workload_overrides.items()
        if value != base_resolved[key]
    }
    if seed != base.seed:
        expected.add("seed")
    if changed != expected:
        raise ValueError("actionability regime changed unintended workload fields")
    return resolved


def resolve_actionability_simulation_config(
    record_sizes: Sequence[int],
    *,
    fast_read_cost: float = 1.0,
    slow_read_cost: float = 10.0,
) -> SimulationConfig:
    """Resolve the accepted 25% / two-saved-read storage configuration."""

    baseline = VariantDefinition("baseline", "baseline", {}, 0.25, 2.0)
    return resolve_simulation_config(
        baseline,
        record_sizes,
        fast_read_cost=fast_read_cost,
        slow_read_cost=slow_read_cost,
    )
