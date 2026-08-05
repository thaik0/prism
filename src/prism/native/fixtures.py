"""Hand-checkable deterministic Milestone 7 parity fixture."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from prism.native.parity import (
    ParityExecution,
    PolicyParityInputs,
    run_four_policy_parity,
    write_parity_artifacts,
)
from prism.native.payloads import (
    NativeStoreArtifacts,
    _build_verified_payload_store,
    generate_payloads,
)
from prism.simulation import SimulationConfig
from prism.workload.models import ObservableEvent


@dataclass(frozen=True, slots=True)
class ForcedFixtureResult:
    store_artifacts: NativeStoreArtifacts
    parity: ParityExecution


def _fixture_definition() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workload_seed": 7007,
        "record_sizes": [2, 3, 4, 6, 11],
        "fast_capacity_bytes": 9,
        "fast_read_cost": 1.0,
        "slow_read_cost": 10.0,
        "promotion_cost_per_byte": 0.0,
        "validation_start": 2,
        "test_start": 4,
        "evaluation_end": 6,
        "events_by_window": [
            [0, 1],
            [1, 2],
            [0, 1, 2, 0],
            [3, 4],
            [1, 3, 0],
            [2, 3],
        ],
        "predictive_forecasts": {
            "2": [10.0, 10.0, 10.0, 0.0, 0.0],
            "3": [0.0, 10.0, 0.0, 10.0, 0.0],
            "4": [10.0, 0.0, 0.0, 10.0, 0.0],
            "5": [10.0, 0.0, 0.0, 10.0, 0.0],
        },
    }


def run_forced_fixture(output_dir: str | Path) -> ForcedFixtureResult:
    """Build one exact four-artifact fixture root and certify four policies."""

    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError("fixture output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    definition = _fixture_definition()
    definition_bytes = json.dumps(
        definition, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    sizes = tuple(int(value) for value in definition["record_sizes"])
    seed = int(definition["workload_seed"])
    payloads = generate_payloads(workload_seed=seed, record_sizes=sizes)
    capacity = int(definition["fast_capacity_bytes"])
    store_artifacts = _build_verified_payload_store(
        payloads,
        (("fixture_definition", hashlib.sha256(definition_bytes).hexdigest()),),
        seed,
        destination / "native_store",
        capacity,
        manifest_path=destination / "native_store_manifest.json",
    )

    rows = definition["events_by_window"]
    events = []
    demand = np.zeros((len(rows), len(sizes)), dtype=np.int64)
    event_index = 0
    for window_id, record_ids in enumerate(rows):
        for record_id in record_ids:
            demand[window_id, record_id] += 1
            events.append(
                ObservableEvent(
                    event_index=event_index,
                    window_id=window_id,
                    record_id=record_id,
                    record_size_bytes=sizes[record_id],
                    user_id=0,
                    session_id=window_id,
                    request_id=event_index,
                    request_type="fixture",
                    operation_type="read",
                )
            )
            event_index += 1
    predicted = np.zeros_like(demand, dtype=np.float64)
    available = np.zeros(len(rows), dtype=np.bool_)
    for raw_window, forecast in definition["predictive_forecasts"].items():
        window_id = int(raw_window)
        predicted[window_id] = forecast
        available[window_id] = True
    config = SimulationConfig(
        fast_capacity_bytes=capacity,
        fast_read_cost=float(definition["fast_read_cost"]),
        slow_read_cost=float(definition["slow_read_cost"]),
        promotion_cost_per_byte=float(definition["promotion_cost_per_byte"]),
    )
    inputs = PolicyParityInputs(
        observable_events=tuple(events),
        observable_demand=demand,
        record_ids=tuple(range(len(sizes))),
        record_sizes=sizes,
        predicted_record_demand=predicted,
        prediction_available=available,
        config=config,
        validation_start=int(definition["validation_start"]),
        test_start=int(definition["test_start"]),
        evaluation_end=int(definition["evaluation_end"]),
    )
    parity = run_four_policy_parity(
        inputs,
        payloads,
        destination / "native_store",
        source_information={
            "input_kind": "forced_fixture",
            "fixture_definition_sha256": hashlib.sha256(definition_bytes).hexdigest(),
            "workload_seed": seed,
            "capacity_bytes": capacity,
            "payload_schema_version": 1,
            "native_store_verification_passed": True,
            "payload_verification_passed": (
                store_artifacts.manifest.payload_verification_passed
            ),
            "validation_start": inputs.validation_start,
            "test_start": inputs.test_start,
            "evaluation_end": inputs.evaluation_end,
        },
    )
    write_parity_artifacts(parity, destination)
    actual = {path.name for path in destination.iterdir()}
    expected = {
        "native_store",
        "native_store_manifest.json",
        "parity_operations.jsonl",
        "parity_report.json",
    }
    if actual != expected:
        raise RuntimeError("fixture output root violated the four-entry contract")
    return ForcedFixtureResult(store_artifacts, parity)
