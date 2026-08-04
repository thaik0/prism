"""Operation-level parity execution for the four Milestone 7 policy paths."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from prism.native.errors import NativeStoreError
from prism.native.payloads import GeneratedPayload
from prism.native.reference import ReferenceLedger, ReferenceStoreError
from prism.native.store import (
    EvictionResult,
    PromotionResult,
    ReadResult,
    RecordMetadata,
    ResidencySnapshot,
    StoreStats,
    TargetSetResult,
    TieredStore,
)
from prism.simulation import (
    POLICY_DISPLAY_NAMES,
    SimulationConfig,
    greedy_policy_target,
    select_lfu_victim,
    select_lru_victim,
    training_popularity_forecast,
)
from prism.workload.models import ObservableEvent


PARITY_POLICY_ORDER = (
    "training_popularity_static",
    "predictive_greedy",
    "lru",
    "lfu",
)
MISMATCH_CATEGORIES = (
    "operation_result",
    "error_code",
    "read_tier",
    "payload_size",
    "payload_digest",
    "residency",
    "resident_bytes",
    "counter",
    "capacity_invariant",
    "unexpected_exception",
)


@dataclass(frozen=True, slots=True)
class PolicyParityInputs:
    observable_events: tuple[ObservableEvent, ...]
    observable_demand: np.ndarray
    record_ids: tuple[int, ...]
    record_sizes: tuple[int, ...]
    predicted_record_demand: np.ndarray
    prediction_available: np.ndarray
    config: SimulationConfig
    validation_start: int
    test_start: int
    evaluation_end: int
    forecast_horizon_windows: int = 1

    def __post_init__(self) -> None:
        demand = np.asarray(self.observable_demand)
        predicted = np.asarray(self.predicted_record_demand, dtype=np.float64)
        available = np.asarray(self.prediction_available, dtype=np.bool_)
        if demand.ndim != 2 or demand.shape[1] != len(self.record_ids):
            raise ValueError("observable demand and record IDs are incompatible")
        if self.record_ids != tuple(sorted(self.record_ids)) or len(
            set(self.record_ids)
        ) != len(self.record_ids):
            raise ValueError("record IDs must be unique and ascending")
        if len(self.record_sizes) != len(self.record_ids):
            raise ValueError("record sizes and IDs are incompatible")
        self.config.validate_record_sizes(self.record_sizes)
        if predicted.shape != demand.shape or available.shape != (demand.shape[0],):
            raise ValueError("prediction matrices are incompatible")
        if not (
            0 < self.validation_start < self.test_start < self.evaluation_end
            <= demand.shape[0]
        ):
            raise ValueError("parity chronological boundaries are invalid")
        if (
            isinstance(self.forecast_horizon_windows, bool)
            or not isinstance(self.forecast_horizon_windows, int)
            or self.forecast_horizon_windows <= 0
        ):
            raise ValueError("forecast horizon must be a positive integer")
        record_index = {
            record_id: index for index, record_id in enumerate(self.record_ids)
        }
        reconstructed = np.zeros_like(demand, dtype=np.int64)
        for expected_index, event in enumerate(self.observable_events):
            if not isinstance(event, ObservableEvent):
                raise ValueError("observable events must use ObservableEvent")
            if event.event_index != expected_index:
                raise ValueError("observable event indices must be contiguous")
            if event.record_id not in record_index:
                raise ValueError("observable event references an unknown record")
            index = record_index[event.record_id]
            if event.record_size_bytes != self.record_sizes[index]:
                raise ValueError("observable event size is inconsistent")
            if not 0 <= event.window_id < demand.shape[0]:
                raise ValueError("observable event window is invalid")
            reconstructed[event.window_id, index] += 1
        if not np.array_equal(reconstructed, demand):
            raise ValueError("observable events do not match observable demand")
        for name, value in (
            ("observable_demand", demand),
            ("predicted_record_demand", predicted),
            ("prediction_available", available),
        ):
            immutable = np.array(value, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)


@dataclass(frozen=True, slots=True)
class ParityExecution:
    operations: tuple[dict[str, Any], ...]
    policy_summaries: tuple[tuple[str, dict[str, Any]], ...]
    report: dict[str, Any]


class OperationRecorder:
    def __init__(self) -> None:
        self.operations: list[dict[str, Any]] = []
        self._next_sequence = 0

    def append(self, operation: dict[str, Any]) -> dict[str, Any]:
        operation["operation_sequence"] = self._next_sequence
        self._next_sequence += 1
        self.operations.append(operation)
        return operation


@dataclass(frozen=True, slots=True)
class _Outcome:
    success: bool
    result: object | None
    error: BaseException | None
    unexpected: bool = False


class ParitySession:
    """Compare one independent policy's expected and native state."""

    def __init__(
        self,
        policy_id: str,
        reference: ReferenceLedger,
        native_store: TieredStore,
        recorder: OperationRecorder,
    ) -> None:
        if policy_id not in PARITY_POLICY_ORDER:
            raise ValueError(f"unsupported parity policy: {policy_id}")
        self.policy_id = policy_id
        self.reference = reference
        self.native_store = native_store
        self.recorder = recorder
        self.invalidated = False
        self.invalidated_at: int | None = None
        self.capacity_violations = 0
        self.unexpected_exceptions = 0

    def execute(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        *,
        phase: str,
        window_id: int | None = None,
        event_index: int | None = None,
    ) -> dict[str, Any]:
        candidate = self.reference.clone()
        expected = _capture(candidate, operation, arguments)
        if self.invalidated:
            if not expected.unexpected:
                self.reference = candidate
            record = self._base_record(
                operation, arguments, phase, window_id, event_index
            )
            record.update(
                {
                    "python_expected_success": expected.success,
                    "native_success": None,
                    "python_expected_error": _error_dict(expected.error),
                    "native_error": None,
                    "python_expected_result": _result_dict(expected.result),
                    "native_result": None,
                    "python_expected_snapshot": _snapshot_dict(candidate.snapshot()),
                    "native_snapshot": _snapshot_dict(self.native_store.snapshot()),
                    "python_expected_stats": candidate.stats().to_dict(),
                    "native_stats": self.native_store.stats().to_dict(),
                    "parity_status": "not_compared_due_to_prior_divergence",
                    "mismatch_details": [],
                }
            )
            return self.recorder.append(record)

        native_outcome = _capture(self.native_store, operation, arguments)
        expected_snapshot = candidate.snapshot()
        expected_stats = candidate.stats()
        native_snapshot = self.native_store.snapshot()
        native_stats = self.native_store.stats()
        mismatches = _compare_operation(
            expected,
            native_outcome,
            expected_snapshot,
            native_snapshot,
            expected_stats,
            native_stats,
        )
        if expected_snapshot.resident_bytes > expected_snapshot.capacity_bytes or (
            native_snapshot.resident_bytes > native_snapshot.capacity_bytes
        ):
            self.capacity_violations += 1
        if expected.unexpected or native_outcome.unexpected:
            self.unexpected_exceptions += 1
        status = "match" if not mismatches else "mismatch"
        if not expected.unexpected:
            self.reference = candidate

        record = self._base_record(
            operation, arguments, phase, window_id, event_index
        )
        record.update(
            {
                "python_expected_success": expected.success,
                "native_success": native_outcome.success,
                "python_expected_error": _error_dict(expected.error),
                "native_error": _error_dict(native_outcome.error),
                "python_expected_result": _result_dict(expected.result),
                "native_result": _result_dict(native_outcome.result),
                "python_expected_snapshot": _snapshot_dict(expected_snapshot),
                "native_snapshot": _snapshot_dict(native_snapshot),
                "python_expected_stats": expected_stats.to_dict(),
                "native_stats": native_stats.to_dict(),
                "parity_status": status,
                "mismatch_details": mismatches,
            }
        )
        appended = self.recorder.append(record)
        if mismatches:
            self.invalidated = True
            self.invalidated_at = appended["operation_sequence"]
        return appended

    def summary(self) -> dict[str, Any]:
        operations = [
            row
            for row in self.recorder.operations
            if row["policy_id"] == self.policy_id
        ]
        native_stats = self.native_store.stats()
        expected_stats = self.reference.stats()
        native_snapshot = self.native_store.snapshot()
        expected_snapshot = self.reference.snapshot()
        reads = [row for row in operations if row["operation_type"] == "read"]
        mismatches = [row for row in operations if row["parity_status"] == "mismatch"]
        compared = [
            row
            for row in operations
            if row["parity_status"] in {"match", "mismatch"}
        ]
        return {
            "policy_id": self.policy_id,
            "policy_display_name": POLICY_DISPLAY_NAMES[self.policy_id],
            "operation_count": len(operations),
            "compared_operation_count": len(compared),
            "passed_operation_count": sum(
                row["parity_status"] == "match" for row in operations
            ),
            "invalidated_operation_count": sum(
                row["parity_status"]
                == "not_compared_due_to_prior_divergence"
                for row in operations
            ),
            "read_count": len(reads),
            "fast_read_count": native_stats.successful_fast_reads,
            "slow_read_count": native_stats.successful_slow_reads,
            "promotion_count": native_stats.committed_promotions,
            "promotion_bytes": native_stats.committed_promotion_bytes,
            "eviction_count": native_stats.committed_evictions,
            "eviction_bytes": native_stats.committed_eviction_bytes,
            "target_set_calls": native_stats.target_set_calls,
            "mismatch_count": len(mismatches),
            "invalidated": self.invalidated,
            "invalidated_at_operation": self.invalidated_at,
            "unexpected_exception_count": self.unexpected_exceptions,
            "capacity_violation_count": self.capacity_violations,
            "final_expected_resident_record_ids": list(
                expected_snapshot.resident_record_ids
            ),
            "final_native_resident_record_ids": list(
                native_snapshot.resident_record_ids
            ),
            "final_expected_resident_bytes": expected_snapshot.resident_bytes,
            "final_native_resident_bytes": native_snapshot.resident_bytes,
            "final_expected_counters": expected_stats.to_dict(),
            "final_native_counters": native_stats.to_dict(),
            "final_state_equality": expected_snapshot == native_snapshot,
            "final_counter_equality": (
                expected_stats.to_dict() == native_stats.to_dict()
            ),
            "parity_passed": (
                not mismatches
                and not self.invalidated
                and self.unexpected_exceptions == 0
                and self.capacity_violations == 0
                and expected_snapshot == native_snapshot
                and expected_stats.to_dict() == native_stats.to_dict()
            ),
        }

    def _base_record(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        phase: str,
        window_id: int | None,
        event_index: int | None,
    ) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_display_name": POLICY_DISPLAY_NAMES[self.policy_id],
            "phase": phase,
            "window_id": window_id,
            "event_index": event_index,
            "operation_type": operation,
            "operation_arguments": _json_value(dict(arguments)),
        }


def _capture(
    target: object, operation: str, arguments: Mapping[str, Any]
) -> _Outcome:
    try:
        if operation in {"read", "promote", "evict", "record_metadata"}:
            result = getattr(target, operation)(arguments["record_id"])
        elif operation == "apply_target_set":
            result = getattr(target, operation)(arguments["record_ids"])
        elif operation == "snapshot":
            result = getattr(target, operation)()
        else:
            raise ValueError(f"unsupported parity operation: {operation}")
        return _Outcome(True, result, None)
    except (ReferenceStoreError, NativeStoreError) as error:
        return _Outcome(False, None, error)
    except Exception as error:  # recorded explicitly and invalidates only this policy
        return _Outcome(False, None, error, unexpected=True)


def _compare_operation(
    expected: _Outcome,
    native: _Outcome,
    expected_snapshot: ResidencySnapshot,
    native_snapshot: ResidencySnapshot,
    expected_stats: StoreStats,
    native_stats: StoreStats,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    def mismatch(category: str, field: str, left: Any, right: Any) -> None:
        mismatches.append(
            {
                "category": category,
                "field": field,
                "python_expected": _json_value(left),
                "native": _json_value(right),
            }
        )

    if expected.unexpected or native.unexpected:
        mismatch(
            "unexpected_exception",
            "exception_type",
            type(expected.error).__name__ if expected.error else None,
            type(native.error).__name__ if native.error else None,
        )
    if expected.success != native.success:
        mismatch("operation_result", "success", expected.success, native.success)
    elif not expected.success:
        expected_error = _error_dict(expected.error)
        native_error = _error_dict(native.error)
        if expected_error and native_error and (
            expected_error["code"] != native_error["code"]
        ):
            mismatch(
                "error_code",
                "code",
                expected_error["code"],
                native_error["code"],
            )
        for field in ("record_id", "offset"):
            if expected_error and native_error and (
                expected_error[field] != native_error[field]
            ):
                mismatch(
                    "operation_result",
                    f"error.{field}",
                    expected_error[field],
                    native_error[field],
                )
    else:
        expected_result = _result_dict(expected.result)
        native_result = _result_dict(native.result)
        if isinstance(expected.result, ReadResult) and isinstance(
            native.result, ReadResult
        ):
            for field, category in (
                ("tier", "read_tier"),
                ("byte_count", "payload_size"),
                ("payload_sha256", "payload_digest"),
            ):
                if expected_result[field] != native_result[field]:
                    mismatch(
                        category,
                        field,
                        expected_result[field],
                        native_result[field],
                    )
        elif expected_result != native_result:
            mismatch(
                "operation_result", "result", expected_result, native_result
            )
    if expected_snapshot.resident_record_ids != native_snapshot.resident_record_ids:
        mismatch(
            "residency",
            "resident_record_ids",
            expected_snapshot.resident_record_ids,
            native_snapshot.resident_record_ids,
        )
    if expected_snapshot.resident_bytes != native_snapshot.resident_bytes:
        mismatch(
            "resident_bytes",
            "resident_bytes",
            expected_snapshot.resident_bytes,
            native_snapshot.resident_bytes,
        )
    if expected_snapshot.capacity_bytes != native_snapshot.capacity_bytes:
        mismatch(
            "capacity_invariant",
            "capacity_bytes",
            expected_snapshot.capacity_bytes,
            native_snapshot.capacity_bytes,
        )
    expected_counter_values = expected_stats.to_dict()
    native_counter_values = native_stats.to_dict()
    if expected_counter_values != native_counter_values:
        for key in sorted(expected_counter_values):
            if expected_counter_values[key] != native_counter_values[key]:
                mismatch(
                    "counter",
                    key,
                    expected_counter_values[key],
                    native_counter_values[key],
                )
    if expected_snapshot.resident_bytes > expected_snapshot.capacity_bytes or (
        native_snapshot.resident_bytes > native_snapshot.capacity_bytes
    ):
        mismatch(
            "capacity_invariant",
            "resident_bytes_lte_capacity",
            expected_snapshot.resident_bytes <= expected_snapshot.capacity_bytes,
            native_snapshot.resident_bytes <= native_snapshot.capacity_bytes,
        )
    return mismatches


def _error_dict(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    path = getattr(error, "path", None)
    return {
        "code": getattr(error, "code", type(error).__name__),
        "message": getattr(error, "message", str(error)),
        "record_id": getattr(error, "record_id", None),
        "offset": getattr(error, "offset", None),
        "path": Path(path).name if path else None,
    }


def _snapshot_dict(snapshot: ResidencySnapshot) -> dict[str, Any]:
    return {
        "resident_record_ids": list(snapshot.resident_record_ids),
        "resident_bytes": snapshot.resident_bytes,
        "capacity_bytes": snapshot.capacity_bytes,
    }


def _result_dict(result: object | None) -> dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, ReadResult):
        return {
            "tier": result.tier,
            "byte_count": result.byte_count,
            "payload_sha256": hashlib.sha256(result.payload).hexdigest(),
        }
    if isinstance(result, (PromotionResult, EvictionResult)):
        return {
            "moved": result.moved,
            "record_id": result.record_id,
            "bytes_moved": result.bytes_moved,
        }
    if isinstance(result, ResidencySnapshot):
        return _snapshot_dict(result)
    if isinstance(result, TargetSetResult):
        return {
            "incoming_record_ids": list(result.incoming_record_ids),
            "outgoing_record_ids": list(result.outgoing_record_ids),
            "promotion_count": result.promotion_count,
            "promotion_bytes": result.promotion_bytes,
            "eviction_count": result.eviction_count,
            "eviction_bytes": result.eviction_bytes,
            "target_changed": result.target_changed,
            "residency": _snapshot_dict(result.residency),
        }
    if isinstance(result, RecordMetadata):
        return {
            "record_id": result.record_id,
            "byte_offset": result.byte_offset,
            "byte_length": result.byte_length,
            "crc32": result.crc32,
        }
    raise TypeError(f"unsupported parity result type: {type(result).__name__}")


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_four_policy_parity(
    inputs: PolicyParityInputs,
    payloads: Sequence[GeneratedPayload],
    store_dir: str | Path,
    *,
    source_information: Mapping[str, Any] | None = None,
    native_store_factory: Callable[[str | Path, int], TieredStore] = TieredStore.open,
) -> ParityExecution:
    """Replay validation and test with independent state for exactly four policies."""

    payload_map = {record.record_id: record.payload for record in payloads}
    if tuple(sorted(payload_map)) != inputs.record_ids:
        raise ValueError("payload IDs do not match parity record IDs")
    if tuple(len(payload_map[item]) for item in inputs.record_ids) != inputs.record_sizes:
        raise ValueError("payload sizes do not match parity record sizes")
    grouped: list[list[ObservableEvent]] = [
        [] for _ in range(inputs.observable_demand.shape[0])
    ]
    for event in inputs.observable_events:
        grouped[event.window_id].append(event)
    record_sizes = dict(zip(inputs.record_ids, inputs.record_sizes, strict=True))
    recorder = OperationRecorder()
    summaries = []

    for policy_id in PARITY_POLICY_ORDER:
        reference = ReferenceLedger(payload_map, inputs.config.fast_capacity_bytes)
        native_store = native_store_factory(
            store_dir, inputs.config.fast_capacity_bytes
        )
        session = ParitySession(policy_id, reference, native_store, recorder)
        lru_last_access: dict[int, int] = {}
        lfu_frequency = {record_id: 0 for record_id in inputs.record_ids}
        lfu_last_access: dict[int, int] = {}

        if policy_id == "training_popularity_static":
            forecast = training_popularity_forecast(
                inputs.observable_demand,
                inputs.validation_start,
                inputs.forecast_horizon_windows,
            )
            selection = greedy_policy_target(
                forecast,
                inputs.record_ids,
                record_sizes,
                set(),
                inputs.config,
            )
            session.execute(
                "apply_target_set",
                {"record_ids": list(selection.target_record_ids)},
                phase="validation",
                window_id=inputs.validation_start,
            )

        for window_id in range(inputs.validation_start, inputs.evaluation_end):
            phase = "validation" if window_id < inputs.test_start else "test"
            if policy_id == "predictive_greedy":
                if not inputs.prediction_available[window_id]:
                    raise ValueError(
                        f"predictive forecast is unavailable for window {window_id}"
                    )
                selection = greedy_policy_target(
                    inputs.predicted_record_demand[window_id],
                    inputs.record_ids,
                    record_sizes,
                    set(session.reference.resident_record_ids),
                    inputs.config,
                )
                session.execute(
                    "apply_target_set",
                    {"record_ids": list(selection.target_record_ids)},
                    phase=phase,
                    window_id=window_id,
                )

            for event in grouped[window_id]:
                if policy_id == "lfu":
                    lfu_frequency[event.record_id] += 1
                    lfu_last_access[event.record_id] = event.event_index
                hit = event.record_id in session.reference.resident_record_ids
                session.execute(
                    "read",
                    {"record_id": event.record_id},
                    phase=phase,
                    window_id=window_id,
                    event_index=event.event_index,
                )
                if policy_id == "lru":
                    if hit:
                        lru_last_access[event.record_id] = event.event_index
                    elif session.reference.fits(event.record_id):
                        while not session.reference.can_admit(event.record_id):
                            victim = select_lru_victim(
                                set(session.reference.resident_record_ids),
                                lru_last_access,
                            )
                            session.execute(
                                "evict",
                                {"record_id": victim},
                                phase=phase,
                                window_id=window_id,
                                event_index=event.event_index,
                            )
                            lru_last_access.pop(victim)
                        session.execute(
                            "promote",
                            {"record_id": event.record_id},
                            phase=phase,
                            window_id=window_id,
                            event_index=event.event_index,
                        )
                        lru_last_access[event.record_id] = event.event_index
                elif policy_id == "lfu" and not hit and session.reference.fits(
                    event.record_id
                ):
                    while not session.reference.can_admit(event.record_id):
                        victim = select_lfu_victim(
                            set(session.reference.resident_record_ids),
                            lfu_frequency,
                            lfu_last_access,
                        )
                        session.execute(
                            "evict",
                            {"record_id": victim},
                            phase=phase,
                            window_id=window_id,
                            event_index=event.event_index,
                        )
                        lfu_last_access.pop(victim)
                    session.execute(
                        "promote",
                        {"record_id": event.record_id},
                        phase=phase,
                        window_id=window_id,
                        event_index=event.event_index,
                    )

            session.execute(
                "snapshot",
                {"reason": "end_window"},
                phase=phase,
                window_id=window_id,
            )
        summaries.append((policy_id, session.summary()))

    report = build_parity_report(
        recorder.operations,
        dict(summaries),
        source_information=source_information or {},
    )
    return ParityExecution(
        tuple(recorder.operations), tuple(summaries), report
    )


def build_parity_report(
    operations: Sequence[Mapping[str, Any]],
    policy_summaries: Mapping[str, Mapping[str, Any]],
    *,
    source_information: Mapping[str, Any],
) -> dict[str, Any]:
    mismatch_counts = {category: 0 for category in MISMATCH_CATEGORIES}
    for operation in operations:
        for mismatch in operation["mismatch_details"]:
            mismatch_counts[mismatch["category"]] += 1
    phases = {}
    for phase in ("validation", "test"):
        rows = [row for row in operations if row["phase"] == phase]
        mismatch_operation_count = sum(
            row["parity_status"] == "mismatch" for row in rows
        )
        not_compared_operation_count = sum(
            row["parity_status"] == "not_compared_due_to_prior_divergence"
            for row in rows
        )
        phases[phase] = {
            "operation_count": len(rows),
            "compared_operation_count": sum(
                row["parity_status"] in {"match", "mismatch"} for row in rows
            ),
            "passed_operation_count": sum(
                row["parity_status"] == "match" for row in rows
            ),
            "mismatch_operation_count": mismatch_operation_count,
            "not_compared_operation_count": not_compared_operation_count,
            "parity_passed": (
                mismatch_operation_count == 0
                and not_compared_operation_count == 0
            ),
        }
    invalidated_count = sum(
        bool(summary["invalidated"]) for summary in policy_summaries.values()
    )
    unexpected_count = sum(
        int(summary["unexpected_exception_count"])
        for summary in policy_summaries.values()
    )
    capacity_violations = sum(
        int(summary["capacity_violation_count"])
        for summary in policy_summaries.values()
    )
    total_mismatches = sum(mismatch_counts.values())
    all_policies_present = tuple(policy_summaries) == PARITY_POLICY_ORDER
    policy_parity_passed = all(
        bool(summary["parity_passed"])
        for summary in policy_summaries.values()
    )
    state_parity_passed = all(
        bool(summary["final_state_equality"])
        for summary in policy_summaries.values()
    )
    counter_parity_passed = all(
        bool(summary["final_counter_equality"])
        for summary in policy_summaries.values()
    )
    native_store_verified = bool(
        source_information.get("native_store_verification_passed", True)
    )
    payloads_verified = bool(
        source_information.get("payload_verification_passed", True)
    )
    return {
        "schema_version": 1,
        "source_information": _json_value(dict(source_information)),
        "policy_order": list(PARITY_POLICY_ORDER),
        "policy_summaries": {
            policy_id: dict(policy_summaries[policy_id])
            for policy_id in PARITY_POLICY_ORDER
        },
        "phase_summaries": phases,
        "overall_gates": {
            "mismatch_counts_by_category": mismatch_counts,
            "total_mismatch_count": total_mismatches,
            "invalidated_policy_count": invalidated_count,
            "unexpected_exception_count": unexpected_count,
            "capacity_violation_count": capacity_violations,
            "all_four_policies_present": all_policies_present,
            "native_store_verification_passed": native_store_verified,
            "payload_verification_passed": payloads_verified,
            "operation_parity_passed": total_mismatches == 0,
            "state_parity_passed": state_parity_passed,
            "counter_parity_passed": counter_parity_passed,
            "capacity_invariants_passed": capacity_violations == 0,
            "deterministic_artifact_verification_status": (
                "pending_external_repeat"
            ),
            "overall_parity_passed": (
                total_mismatches == 0
                and invalidated_count == 0
                and unexpected_count == 0
                and capacity_violations == 0
                and all_policies_present
                and policy_parity_passed
                and state_parity_passed
                and counter_parity_passed
                and native_store_verified
                and payloads_verified
            ),
        },
        "limitations": [
            "Execution is synchronous and this report has no wall-clock timing gate.",
            "Native payloads are intentionally copied into Python bytes for verification.",
            "The Python GIL remains held across native calls.",
            "C++ executes storage operations only; Python retains all policy decisions.",
            "Parity does not establish new predictive-actionability evidence.",
        ],
    }


def write_parity_artifacts(
    execution: ParityExecution, output_dir: str | Path
) -> None:
    root = Path(output_dir)
    operations_path = root / "parity_operations.jsonl"
    operations_path.write_text(
        "".join(
            json.dumps(
                operation,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for operation in execution.operations
        ),
        encoding="utf-8",
    )
    (root / "parity_report.json").write_text(
        json.dumps(
            execution.report, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
