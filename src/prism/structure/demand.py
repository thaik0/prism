"""Observable-only raw demand-window construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prism.workload.config import WorkloadConfig, WorkloadConfigError
from prism.workload.models import OBSERVABLE_EVENT_FIELDS


SUPPORTED_SOURCE_SCHEMA_VERSION = 1


class DemandMatrixError(ValueError):
    """Raised when observable source artifacts cannot define a demand matrix."""


@dataclass(frozen=True)
class DemandMatrix:
    """Dense raw access counts with deterministic row and column identifiers."""

    X: np.ndarray
    window_ids: np.ndarray
    record_ids: np.ndarray

    def __post_init__(self) -> None:
        raw_X = np.asarray(self.X)
        raw_window_ids = np.asarray(self.window_ids)
        raw_record_ids = np.asarray(self.record_ids)
        if not np.issubdtype(raw_X.dtype, np.integer):
            raise DemandMatrixError("X must use an integer count dtype")
        if not np.issubdtype(raw_window_ids.dtype, np.integer):
            raise DemandMatrixError("window_ids must use an integer dtype")
        if not np.issubdtype(raw_record_ids.dtype, np.integer):
            raise DemandMatrixError("record_ids must use an integer dtype")
        X = np.array(raw_X, dtype=np.int64, copy=True)
        window_ids = np.array(raw_window_ids, dtype=np.int64, copy=True)
        record_ids = np.array(raw_record_ids, dtype=np.int64, copy=True)
        if X.ndim != 2:
            raise DemandMatrixError("X must be a two-dimensional matrix")
        if window_ids.ndim != 1 or record_ids.ndim != 1:
            raise DemandMatrixError("window_ids and record_ids must be vectors")
        if X.shape != (len(window_ids), len(record_ids)):
            raise DemandMatrixError("X shape must match window_ids and record_ids")
        if np.any(X < 0):
            raise DemandMatrixError("X must contain nonnegative counts")
        if len(window_ids) and not np.all(window_ids[:-1] < window_ids[1:]):
            raise DemandMatrixError("window_ids must be strictly ascending")
        if len(record_ids) and not np.all(record_ids[:-1] < record_ids[1:]):
            raise DemandMatrixError("record_ids must be strictly ascending")
        X.setflags(write=False)
        window_ids.setflags(write=False)
        record_ids.setflags(write=False)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "window_ids", window_ids)
        object.__setattr__(self, "record_ids", record_ids)

    @property
    def event_count(self) -> int:
        return int(self.X.sum(dtype=np.int64))


def build_demand_matrix(run_dir: str | Path) -> DemandMatrix:
    """Build raw per-window, per-record counts from observable events only."""

    source = Path(run_dir)
    if not source.is_dir():
        raise DemandMatrixError(f"run directory does not exist: {source}")

    config_path = source / "config.json"
    events_path = source / "observable_events.jsonl"
    summary_path = source / "summary.json"
    for path in (config_path, events_path, summary_path):
        if not path.is_file():
            raise DemandMatrixError(f"missing required demand artifact: {path.name}")

    try:
        config = WorkloadConfig.from_json(config_path)
    except (OSError, WorkloadConfigError) as error:
        raise DemandMatrixError(f"config.json: {error}") from error
    summary = _load_json(summary_path)
    _validate_summary(summary, config)

    X = np.zeros((config.num_windows, config.num_records), dtype=np.int64)
    event_count = 0
    try:
        with events_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise DemandMatrixError(
                        "observable_events.jsonl must end every event with a newline"
                    )
                if not line.strip():
                    raise DemandMatrixError(
                        f"observable_events.jsonl line {line_number} must not be blank"
                    )
                try:
                    event = json.loads(line, object_pairs_hook=_unique_object)
                except json.JSONDecodeError as error:
                    raise DemandMatrixError(
                        "malformed JSON in observable_events.jsonl line "
                        f"{line_number}: {error.msg}"
                    ) from error
                _count_event(X, event, line_number, event_count, config)
                event_count += 1
    except UnicodeDecodeError as error:
        raise DemandMatrixError(
            f"malformed UTF-8 in observable_events.jsonl: {error}"
        ) from error

    if event_count != summary["total_events"]:
        raise DemandMatrixError(
            "observable event count does not match summary.total_events: "
            f"{event_count} != {summary['total_events']}"
        )
    if int(X.sum(dtype=np.int64)) != event_count:
        raise DemandMatrixError("demand matrix sum does not match observable events")

    return DemandMatrix(
        X=X,
        window_ids=np.arange(config.num_windows, dtype=np.int64),
        record_ids=np.arange(config.num_records, dtype=np.int64),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise DemandMatrixError(f"malformed JSON in {path.name}: {error.msg}") from error
    if not isinstance(value, dict):
        raise DemandMatrixError(f"{path.name} root must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DemandMatrixError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _validate_summary(summary: Mapping[str, Any], config: WorkloadConfig) -> None:
    required = {"schema_version", "num_windows", "num_records", "total_events"}
    missing = sorted(required - set(summary))
    if missing:
        raise DemandMatrixError(
            "summary.json is missing required fields: " + ", ".join(missing)
        )
    expected = {
        "schema_version": SUPPORTED_SOURCE_SCHEMA_VERSION,
        "num_windows": config.num_windows,
        "num_records": config.num_records,
    }
    for field, expected_value in expected.items():
        actual = _integer(f"summary.{field}", summary[field], minimum=0)
        if actual != expected_value:
            raise DemandMatrixError(
                f"summary.{field} is {actual}, expected {expected_value}"
            )
    _integer("summary.total_events", summary["total_events"], minimum=0)


def _count_event(
    X: np.ndarray,
    event: Any,
    line_number: int,
    expected_event_index: int,
    config: WorkloadConfig,
) -> None:
    path = f"observable_events.jsonl line {line_number}"
    if not isinstance(event, dict):
        raise DemandMatrixError(f"{path} must be a JSON object")
    actual_fields = set(event)
    expected_fields = set(OBSERVABLE_EVENT_FIELDS)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unexpected {unknown}")
        raise DemandMatrixError(f"{path} fields are invalid: {', '.join(details)}")

    event_index = _integer(f"{path}.event_index", event["event_index"], minimum=0)
    if event_index != expected_event_index:
        raise DemandMatrixError(
            f"{path}.event_index must be {expected_event_index}; got {event_index}"
        )
    window_id = _integer(
        f"{path}.window_id",
        event["window_id"],
        minimum=0,
        maximum=config.num_windows - 1,
    )
    record_id = _integer(
        f"{path}.record_id",
        event["record_id"],
        minimum=0,
        maximum=config.num_records - 1,
    )
    X[window_id, record_id] += 1


def _integer(
    path: str,
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DemandMatrixError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise DemandMatrixError(f"{path} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise DemandMatrixError(f"{path} must be at most {maximum}")
    return value
