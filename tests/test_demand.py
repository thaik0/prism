from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from prism.structure import DemandMatrix, DemandMatrixError, build_demand_matrix
from prism.workload import generate_workload, persist_workload
from tests.conftest import base_config_dict


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _event(event_index: int, window_id: int, record_id: int) -> dict[str, object]:
    return {
        "event_index": event_index,
        "operation_type": "read",
        "record_id": record_id,
        "record_size_bytes": 100,
        "request_id": event_index,
        "request_type": "interactive",
        "session_id": event_index,
        "user_id": 0,
        "window_id": window_id,
    }


def _write_demand_source(run_dir: Path) -> list[dict[str, object]]:
    run_dir.mkdir()
    config = base_config_dict()
    config.update(
        {
            "num_windows": 4,
            "num_records": 5,
            "working_set_support_min": 2,
            "working_set_support_max": 4,
        }
    )
    events = [_event(0, 0, 0), _event(1, 0, 0), _event(2, 2, 3)]
    _write_json(run_dir / "config.json", config)
    (run_dir / "observable_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "num_windows": 4,
            "num_records": 5,
            "total_events": 3,
        },
    )
    return events


def test_raw_demand_matrix_counts_all_events_and_zero_axes(tmp_path) -> None:
    run_dir = tmp_path / "run"
    events = _write_demand_source(run_dir)

    demand = build_demand_matrix(run_dir)

    assert demand.X.dtype == np.int64
    assert demand.X.shape == (4, 5)
    assert demand.window_ids.tolist() == [0, 1, 2, 3]
    assert demand.record_ids.tolist() == [0, 1, 2, 3, 4]
    assert demand.X.tolist() == [
        [2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    assert demand.event_count == len(events)
    assert demand.X.sum(axis=1).tolist() == [2, 0, 1, 0]
    assert demand.X.sum(axis=0).tolist() == [2, 0, 0, 1, 0]


def test_demand_matrix_rejects_noninteger_counts() -> None:
    with pytest.raises(DemandMatrixError, match="integer count dtype"):
        DemandMatrix(
            X=np.array([[1.5]]),
            window_ids=np.array([0]),
            record_ids=np.array([0]),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("window_id", 4, "window_id must be at most"),
        ("record_id", 5, "record_id must be at most"),
        ("record_id", True, "record_id must be an integer"),
    ],
)
def test_malformed_event_references_are_rejected(
    tmp_path, field: str, value: object, message: str
) -> None:
    run_dir = tmp_path / "run"
    events = _write_demand_source(run_dir)
    events[0][field] = value
    (run_dir / "observable_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(DemandMatrixError, match=message):
        build_demand_matrix(run_dir)


def test_noncontiguous_event_ids_are_rejected(tmp_path) -> None:
    run_dir = tmp_path / "run"
    events = _write_demand_source(run_dir)
    events[1]["event_index"] = 9
    (run_dir / "observable_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(DemandMatrixError, match="event_index must be 1"):
        build_demand_matrix(run_dir)


def test_hidden_truth_changes_do_not_change_demand(tmp_path, make_config) -> None:
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    persist_workload(generate_workload(make_config()), source_a)
    shutil.copytree(source_a, source_b)
    hidden_path = source_b / "hidden_ground_truth.json"
    hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
    hidden["working_set_memberships"].reverse()
    _write_json(hidden_path, hidden)

    demand_a = build_demand_matrix(source_a)
    demand_b = build_demand_matrix(source_b)

    assert np.array_equal(demand_a.X, demand_b.X)
    assert np.array_equal(demand_a.window_ids, demand_b.window_ids)
    assert np.array_equal(demand_a.record_ids, demand_b.record_ids)


def test_non_demand_observable_fields_do_not_change_counts(tmp_path) -> None:
    source_a = tmp_path / "a"
    events = _write_demand_source(source_a)
    source_b = tmp_path / "b"
    shutil.copytree(source_a, source_b)
    for event in events:
        event["record_size_bytes"] = 999999
        event["user_id"] = 999
        event["session_id"] = 999
        event["request_id"] = 999
        event["request_type"] = "ignored-context"
        event["operation_type"] = "ignored-operation"
    (source_b / "observable_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    assert np.array_equal(
        build_demand_matrix(source_a).X, build_demand_matrix(source_b).X
    )
