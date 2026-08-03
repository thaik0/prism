from __future__ import annotations

import json

import numpy as np
import pytest

from prism.predictor import (
    PredictorConfig,
    PredictorFeatureError,
    build_predictor_dataset,
    reconstruct_window_context,
)
from prism.structure import DemandMatrix
from prism.workload import WorkloadConfig
from tests.conftest import base_config_dict


def _source_config() -> WorkloadConfig:
    raw = base_config_dict()
    raw.update(
        {
            "num_windows": 8,
            "num_records": 2,
            "working_set_support_min": 1,
            "working_set_support_max": 2,
        }
    )
    return WorkloadConfig.from_dict(raw)


def _event(index: int, window: int, request: int, session: int, user: int, kind: str):
    return {
        "event_index": index,
        "operation_type": "read",
        "record_id": index % 2,
        "record_size_bytes": 100,
        "request_id": request,
        "request_type": kind,
        "session_id": session,
        "user_id": user,
        "window_id": window,
    }


def _write_events(run_dir) -> list[dict[str, object]]:
    run_dir.mkdir()
    events = []
    index = 0
    request = 0
    session = 0
    for window in range(8):
        window_requests = [(0, "interactive")]
        if window == 2:
            window_requests.append((1, "batch"))
        for user, kind in window_requests:
            access_count = 2 if window == 2 and user == 0 else 1
            for _ in range(access_count):
                events.append(_event(index, window, request, session, user, kind))
                index += 1
            request += 1
            session += 1
    (run_dir / "observable_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return events


def test_request_reconstruction_and_exact_factor_specific_features(tmp_path) -> None:
    source = tmp_path / "source"
    events = _write_events(source)
    config = _source_config()
    contexts, user_ids, request_types, warnings = reconstruct_window_context(
        source, config
    )

    assert user_ids.tolist() == [0, 1]
    assert request_types == ("batch", "interactive")
    assert warnings == ()
    assert contexts[2].session_count == 2
    assert contexts[2].request_count == 2
    assert contexts[2].access_count == 3
    assert contexts[2].user_fractions.tolist() == [0.5, 0.5]
    assert contexts[2].request_type_fractions.tolist() == [0.5, 0.5]

    X = np.zeros((8, 2), dtype=np.int64)
    for event in events:
        X[event["window_id"], event["record_id"]] += 1
    demand = DemandMatrix(X, np.arange(8), np.arange(2))
    predictor_config = PredictorConfig(7, 0.6, 0.2, 100, 1e-6, 1.0, 5)
    dataset = build_predictor_dataset(
        source,
        demand,
        np.eye(2),
        predictor_config.split_boundaries(8),
        config,
    )

    assert dataset.example_count == 10
    assert dataset.feature_window_ids[:2].tolist() == [2, 2]
    assert dataset.factor_ids[:2].tolist() == [0, 1]
    assert dataset.target_window_ids[:2].tolist() == [3, 3]
    assert dataset.recent_features[0, :8].tolist() == pytest.approx(
        [2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0]
    )
    assert dataset.recent_features[0, 8:].tolist() == [1.0, 0.0]
    assert dataset.recent_features[1, 8:].tolist() == [0.0, 1.0]
    user_start = len(dataset.recent_feature_names)
    assert dataset.context_features[0, user_start : user_start + 2].tolist() == [
        0.5,
        0.5,
    ]
    assert dataset.context_features[0, user_start + 2 : user_start + 4].tolist() == [
        0.0,
        0.0,
    ]
    assert dataset.context_features[1, user_start : user_start + 2].tolist() == [
        0.0,
        0.0,
    ]
    assert dataset.context_features[1, user_start + 2 : user_start + 4].tolist() == [
        0.5,
        0.5,
    ]
    assert dataset.projected_factor_demand.tolist() == X.tolist()


def test_inconsistent_request_metadata_is_rejected(tmp_path) -> None:
    source = tmp_path / "source"
    events = _write_events(source)
    repeated = next(
        index
        for index in range(1, len(events))
        if events[index]["request_id"] == events[index - 1]["request_id"]
    )
    events[repeated]["request_type"] = "batch"
    (source / "observable_events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(PredictorFeatureError, match="request.*inconsistent"):
        reconstruct_window_context(source, _source_config())
