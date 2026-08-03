from __future__ import annotations

import json
import math

import pytest

from prism.workload import WorkloadConfig, WorkloadConfigError
from tests.conftest import base_config_dict


def test_valid_config_is_canonicalized() -> None:
    raw = base_config_dict()
    raw["request_types"] = ["interactive", "batch"]
    raw["operation_type_probabilities"] = {"write": 0.2, "read": 0.8}

    config = WorkloadConfig.from_dict(raw)

    assert config.request_types == ("interactive", "batch")
    assert list(config.operation_type_probabilities) == ["read", "write"]
    assert config.operation_type_probabilities == {"read": 0.8, "write": 0.2}
    assert config.to_dict()["request_types"] == ["interactive", "batch"]


def test_missing_and_unknown_fields_are_rejected() -> None:
    missing = base_config_dict()
    del missing["num_windows"]
    with pytest.raises(WorkloadConfigError, match="missing.*num_windows"):
        WorkloadConfig.from_dict(missing)

    unknown = base_config_dict()
    unknown["num_window"] = 10
    with pytest.raises(WorkloadConfigError, match="unknown.*num_window"):
        WorkloadConfig.from_dict(unknown)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_records": True}, "num_records"),
        ({"num_windows": 1}, "num_windows"),
        (
            {"record_size_min_bytes": 300, "record_size_max_bytes": 200},
            "record_size_min_bytes",
        ),
        (
            {"working_set_support_max": 9},
            "working_set_support_max.*num_records",
        ),
        ({"request_types": []}, "request_types"),
        ({"request_types": ["same", "same"]}, "request_types.*unique"),
        ({"request_types": [""]}, "request_types.*nonempty"),
        ({"operation_type_probabilities": {}}, "operation_type_probabilities"),
        (
            {"operation_type_probabilities": {"read": 0.8, "write": 0.3}},
            "sum to 1",
        ),
        (
            {"operation_type_probabilities": {"read": 1.1, "write": -0.1}},
            "operation_type_probabilities.write",
        ),
        ({"spontaneous_activation_probability": 1.1}, "spontaneous"),
        ({"precursor_probability_scale": -0.1}, "precursor"),
        ({"burst_intensity_min": 0.0}, "burst_intensity_min"),
        ({"burst_duration_min_windows": 0}, "burst_duration_min_windows"),
        (
            {"baseline_access_weight": 0.0, "noise_access_weight": 0.0},
            "cannot both be zero",
        ),
        ({"burst_access_weight": -1.0}, "burst_access_weight"),
        ({"burst_intensity_max": math.inf}, "burst_intensity_max.*finite"),
        ({"noise_access_weight": math.nan}, "noise_access_weight.*finite"),
    ],
)
def test_invalid_values_are_rejected(
    updates: dict[str, object], message: str
) -> None:
    raw = base_config_dict()
    raw.update(updates)

    with pytest.raises(WorkloadConfigError, match=message):
        WorkloadConfig.from_dict(raw)


def test_json_loader_rejects_duplicate_object_keys(tmp_path) -> None:
    raw = base_config_dict()
    raw["operation_type_probabilities"] = {"read": 1.0}
    serialized = json.dumps(raw)
    serialized = serialized.replace(
        '"operation_type_probabilities": {"read": 1.0}',
        '"operation_type_probabilities": {"read": 1.0, "read": 0.0}',
    )
    config_path = tmp_path / "duplicate.json"
    config_path.write_text(serialized, encoding="utf-8")

    with pytest.raises(WorkloadConfigError, match="duplicate JSON field: read"):
        WorkloadConfig.from_json(config_path)


def test_probability_sum_tolerance_is_tight_but_documented() -> None:
    raw = base_config_dict()
    raw["operation_type_probabilities"] = {
        "read": 0.3333333333333,
        "write": 0.6666666666667,
    }

    assert WorkloadConfig.from_dict(raw).operation_type_probabilities == raw[
        "operation_type_probabilities"
    ]
