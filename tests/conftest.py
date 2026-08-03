from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from prism.workload import WorkloadConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "milestone1_representative.json"
)
REPRESENTATIVE_STRUCTURE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "milestone2_representative.json"
)


def base_config_dict() -> dict[str, Any]:
    return {
        "seed": 41,
        "num_windows": 4,
        "num_records": 8,
        "num_working_sets": 2,
        "num_users": 2,
        "request_types": ["interactive", "batch"],
        "operation_type_probabilities": {"read": 0.8, "write": 0.2},
        "record_size_min_bytes": 100,
        "record_size_max_bytes": 200,
        "working_set_support_min": 2,
        "working_set_support_max": 4,
        "min_sessions_per_window": 1,
        "max_sessions_per_window": 2,
        "min_requests_per_session": 1,
        "max_requests_per_session": 2,
        "min_accesses_per_request": 1,
        "max_accesses_per_request": 3,
        "spontaneous_activation_probability": 0.2,
        "precursor_probability_scale": 0.5,
        "burst_duration_min_windows": 1,
        "burst_duration_max_windows": 2,
        "burst_intensity_min": 1.0,
        "burst_intensity_max": 2.0,
        "burst_access_weight": 1.0,
        "baseline_access_weight": 1.0,
        "noise_access_weight": 0.5,
    }


@pytest.fixture
def make_config():
    def factory(**overrides: Any) -> WorkloadConfig:
        raw = base_config_dict()
        raw.update(overrides)
        return WorkloadConfig.from_dict(raw)

    return factory
