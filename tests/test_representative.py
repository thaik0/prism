from __future__ import annotations

from prism.workload import WorkloadConfig, generate_workload
from tests.conftest import REPRESENTATIVE_CONFIG_PATH


def test_representative_configuration_meets_demonstration_conditions() -> None:
    config = WorkloadConfig.from_json(REPRESENTATIVE_CONFIG_PATH)
    result = generate_workload(config)
    summary = result.summary
    hidden = result.hidden_ground_truth

    assert summary.to_dict() == {
        "schema_version": 1,
        "seed": 1729,
        "num_windows": 40,
        "num_records": 64,
        "total_sessions": 157,
        "total_requests": 466,
        "total_events": 2604,
        "total_bursts": 34,
        "baseline_access_count": 240,
        "noise_access_count": 84,
        "working_set_access_count": 2280,
        "records_in_multiple_working_sets": 17,
    }
    assert len(set(hidden.record_sizes_bytes)) > 1
    assert any(
        trial.activated and trial.previous_window_precursor_score > 0.0
        for trial in hidden.activation_trials
    )
    assert any(
        not trial.activated and trial.previous_window_precursor_score > 0.0
        for trial in hidden.activation_trials
    )
