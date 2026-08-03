from __future__ import annotations

import numpy as np

from prism.simulation import (
    SimulationConfig,
    evaluate_causal_diagnostics,
    run_policy_replay,
)
from prism.workload.models import ObservableEvent


def _events(rows: list[list[int]]) -> tuple[ObservableEvent, ...]:
    events = []
    index = 0
    for window_id, records in enumerate(rows):
        for record_id in records:
            events.append(
                ObservableEvent(
                    event_index=index,
                    window_id=window_id,
                    record_id=record_id,
                    record_size_bytes=[2, 3, 4][record_id],
                    user_id=0,
                    session_id=window_id,
                    request_id=index,
                    request_type="request",
                    operation_type="read",
                )
            )
            index += 1
    return tuple(events)


def _demand(rows: list[list[int]]) -> np.ndarray:
    result = np.zeros((len(rows), 3), dtype=np.int64)
    for window_id, records in enumerate(rows):
        for record_id in records:
            result[window_id, record_id] += 1
    return result


def test_dynamic_changes_disagreements_and_pre_transition_coverage_are_exact() -> None:
    rows = [[0], [0], [0], [1, 1], [1]]
    demand = _demand(rows)
    predictive = np.zeros((5, 3), dtype=np.float64)
    predictive[2, 0] = 2.0
    predictive[3:, 1] = 2.0
    residual = np.zeros_like(predictive)
    residual[2:, 0] = 2.0
    forecasts = {
        "predictive_greedy": predictive,
        "recent_state_only": predictive,
        "activation_intensity_only": predictive,
        "residual_baseline_only": residual,
    }
    config = SimulationConfig(4, 1.0, 10.0, 0.0)
    replay = run_policy_replay(
        _events(rows),
        demand,
        np.arange(3),
        [2, 3, 4],
        predictive,
        np.ones(5, dtype=np.bool_),
        config,
        validation_start=2,
        test_start=3,
        forecast_variants=forecasts,
    )
    hidden = {
        "working_set_memberships": [
            {
                "working_set_id": 0,
                "members": [{"record_id": 1, "weight": 1.0}],
            }
        ],
        "bursts": [
            {
                "burst_id": 0,
                "working_set_id": 0,
                "start_window": 3,
            }
        ],
    }

    report = evaluate_causal_diagnostics(
        replay, demand, np.arange(3), [2, 3, 4], hidden, config
    )
    dynamic = report["dynamic_action"]
    first = dynamic["per_window_by_policy"]["predictive_greedy"][0]

    assert first["target_set_changed"]
    assert first["records_added"] == 1
    assert first["records_removed"] == 1
    assert first["bytes_added"] == 3
    assert first["bytes_removed"] == 2
    assert first["jaccard_similarity"] == 0.0
    assert first["promotion_count"] == 1
    assert dynamic["predictive_greedy_summary"]["dynamic_test_action"]
    assert dynamic["predictive_greedy_summary"]["comparisons"][
        "validation_final_frozen"
    ]["record_window_residency_disagreements"] == 4

    transition = report["pre_transition"]["per_burst"][0]
    predictive_row = transition["policy_metrics"]["predictive_greedy"]
    frozen_row = transition["policy_metrics"]["validation_final_frozen"]
    assert predictive_row["supported_record_coverage_fraction"] == 1.0
    assert predictive_row["membership_weighted_coverage"] == 1.0
    assert predictive_row["pre_transition_realized_demand_coverage_fraction"] == 1.0
    assert predictive_row["proactive_useful_changes"][
        "accessed_in_start_window_count"
    ] == 1
    assert frozen_row["supported_record_coverage_fraction"] == 0.0
    assert transition["missed_oracle_opportunities"]["record_count"] == 0


def test_missed_oracle_access_cost_uses_current_window_only() -> None:
    rows = [[0], [0], [0], [1, 1], [2, 2, 2]]
    demand = _demand(rows)
    predictive = np.zeros((5, 3), dtype=np.float64)
    predictive[2:, 0] = 2.0
    forecasts = {name: predictive for name in (
        "predictive_greedy",
        "recent_state_only",
        "activation_intensity_only",
        "residual_baseline_only",
    )}
    config = SimulationConfig(4, 1.0, 10.0, 0.0)
    replay = run_policy_replay(
        _events(rows), demand, np.arange(3), [2, 3, 4], predictive,
        np.ones(5, dtype=np.bool_), config, 2, 3, forecast_variants=forecasts
    )
    hidden = {
        "working_set_memberships": [
            {"working_set_id": 0, "members": [{"record_id": 1, "weight": 1.0}]}
        ],
        "bursts": [{"burst_id": 0, "working_set_id": 0, "start_window": 3}],
    }
    report = evaluate_causal_diagnostics(
        replay, demand, np.arange(3), [2, 3, 4], hidden, config
    )
    missed = report["pre_transition"]["per_burst"][0][
        "missed_oracle_opportunities"
    ]
    assert missed["record_ids"] == [1]
    assert missed["actual_current_window_accesses"] == 2
    assert missed["additional_access_cost"] == 18.0
