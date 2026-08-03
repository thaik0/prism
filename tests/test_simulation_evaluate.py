from __future__ import annotations

import numpy as np

from prism.simulation.evaluate import evaluate_policy_results
from prism.simulation.replay import POLICY_NAMES, PolicyReplayResult


def _replay() -> PolicyReplayResult:
    totals = {
        "lru": 120.0,
        "lfu": 110.0,
        "recent_demand_greedy": 100.0,
        "predictive_greedy": 90.0,
        "oracle_greedy": 80.0,
        "oracle_exact": 79.0,
    }
    policy_metrics = {
        name: {"total_combined_cost": totals[name]} for name in POLICY_NAMES
    }
    shape = (len(POLICY_NAMES), 2)
    per_window = {
        "access_count": np.zeros(shape, dtype=np.int64),
        "hit_count": np.ones(shape, dtype=np.int64),
        "miss_count": np.ones(shape, dtype=np.int64),
        "access_cost": np.full(shape, 10.0),
        "promotion_cost": np.zeros(shape),
        "combined_cost": np.full(shape, 10.0),
        "promotion_count": np.zeros(shape, dtype=np.int64),
        "eviction_count": np.zeros(shape, dtype=np.int64),
        "bytes_promoted": np.zeros(shape, dtype=np.int64),
        "bytes_evicted": np.zeros(shape, dtype=np.int64),
        "end_occupancy_bytes": np.zeros(shape, dtype=np.int64),
        "end_resident_record_count": np.zeros(shape, dtype=np.int64),
    }
    recent = POLICY_NAMES.index("recent_demand_greedy")
    predictive = POLICY_NAMES.index("predictive_greedy")
    per_window["combined_cost"][recent] = [20.0, 20.0]
    per_window["combined_cost"][predictive] = [10.0, 15.0]
    return PolicyReplayResult(
        policy_names=POLICY_NAMES,
        test_window_ids=np.array([3, 4]),
        test_event_indices=np.array([0]),
        per_event_tier_cost=np.ones((6, 1)),
        per_event_hit=np.ones((6, 1), dtype=np.bool_),
        per_window=per_window,
        final_resident_indicator=np.zeros((6, 2), dtype=np.bool_),
        policy_metrics=policy_metrics,
        exact_solver_diagnostics=(
            {
                "window_id": 3,
                "period": "test",
                "solver_status": "optimal_status_0",
                "objective_value": 1.0,
                "selected_record_count": 1,
            },
        ),
        capacity_violations=0,
    )


def test_transition_windows_are_unique_and_all_five_gates_are_exact() -> None:
    bursts = [
        {"burst_id": 0, "working_set_id": 0, "start_window": 3},
        {"burst_id": 1, "working_set_id": 1, "start_window": 3},
        {"burst_id": 2, "working_set_id": 0, "start_window": 4},
        {"burst_id": 3, "working_set_id": 0, "start_window": 2},
    ]

    report = evaluate_policy_results(_replay(), bursts)
    transition = report["transition_metrics"]
    gates = report["scientific_gates"]

    assert transition["test_burst_start_count"] == 3
    assert transition["unique_burst_start_window_count"] == 2
    assert transition["unique_first_two_window_count"] == 2
    assert transition["burst_start_window_ids"] == [3, 4]
    assert len(transition["per_burst_descriptive_costs"]) == 3
    assert gates["all_passed"]
    assert gates["gate_1"]["right_minus_left"] == 10.0
    assert gates["gate_2"]["lru_comparison"]["right_minus_left"] == 30.0
    assert gates["gate_3"]["right_minus_left"] == 15.0
    assert gates["gate_4"]["right_minus_left"] == 20.0
    assert gates["gate_5"]["right_minus_left"] == 1.0
    assert report["oracle_ordering_diagnostic"]["passed"]


def test_strict_gate_equality_fails_without_changing_exact_tolerance() -> None:
    replay = _replay()
    replay.policy_metrics["predictive_greedy"]["total_combined_cost"] = 100.0

    report = evaluate_policy_results(replay, [])

    assert not report["scientific_gates"]["gate_1"]["passed"]
    assert not report["scientific_gates"]["gate_3"]["passed"]
    assert not report["scientific_gates"]["all_passed"]
