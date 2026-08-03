from __future__ import annotations

import numpy as np

from prism.simulation import POLICY_NAMES, SimulationConfig, run_policy_replay
from prism.workload.models import ObservableEvent


def _events(window_records: list[list[int]], sizes: list[int]) -> tuple[ObservableEvent, ...]:
    result = []
    event_index = 0
    for window_id, records in enumerate(window_records):
        for record_id in records:
            result.append(
                ObservableEvent(
                    event_index=event_index,
                    window_id=window_id,
                    record_id=record_id,
                    record_size_bytes=sizes[record_id],
                    user_id=0,
                    session_id=window_id,
                    request_id=event_index,
                    request_type="request",
                    operation_type="read",
                )
            )
            event_index += 1
    return tuple(result)


def _demand(window_records: list[list[int]], record_count: int) -> np.ndarray:
    result = np.zeros((len(window_records), record_count), dtype=np.int64)
    for window_id, records in enumerate(window_records):
        for record_id in records:
            result[window_id, record_id] += 1
    return result


def _run(
    window_records: list[list[int]],
    sizes: list[int],
    capacity: int,
    predictive: np.ndarray | None = None,
):
    demand = _demand(window_records, len(sizes))
    if predictive is None:
        predictive = demand.astype(np.float64)
    available = np.ones(len(window_records), dtype=np.bool_)
    return run_policy_replay(
        _events(window_records, sizes),
        demand,
        np.arange(len(sizes)),
        sizes,
        predictive,
        available,
        SimulationConfig(capacity, 1.0, 10.0, 0.5),
        validation_start=2,
        test_start=3,
    )


def test_validation_state_carries_and_reactive_miss_is_charged_before_admission() -> None:
    result = _run(
        [[0], [0], [0, 0], [0, 1, 1], [2]],
        sizes=[1, 1, 3],
        capacity=2,
    )
    lru_index = result.policy_names.index("lru")
    metrics = result.policy_metrics["lru"]

    np.testing.assert_array_equal(
        result.per_event_tier_cost[lru_index], [1.0, 10.0, 1.0, 10.0]
    )
    np.testing.assert_array_equal(result.per_event_hit[lru_index], [1, 0, 1, 0])
    assert metrics["promotion_count"] == 1
    assert metrics["bytes_promoted"] == 1
    assert metrics["slow_reads"] == 2
    assert metrics["total_access_cost"] == 22.0
    assert result.per_window["promotion_count"][lru_index, 0] == 1
    assert result.final_resident_indicator[lru_index].tolist() == [True, True, False]


def test_lru_and_lfu_use_their_exact_eviction_orders_without_decay() -> None:
    result = _run(
        [[0], [1], [0, 0, 0, 1], [2], [2]],
        sizes=[1, 1, 1],
        capacity=2,
    )
    lru = result.policy_names.index("lru")
    lfu = result.policy_names.index("lfu")

    assert result.final_resident_indicator[lru].tolist() == [False, True, True]
    assert result.final_resident_indicator[lfu].tolist() == [True, False, True]
    assert result.policy_metrics["lru"]["promotion_count"] == 1
    assert result.policy_metrics["lfu"]["promotion_count"] == 1
    assert result.policy_metrics["lfu"]["eviction_count"] == 1


def test_boundary_policies_place_before_events_and_remain_fixed_within_window() -> None:
    records = [[0], [0], [0, 0], [1, 1], [0, 1]]
    predictive = np.zeros((5, 2), dtype=np.float64)
    predictive[2, 0] = 2.0
    predictive[3, 1] = 2.0
    predictive[4, 0] = 1.0
    result = _run(records, sizes=[1, 1], capacity=1, predictive=predictive)
    predictive_index = result.policy_names.index("predictive_greedy")
    recent_index = result.policy_names.index("recent_demand_greedy")

    np.testing.assert_array_equal(
        result.per_event_hit[predictive_index], [True, True, True, False]
    )
    np.testing.assert_array_equal(
        result.per_event_hit[recent_index], [False, False, False, True]
    )
    assert result.per_window["promotion_count"][predictive_index, 0] == 1
    assert result.per_window["promotion_count"][recent_index, 0] == 0


def test_all_eleven_policies_receive_identical_test_events_and_independent_state() -> None:
    result = _run(
        [[0], [1], [0, 1], [2, 0, 1], [1, 2]],
        sizes=[1, 1, 1],
        capacity=2,
    )

    assert result.policy_names == POLICY_NAMES
    assert result.per_event_tier_cost.shape == (11, 5)
    assert result.per_event_hit.shape == (11, 5)
    np.testing.assert_array_equal(result.test_event_indices, [4, 5, 6, 7, 8])
    assert result.capacity_violations == 0
    assert result.final_resident_indicator.shape == (11, 3)
    assert len(result.exact_solver_diagnostics) == 3
    assert [row["period"] for row in result.exact_solver_diagnostics] == [
        "validation",
        "test",
        "test",
    ]
    assert all(
        row["solver_status"].startswith("optimal")
        for row in result.exact_solver_diagnostics
    )


def test_static_controls_are_independent_and_never_migrate_during_test() -> None:
    records = [[0], [0], [0, 0], [1, 1], [0, 1]]
    predictive = np.zeros((5, 2), dtype=np.float64)
    predictive[2:, 1] = 2.0
    result = _run(records, sizes=[1, 1], capacity=1, predictive=predictive)
    training = result.policy_names.index("training_popularity_static")
    frozen = result.policy_names.index("validation_final_frozen")
    predictive_index = result.policy_names.index("predictive_greedy")

    assert result.policy_metrics["training_popularity_static"]["promotion_count"] == 0
    assert result.policy_metrics["validation_final_frozen"]["promotion_count"] == 0
    assert result.policy_metrics["training_popularity_static"]["validation_setup"][
        "promotion_count"
    ] == 1
    np.testing.assert_array_equal(
        result.pre_window_resident_indicator[training, 0],
        result.pre_window_resident_indicator[training, 1],
    )
    np.testing.assert_array_equal(
        result.previous_target_indicator[frozen],
        result.pre_window_resident_indicator[frozen, 0],
    )
    assert not np.shares_memory(
        result.pre_window_resident_indicator[frozen],
        result.pre_window_resident_indicator[predictive_index],
    )


def test_wasted_promotions_count_only_test_started_zero_access_episodes() -> None:
    records = [[0], [0], [0], [1], [0]]
    predictive = np.zeros((5, 2), dtype=np.float64)
    predictive[2, 0] = 1.0
    predictive[3, 0] = 1.0  # retained from validation, but unused in test window 3
    predictive[4, 1] = 1.0  # promotes record 1 and never accesses it
    result = _run(records, sizes=[1, 1], capacity=1, predictive=predictive)
    metrics = result.policy_metrics["predictive_greedy"]

    assert metrics["promotion_count"] == 1
    assert metrics["wasted_promotion_count"] == 1
    assert metrics["wasted_promoted_bytes"] == 1
    assert metrics["wasted_promotion_cost"] == 0.5


def test_test_totals_equal_per_window_and_validation_costs_are_excluded() -> None:
    result = _run(
        [[0], [0], [0, 1], [1, 1], [0]], sizes=[1, 1], capacity=1
    )
    for policy_index, policy_name in enumerate(result.policy_names):
        metrics = result.policy_metrics[policy_name]
        assert metrics["event_count"] == 3
        assert metrics["total_access_cost"] == np.sum(
            result.per_window["access_cost"][policy_index]
        )
        assert metrics["total_promotion_cost"] == np.sum(
            result.per_window["promotion_cost"][policy_index]
        )
        assert metrics["total_combined_cost"] == np.sum(
            result.per_window["combined_cost"][policy_index]
        )
