from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prism.experiments.actionability import CommonWindowProtocol
from prism.experiments.actionability_aggregate import _cell_gates, decide_thesis_status
from prism.experiments.runner import run_experiments
from prism.simulation import SimulationConfig, run_policy_replay
from prism.simulation.actionability import (
    REJECTION_CODES,
    _controller_report,
    _diagnostic_candidate_set,
    _promotion_episodes,
    average_ranks,
)
from prism.simulation.projection import fit_record_demand_projection
from prism.workload.models import ObservableEvent


def _events(rows: list[list[int]], sizes: list[int]) -> tuple[ObservableEvent, ...]:
    events = []
    event_index = 0
    for window, records in enumerate(rows):
        for record in records:
            events.append(
                ObservableEvent(
                    event_index,
                    window,
                    record,
                    sizes[record],
                    0,
                    window,
                    event_index,
                    "request",
                    "read",
                )
            )
            event_index += 1
    return tuple(events)


def _demand(rows: list[list[int]], records: int) -> np.ndarray:
    result = np.zeros((len(rows), records), dtype=np.int64)
    for window, values in enumerate(rows):
        for record in values:
            result[window, record] += 1
    return result


def test_horizon_projection_uses_cumulative_targets_without_division() -> None:
    X = np.asarray([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]], dtype=np.int64)
    membership = np.eye(2)
    probability = np.zeros((5, 2))
    intensity = np.ones((5, 2))
    available = np.asarray([False, True, True, True, False])
    result = fit_record_demand_projection(
        X,
        membership,
        probability,
        intensity,
        available,
        4,
        forecast_horizon_windows=2,
        training_target_window_ids=np.asarray([1, 2]),
    )
    np.testing.assert_array_equal(
        result.cumulative_future_record_demand[2], X[2] + X[3]
    )
    assert result.predicted_record_demand[2].sum() > X[2].sum()
    assert result.model.training_target_window_ids.tolist() == [1, 2]


def test_common_replay_embargo_tail_and_matched_recent_oracle_inputs() -> None:
    rows = [[0], [0], [0], [0], [1, 1], [1], [0], [1], [0, 0], [1, 1]]
    sizes = [1, 1]
    demand = _demand(rows, 2)
    forecast = np.ones((10, 2), dtype=np.float64)
    available = np.zeros(10, dtype=np.bool_)
    available[[2, 3, 5, 6, 7]] = True
    result = run_policy_replay(
        _events(rows, sizes),
        demand,
        np.arange(2),
        sizes,
        forecast,
        available,
        SimulationConfig(1, 1.0, 10.0, 0.0),
        validation_start=2,
        test_start=5,
        forecast_horizon_windows=2,
        validation_decision_end=4,
        test_evaluation_end=8,
    )
    assert result.test_window_ids.tolist() == [5, 6, 7]
    assert result.test_event_indices.tolist() == [6, 7, 8]
    assert all(row["window_id"] != 4 for row in result.exact_solver_diagnostics)
    assert all(row["window_id"] < 8 for row in result.exact_solver_diagnostics)
    assert result.policy_metrics["training_popularity_static"]["validation_setup"][
        "promotion_count"
    ] == 1


def test_rank_candidate_and_controller_classifications_are_exact() -> None:
    np.testing.assert_array_equal(average_ranks([3.0, 3.0, 1.0]), [1.5, 1.5, 3.0])
    config = SimulationConfig(1, 1.0, 10.0, 18.0)
    ids = np.asarray([0, 1, 2])
    sizes = np.asarray([1, 1, 1])
    assert _diagnostic_candidate_set(
        np.asarray([1.0, 5.0, 4.0]), ids, sizes, config
    ) == {1}
    forecasts = np.zeros((5, 3), dtype=np.float64)
    forecasts[2] = [1.0, 5.0, 4.0]
    forecasts[3] = [1.0, 5.0, 4.0]
    protocol = CommonWindowProtocol(2, 2, 4, 4, 5, 1)
    report, arrays, targets = _controller_report(
        np.asarray([2, 3]), forecasts, ids, sizes, config, protocol
    )
    first = arrays["rejection_reason_code"][0]
    assert first.tolist() == [
        REJECTION_CODES["movement_cost"],
        REJECTION_CODES["selected"],
        REJECTION_CODES["capacity"],
    ]
    assert targets[2] == {1}
    assert report["per_window"][0]["movement_cost_rejection"]["record_count"] == 1
    assert report["per_window"][0]["capacity_rejection"]["record_count"] == 1


def test_promotion_repayment_stops_at_eviction_and_uses_horizon_tolerance() -> None:
    protocol = CommonWindowProtocol(2, 2, 5, 5, 10, 2)
    targets = {
        3: {0},
        5: {0, 1, 2},
        6: {0, 1, 2},
        7: {0},
        8: {0},
    }
    X = np.zeros((10, 3), dtype=np.float64)
    X[6, 1] = 1
    X[7, 1] = 100
    X[8, 2] = 100
    hidden = {
        "working_set_memberships": [
            {
                "working_set_id": 0,
                "members": [
                    {"record_id": 1, "weight": 0.5},
                    {"record_id": 2, "weight": 0.5},
                ],
            }
        ],
        "bursts": [
            {"burst_id": 0, "working_set_id": 0, "start_window": 6}
        ],
    }
    rows = _promotion_episodes(
        targets,
        X,
        np.arange(3),
        np.asarray([1, 1, 1]),
        SimulationConfig(3, 1.0, 10.0, 9.0),
        protocol,
        2,
        hidden,
        "sparse",
        1729,
    )
    by_record = {row["record_id"]: row for row in rows}
    assert by_record[1]["qualifying_fast_access_count"] == 1
    assert by_record[1]["realized_savings"] == 9.0
    assert by_record[1]["repaid"]
    assert by_record[1]["accessed_later_within_horizon"]
    assert by_record[2]["qualifying_fast_access_count"] == 0
    assert not by_record[2]["pre_demand_promotion"]


def _gate_rows() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "predictive_frozen_difference_fraction": 0.10,
            "predictive_cost": 90.0,
            "frozen_cost": 100.0,
            "recent_state_cost": 100.0,
            "predictive_access_cost": 90.0,
            "frozen_access_cost": 100.0,
            "predictive_transition_coverage": 0.60,
            "frozen_transition_coverage": 0.50,
            "promotion_repayment": {
                "total_test_promotions": 4,
                "pre_demand_promotions": 4,
                "repaid_promotions": 2,
                "total_promotion_cost": 10.0,
                "total_realized_savings": 11.0,
            },
        }
        for seed in (1729, 2718, 31415)
    ]


THRESHOLDS = {
    "gate_a_mean_difference_fraction_minimum": 0.10,
    "gate_a_seed_difference_fraction_minimum": 0.10,
    "gate_a_minimum_passing_seed_count": 2,
    "gate_b_minimum_seed_win_count": 2,
    "gate_c_minimum_seed_win_count": 2,
    "gate_c_transition_coverage_absolute_improvement_minimum": 0.05,
    "gate_d_minimum_pre_demand_promotions": 10,
    "gate_d_minimum_repayment_fraction": 0.50,
}


def test_all_four_thesis_gates_pass_at_exact_thresholds() -> None:
    result = _cell_gates("sparse__h2", _gate_rows(), THRESHOLDS)
    assert result["gate_a_dynamic_behavior"]["passed"]
    assert result["gate_b_combined_cost"]["passed"]
    assert result["gate_c_non_migration_value"]["passed"]
    assert result["gate_c_non_migration_value"]["passed_route"] == "access_cost"
    assert result["gate_d_useful_promotions"]["passed"]
    assert result["passed"]


def test_thesis_gate_requires_both_comparators_nonmigration_and_positive_net() -> None:
    rows = _gate_rows()
    rows[0]["recent_state_cost"] = 80.0
    rows[1]["recent_state_cost"] = 80.0
    rows[2]["predictive_access_cost"] = 110.0
    rows[1]["predictive_access_cost"] = 110.0
    rows[0]["predictive_transition_coverage"] = 0.51
    rows[1]["predictive_transition_coverage"] = 0.51
    rows[2]["predictive_transition_coverage"] = 0.51
    for row in rows:
        row["promotion_repayment"]["total_realized_savings"] = 10.0
    result = _cell_gates("very_sparse__h4", rows, THRESHOLDS)
    assert not result["gate_b_combined_cost"]["passed"]
    assert not result["gate_c_non_migration_value"]["passed"]
    assert not result["gate_d_useful_promotions"]["passed"]
    assert not result["passed"]


@pytest.mark.parametrize("passing_count", [1, 2, 4])
def test_final_decision_accepts_one_or_more_passing_candidate_cells(
    passing_count: int,
) -> None:
    cells = [{"passed": index < passing_count} for index in range(4)]
    assert decide_thesis_status(
        cells,
        engineering_complete=True,
        regime_separation_sufficient=True,
    ) == "actionable_predictive_tiering_demonstrated"


def test_final_decision_reframes_when_every_candidate_cell_fails() -> None:
    assert decide_thesis_status(
        [{"passed": False} for _ in range(4)],
        engineering_complete=True,
        regime_separation_sufficient=True,
    ) == "stable_cost_aware_tiering_reframe"


@pytest.mark.parametrize(
    ("engineering_complete", "regime_separation_sufficient"),
    [(False, True), (True, False), (False, False)],
)
def test_final_decision_reports_insufficient_evidence_for_incomplete_foundations(
    engineering_complete: bool,
    regime_separation_sufficient: bool,
) -> None:
    assert decide_thesis_status(
        [{"passed": True}],
        engineering_complete=engineering_complete,
        regime_separation_sufficient=regime_separation_sufficient,
    ) == "insufficient_evidence"


def test_actionability_runner_has_27_statuses_and_hash_verified_resume(
    tmp_path, monkeypatch
) -> None:
    manifest = Path(__file__).resolve().parents[1] / "configs/milestone55_actionability.json"

    def completed(manifest, regime, horizon, seed, run_dir, entry):
        (run_dir / "marker.txt").write_text(
            f"{regime}:{horizon}:{seed}\n", encoding="utf-8"
        )
        from prism.experiments.runner import _hash_run_artifacts
        return {
            "resolved_configuration_sha256": {"fixture": "a" * 64},
            "source_artifact_sha256": {"fixture": "b" * 64},
            "artifact_sha256": _hash_run_artifacts(run_dir),
            "scientific_gate_outcomes": {"unfavorable": False},
        }

    monkeypatch.setattr(
        "prism.experiments.runner._execute_actionability_run", completed
    )
    monkeypatch.setattr(
        "prism.experiments.actionability_aggregate.write_actionability_outputs",
        lambda output_dir, manifest: {},
    )
    output = tmp_path / "output"
    first = run_experiments(
        manifest,
        output,
        experiment_id="very_sparse__h4__seed_31415",
    )
    index = json.loads((output / "experiment_index.json").read_text())
    assert len(index["runs"]) == 27
    assert first.completed_count == 1
    selected = next(row for row in index["runs"] if row["status"] == "completed")
    assert selected["regime"] == "very_sparse"
    assert selected["horizon"] == 4
    assert selected["scientific_gate_outcomes"] == {"unfavorable": False}

    monkeypatch.setattr(
        "prism.experiments.runner._execute_actionability_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("recomputed")),
    )
    resumed = run_experiments(
        manifest,
        output,
        experiment_id="very_sparse__h4__seed_31415",
        resume=True,
    )
    assert resumed.reused_experiment_ids == ("very_sparse__h4__seed_31415",)
