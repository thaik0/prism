from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prism.structure import (
    DemandMatrix,
    LearnedStructure,
    StructureLearnerConfig,
    evaluate_recovery,
)
from prism.structure.evaluate import (
    _aggregate_support_metrics,
    _support_metrics,
    match_factors,
)


def test_cosine_similarity_permutation_and_deterministic_assignment() -> None:
    planted = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]], dtype=np.float64
    )
    learned = planted[[1, 0]]

    first = match_factors(learned, planted)
    second = match_factors(learned, planted)

    similarity, assignment, warnings = first
    np.testing.assert_allclose(similarity, [[0.0, 1.0], [1.0, 0.0]])
    assert assignment == ((0, 1), (1, 0))
    assert warnings == ()
    assert np.array_equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_optimal_assignment_beats_row_order_greedy_choice() -> None:
    planted = np.eye(2, dtype=np.float64)
    learned = np.array([[0.8, 0.6], [0.99, 0.1410673598]])

    similarity, assignment, _ = match_factors(learned, planted)

    assert assignment == ((0, 1), (1, 0))
    optimal_total = sum(similarity[left, right] for left, right in assignment)
    row_order_greedy_total = similarity[0, 0] + similarity[1, 1]
    assert optimal_total > row_order_greedy_total


def test_zero_norm_learned_factor_has_zero_similarity_and_warning() -> None:
    similarity, assignment, warnings = match_factors(
        np.array([[0.0, 0.0], [1.0, 0.0]]), np.eye(2)
    )

    assert similarity[0].tolist() == [0.0, 0.0]
    assert sorted(assignment) == [(0, 1), (1, 0)]
    assert warnings == (
        "learned factor 0 has zero membership norm; pairwise cosine similarities are 0.0",
    )


def test_support_ties_use_ascending_record_id() -> None:
    row = _support_metrics(
        0,
        0,
        learned_weights=np.array([0.5, 0.5, 0.5, 0.0]),
        planted_weights=np.array([0.0, 1.0, 0.0, 0.0]),
        record_ids=np.array([5, 2, 7, 9]),
    )

    assert row == {
        "learned_factor_id": 0,
        "planted_factor_id": 0,
        "true_support_size": 1,
        "recovered_overlap_count": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "jaccard_similarity": 1.0,
        "analytic_random_support_expectation": 0.25,
    }


def test_support_aggregate_uses_analytic_chance_and_strict_comparison() -> None:
    rows = [
        {"recall": 0.5, "analytic_random_support_expectation": 0.25},
        {"recall": 0.5, "analytic_random_support_expectation": 0.5},
        {"recall": 0.25, "analytic_random_support_expectation": 0.5},
    ]

    aggregate = _aggregate_support_metrics(rows)

    assert aggregate["mean_learned_support_recall"] == pytest.approx(5 / 12)
    assert aggregate["mean_analytic_random_support_expectation"] == pytest.approx(
        5 / 12
    )
    assert aggregate["difference"] == pytest.approx(0.0)
    assert aggregate["factors_above_chance"] == 1
    assert aggregate["factors_equal_to_chance"] == 1
    assert aggregate["factors_below_chance"] == 1


def _write_evaluation_source(run_dir: Path, *, zero_second_signal: bool) -> None:
    run_dir.mkdir()
    hidden = {
        "schema_version": 1,
        "working_set_memberships": [
            {
                "working_set_id": 0,
                "members": [
                    {"record_id": 0, "weight": 0.8},
                    {"record_id": 1, "weight": 0.2},
                ],
            },
            {
                "working_set_id": 1,
                "members": [
                    {"record_id": 2, "weight": 0.7},
                    {"record_id": 3, "weight": 0.3},
                ],
            },
        ],
        "access_source_counts_by_window": [],
    }
    factor_zero = [1, 2, 0]
    factor_one = [0, 0, 0] if zero_second_signal else [0, 1, 3]
    for window_id, (count_zero, count_one) in enumerate(
        zip(factor_zero, factor_one, strict=True)
    ):
        hidden["access_source_counts_by_window"].append(
            {
                "window_id": window_id,
                "working_set_access_counts": [
                    {"working_set_id": 0, "access_count": count_zero},
                    {"working_set_id": 1, "access_count": count_one},
                ],
            }
        )
    for filename, value in (
        ("config.json", {"placeholder": True}),
        ("hidden_ground_truth.json", hidden),
        ("summary.json", {"placeholder": True}),
    ):
        (run_dir / filename).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (run_dir / "observable_events.jsonl").write_text("{}\n", encoding="utf-8")


def _evaluation_inputs(zero_second_signal: bool = False):
    demand = DemandMatrix(
        X=np.array([[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 2, 1]]),
        window_ids=np.arange(3),
        record_ids=np.arange(4),
    )
    factor_one_signal = (
        np.zeros(3) if zero_second_signal else np.array([0.0, 1.0, 3.0])
    )
    learned = LearnedStructure(
        activation_matrix=np.column_stack(
            [factor_one_signal, np.array([1.0, 2.0, 0.0])]
        ),
        membership_matrix=np.array(
            [[0.0, 0.0, 0.7, 0.3], [0.8, 0.2, 0.0, 0.0]]
        ),
        factor_ids=np.arange(2),
        window_ids=np.arange(3),
        record_ids=np.arange(4),
        converged=True,
        iteration_count=7,
        sklearn_reconstruction_error=1.0,
        convergence_warnings=(),
        warnings=(),
    )
    config = StructureLearnerConfig(2, 7, 100, 1e-4)
    return demand, learned, config


def test_recovery_reports_exact_activation_alignment_and_metrics(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_source(run_dir, zero_second_signal=False)
    demand, learned, config = _evaluation_inputs()

    evaluation = evaluate_recovery(run_dir, demand, learned, config)
    report = evaluation.report

    assert [
        (row["learned_factor_id"], row["planted_factor_id"])
        for row in report["optimal_assignment"]
    ] == [(0, 1), (1, 0)]
    assert [
        row["cosine_similarity"] for row in report["optimal_assignment"]
    ] == pytest.approx([1.0, 1.0])
    assert [
        row["cosine_similarity"] for row in report["activation_alignment"]["per_factor"]
    ] == pytest.approx([1.0, 1.0])
    assert report["activation_alignment"]["aggregate"]["mean"] == pytest.approx(
        1.0
    )
    assert report["support_recovery"]["aggregate"]["factors_above_chance"] == 2
    assert report["representative_gate"]["passed"]
    assert set(report["source_artifact_sha256"]) == {
        "config.json",
        "observable_events.jsonl",
        "hidden_ground_truth.json",
        "summary.json",
    }


def test_zero_true_activation_is_null_and_warned(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_source(run_dir, zero_second_signal=True)
    demand, learned, config = _evaluation_inputs(zero_second_signal=True)

    report = evaluate_recovery(run_dir, demand, learned, config).report

    alignment = report["activation_alignment"]
    assert alignment["per_factor"][0]["cosine_similarity"] is None
    assert alignment["aggregate"]["count"] == 1
    assert any("generated zero" in warning for warning in report["warnings"])


def test_all_zero_demand_has_explicit_undefined_normalized_error(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation_source(run_dir, zero_second_signal=False)
    _, learned, config = _evaluation_inputs()
    demand = DemandMatrix(
        X=np.zeros((3, 4), dtype=np.int64),
        window_ids=np.arange(3),
        record_ids=np.arange(4),
    )

    report = evaluate_recovery(run_dir, demand, learned, config).report

    assert report["reconstruction"]["demand_matrix_frobenius_norm"] == 0.0
    assert report["reconstruction"]["normalized_frobenius_error"] is None
    assert any("all-zero demand" in warning for warning in report["warnings"])
