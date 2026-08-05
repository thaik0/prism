from __future__ import annotations

import numpy as np
import pytest

from prism.predictor.evaluate import (
    _activation_metric_row,
    _scientific_gates,
    calibration_table,
)


def test_activation_metrics_and_calibration_boundaries_are_exact() -> None:
    warnings: list[str] = []
    metrics = _activation_metric_row(
        np.array([0, 1]), np.array([0.25, 0.75]), "manual", warnings
    )

    assert metrics["brier_score"] == pytest.approx(0.0625)
    assert metrics["binary_log_loss"] == pytest.approx(-np.log(0.75))
    assert metrics["average_precision"] == 1.0
    assert metrics["auroc"] == 1.0
    assert warnings == []

    table = calibration_table(np.array([0, 1, 1]), np.array([0.0, 0.5, 1.0]), 2)
    assert table[0]["count"] == 1
    assert table[0]["empirical_activation_rate"] == 0.0
    assert table[1]["count"] == 2
    assert table[1]["mean_predicted_probability"] == 0.75
    assert table[1]["includes_upper_bound"]


def test_undefined_activation_metrics_are_null_and_warned() -> None:
    warnings: list[str] = []

    row = _activation_metric_row(
        np.array([0, 0]), np.array([0.1, 0.2]), "single class", warnings
    )

    assert row["average_precision"] is None
    assert row["auroc"] is None
    assert row["brier_score"] is not None
    assert len(warnings) == 2


def test_all_three_scientific_gates_use_strict_untouched_test_comparisons() -> None:
    def activation_row(score: float):
        return {
            "pooled": {
                "example_count": 10,
                "positive_count": 4,
                "negative_count": 6,
                "brier_score": score,
            }
        }

    activation = {
        "test": {
            "all_examples": {
                "per_factor_constant": activation_row(0.20),
                "context_plus_state_logistic": activation_row(0.18),
            },
            "hidden_eligible_subset": {
                "recent_demand_logistic": activation_row(0.30),
                "context_plus_state_logistic": activation_row(0.29),
            },
        }
    }
    intensity = {
        "test": {
            "per_factor_conditional_mean": {
                "pooled": {"positive_example_count": 4, "rmse": 0.5}
            },
            "context_plus_state_ridge": {
                "pooled": {"positive_example_count": 4, "rmse": 0.4}
            },
        }
    }

    gates = _scientific_gates(activation, intensity)

    assert gates["all_passed"]
    assert gates["gate_1"]["improvement"] == pytest.approx(0.02)
    assert gates["gate_2"]["improvement"] == pytest.approx(0.01)
    assert gates["gate_3"]["improvement"] == pytest.approx(0.1)
    activation["test"]["all_examples"]["context_plus_state_logistic"] = activation_row(0.20)
    assert not _scientific_gates(activation, intensity)["gate_1"]["passed"]
