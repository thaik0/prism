from __future__ import annotations

import numpy as np

from prism.simulation import fit_record_demand_projection, project_record_demand


def _inputs():
    membership = np.array([[1.0, 0.0], [0.0, 1.0]])
    demand = np.array(
        [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64]],
        dtype=np.int64,
    )
    probability = np.zeros((6, 2), dtype=np.float64)
    intensity = np.zeros((6, 2), dtype=np.float64)
    available = np.array([False, True, True, True, True, True])
    return demand, membership, probability, intensity, available


def test_nnls_calibration_formula_residual_baseline_and_projection() -> None:
    demand, membership, probability, intensity, available = _inputs()

    result = fit_record_demand_projection(
        demand, membership, probability, intensity, available, train_end=4
    )

    np.testing.assert_array_equal(result.model.training_target_window_ids, [1, 2, 3])
    np.testing.assert_allclose(result.model.factor_coefficients[:, 0], [2.0, 2.0])
    assert np.all(result.model.factor_coefficients >= 0.0)
    np.testing.assert_allclose(
        result.model.residual_record_baseline, [0.0, 0.0], atol=1e-14
    )
    np.testing.assert_allclose(result.predicted_factor_demand[4], [16.0, 32.0])
    np.testing.assert_allclose(result.predicted_record_demand[5], [32.0, 64.0])
    assert np.all(np.isnan(result.predicted_record_demand[~available]))


def test_validation_and_test_mutations_do_not_change_fitted_projection() -> None:
    demand, membership, probability, intensity, available = _inputs()
    changed = demand.copy()
    changed[4:] = 999

    original = fit_record_demand_projection(
        demand, membership, probability, intensity, available, train_end=4
    )
    mutated = fit_record_demand_projection(
        changed, membership, probability, intensity, available, train_end=4
    )

    np.testing.assert_array_equal(
        original.model.factor_coefficients, mutated.model.factor_coefficients
    )
    np.testing.assert_array_equal(
        original.model.factor_residual_norms, mutated.model.factor_residual_norms
    )
    np.testing.assert_array_equal(
        original.model.residual_record_baseline,
        mutated.model.residual_record_baseline,
    )


def test_activation_intensity_term_and_final_nonnegative_clamp_are_exact() -> None:
    demand = np.array([[0], [0], [2], [0]], dtype=np.int64)
    membership = np.array([[1.0]])
    probability = np.array([[0.0], [0.5], [1.0], [1.0]])
    intensity = np.array([[0.0], [2.0], [2.0], [-100.0]])
    available = np.array([False, True, True, True])

    result = fit_record_demand_projection(
        demand, membership, probability, intensity, available, train_end=3
    )

    assert np.all(result.model.factor_coefficients >= 0.0)
    assert result.predicted_factor_demand[3, 0] == 0.0
    projected = project_record_demand(
        result.predicted_factor_demand, available, result.model
    )
    assert np.all(projected[available] >= 0.0)


def test_repeated_fitting_is_deterministic() -> None:
    inputs = _inputs()
    first = fit_record_demand_projection(*inputs, train_end=4)
    second = fit_record_demand_projection(*inputs, train_end=4)
    np.testing.assert_array_equal(
        first.model.factor_coefficients, second.model.factor_coefficients
    )
    np.testing.assert_array_equal(
        first.predicted_record_demand, second.predicted_record_demand
    )
