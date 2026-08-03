from __future__ import annotations

import numpy as np
import pytest

from prism.structure import (
    DemandMatrix,
    StructureLearnerConfig,
    fit_structure,
    normalize_factors,
)


def _demand() -> DemandMatrix:
    return DemandMatrix(
        X=np.array(
            [
                [6, 1, 0, 0],
                [5, 1, 0, 0],
                [0, 0, 5, 1],
                [0, 0, 6, 1],
            ],
            dtype=np.int64,
        ),
        window_ids=np.arange(4),
        record_ids=np.arange(4),
    )


def _config(**overrides: object) -> StructureLearnerConfig:
    raw = {
        "n_components": 2,
        "fit_seed": 7,
        "max_iter": 500,
        "tolerance": 1e-6,
    }
    raw.update(overrides)
    return StructureLearnerConfig.from_dict(raw)


def test_fit_shapes_nonnegativity_normalization_and_metadata() -> None:
    result = fit_structure(_demand(), _config())

    assert result.activation_matrix.shape == (4, 2)
    assert result.membership_matrix.shape == (2, 4)
    assert result.factor_ids.tolist() == [0, 1]
    assert np.all(result.activation_matrix >= 0.0)
    assert np.all(result.membership_matrix >= 0.0)
    assert result.membership_matrix.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert result.converged
    assert 0 < result.iteration_count < 500
    assert result.convergence_warnings == ()


def test_fit_is_numerically_deterministic() -> None:
    first = fit_structure(_demand(), _config())
    second = fit_structure(_demand(), _config())

    assert np.array_equal(first.activation_matrix, second.activation_matrix)
    assert np.array_equal(first.membership_matrix, second.membership_matrix)
    assert first.converged == second.converged
    assert first.iteration_count == second.iteration_count
    assert first.sklearn_reconstruction_error == second.sklearn_reconstruction_error
    assert first.convergence_warnings == second.convergence_warnings


def test_normalization_preserves_reconstruction() -> None:
    activation_raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    membership_raw = np.array([[2.0, 1.0, 1.0], [0.5, 0.5, 1.0]])

    activation, membership, warnings = normalize_factors(
        activation_raw, membership_raw
    )

    assert membership.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert activation @ membership == pytest.approx(
        activation_raw @ membership_raw
    )
    assert warnings == ()


def test_zero_sum_factor_is_reported_without_division() -> None:
    activation_raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    membership_raw = np.array([[1.0, 1.0], [0.0, 0.0]])

    activation, membership, warnings = normalize_factors(
        activation_raw, membership_raw
    )

    assert membership[0].sum() == pytest.approx(1.0)
    assert membership[1].tolist() == [0.0, 0.0]
    assert activation @ membership == pytest.approx(
        activation_raw @ membership_raw
    )
    assert warnings == (
        "learned factor 1 has zero-sum membership and was left unnormalized",
    )


def test_iteration_limit_is_explicit_nonconvergence() -> None:
    result = fit_structure(
        _demand(), _config(max_iter=1, tolerance=1e-12)
    )

    assert not result.converged
    assert result.iteration_count == 1
    assert result.convergence_warnings
