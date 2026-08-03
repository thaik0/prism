from __future__ import annotations

import numpy as np

from prism.predictor import (
    PredictorConfig,
    PredictorDataset,
    PredictorTargets,
    fit_fast_predictor,
    run_fast_prediction,
)
from prism.predictor.features import WindowContext


def _dataset() -> PredictorDataset:
    rows = 48
    factors = np.tile(np.array([0, 1]), rows // 2)
    windows = np.repeat(np.arange(2, 26), 2)
    splits = np.repeat(np.array([0] * 16 + [1] * 4 + [2] * 4), 2)
    recent = np.zeros((rows, 10), dtype=np.float64)
    context = np.zeros((rows, 14), dtype=np.float64)
    for row in range(rows):
        factor = factors[row]
        recent[row, :8] = [row % 5, row % 3, row % 2, row % 4, row % 7, 2, 3, 4]
        recent[row, 8 + factor] = 1.0
        context[row, :10] = recent[row]
        context[row, 10 + factor * 2 : 12 + factor * 2] = [
            (row % 4) / 3,
            1 - (row % 4) / 3,
        ]
    window_context = WindowContext(1, 1, 1, np.array([1.0]), np.array([1.0]))
    return PredictorDataset(
        projected_factor_demand=np.zeros((26, 2)),
        recent_features=recent,
        context_features=context,
        recent_feature_names=tuple(f"r{i}" for i in range(10)),
        context_feature_names=tuple(f"c{i}" for i in range(14)),
        recent_continuous_indices=np.arange(8),
        context_continuous_indices=np.array([*range(8), *range(10, 14)]),
        feature_window_ids=windows,
        target_window_ids=windows + 1,
        factor_ids=factors,
        split_codes=splits,
        user_ids=np.array([0]),
        request_types=("request",),
        window_contexts=tuple(window_context for _ in range(26)),
        warnings=(),
    )


def _targets(dataset: PredictorDataset) -> PredictorTargets:
    activation = ((np.arange(dataset.example_count) + dataset.factor_ids) % 3 == 0).astype(
        np.int8
    )
    intensity = np.full(dataset.example_count, np.nan)
    intensity[activation == 1] = 1.5 + 0.1 * (
        np.arange(dataset.example_count)[activation == 1] % 5
    )
    return PredictorTargets(
        activation=activation,
        intensity=intensity,
        eligible=np.arange(dataset.example_count) % 2 == 0,
        realized_next_window_accesses=np.arange(dataset.example_count),
        learned_to_planted=np.array([1, 0]),
        matching_report={},
    )


def _config() -> PredictorConfig:
    return PredictorConfig(11, 0.6, 0.2, 500, 1e-8, 1.0, 5)


def test_fixed_models_use_training_populations_and_are_deterministic() -> None:
    dataset = _dataset()
    targets = _targets(dataset)

    first = fit_fast_predictor(dataset, targets, _config())
    second = fit_fast_predictor(dataset, targets, _config())
    outputs = run_fast_prediction(dataset, first)
    training = dataset.split_codes == 0
    positive_training = training & (targets.activation == 1)

    np.testing.assert_array_equal(
        first.recent_activation.coefficients,
        second.recent_activation.coefficients,
    )
    np.testing.assert_allclose(
        first.recent_activation.preprocessing.means,
        dataset.recent_features[training][:, :8].mean(axis=0),
    )
    np.testing.assert_allclose(
        first.context_intensity.preprocessing.means,
        dataset.context_features[positive_training][
            :, dataset.context_continuous_indices
        ].mean(axis=0),
    )
    for factor_id in range(2):
        factor_training = training & (dataset.factor_ids == factor_id)
        assert first.activation_base_rates[factor_id] == np.mean(
            targets.activation[factor_training]
        )
    assert outputs.context_activation_probability.shape == (dataset.example_count,)
    assert outputs.context_intensity_prediction.shape == (dataset.example_count,)
    assert np.all(np.isfinite(outputs.context_activation_probability))
    assert np.all((outputs.context_activation_probability >= 0.0) & (outputs.context_activation_probability <= 1.0))
    assert first.convergence["recent_activation"]["converged"]
    assert first.convergence["context_activation"]["converged"]


def test_validation_test_labels_and_eligibility_do_not_change_fitted_state() -> None:
    dataset = _dataset()
    original = _targets(dataset)
    held_out = dataset.split_codes != 0
    changed_activation = original.activation.copy()
    changed_activation[held_out] = 1 - changed_activation[held_out]
    changed_intensity = original.intensity.copy()
    changed_intensity[held_out] = 99.0
    changed = PredictorTargets(
        activation=changed_activation,
        intensity=changed_intensity,
        eligible=~original.eligible,
        realized_next_window_accesses=original.realized_next_window_accesses + 100,
        learned_to_planted=original.learned_to_planted[::-1],
        matching_report={"changed": True},
    )

    first_model = fit_fast_predictor(dataset, original, _config())
    changed_model = fit_fast_predictor(dataset, changed, _config())
    first_outputs = run_fast_prediction(dataset, first_model)
    changed_outputs = run_fast_prediction(dataset, changed_model)

    np.testing.assert_array_equal(
        first_model.context_activation.coefficients,
        changed_model.context_activation.coefficients,
    )
    np.testing.assert_array_equal(
        first_model.context_intensity.coefficients,
        changed_model.context_intensity.coefficients,
    )
    np.testing.assert_array_equal(
        first_outputs.context_activation_probability,
        changed_outputs.context_activation_probability,
    )
    np.testing.assert_array_equal(
        first_outputs.context_intensity_prediction,
        changed_outputs.context_intensity_prediction,
    )
    assert int(np.sum(dataset.split_codes == 0)) == 32


def test_validation_test_demand_cannot_change_training_fitted_membership() -> None:
    from prism.structure import DemandMatrix, StructureLearnerConfig, fit_structure

    training = np.array(
        [[6, 1, 0, 0], [5, 1, 0, 0], [0, 0, 5, 1], [0, 0, 6, 1]],
        dtype=np.int64,
    )
    full_a = np.vstack([training, np.zeros((3, 4), dtype=np.int64)])
    full_b = np.vstack([training, np.full((3, 4), 99, dtype=np.int64)])
    config = StructureLearnerConfig(2, 7, 500, 1e-6)

    def fit_training_only(full: np.ndarray):
        return fit_structure(
            DemandMatrix(full[:4], np.arange(4), np.arange(4)), config
        ).membership_matrix

    np.testing.assert_array_equal(fit_training_only(full_a), fit_training_only(full_b))
