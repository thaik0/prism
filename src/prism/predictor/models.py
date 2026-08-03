"""Fixed activation and conditional-intensity models for Milestone 3."""

from __future__ import annotations

from dataclasses import dataclass
import warnings as python_warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from prism.predictor.config import PredictorConfig
from prism.predictor.features import PredictorDataset
from prism.predictor.targets import PredictorTargets


class PredictorFitError(ValueError):
    """Raised when the fixed supervised models cannot be fit safely."""


@dataclass(frozen=True)
class PreprocessorState:
    continuous_indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        for name in ("continuous_indices", "means", "scales"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LinearModelState:
    coefficients: np.ndarray
    intercept: np.ndarray
    preprocessing: PreprocessorState

    def __post_init__(self) -> None:
        for name in ("coefficients", "intercept"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FastPredictor:
    recent_activation: LinearModelState
    context_activation: LinearModelState
    context_intensity: LinearModelState
    activation_base_rates: np.ndarray
    intensity_factor_means: np.ndarray
    intensity_factor_available: np.ndarray
    global_intensity_mean: float
    convergence: dict[str, object]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "activation_base_rates",
            "intensity_factor_means",
            "intensity_factor_available",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def converged(self) -> bool:
        return bool(
            self.convergence["recent_activation"]["converged"]
            and self.convergence["context_activation"]["converged"]
        )


@dataclass(frozen=True)
class PredictorOutputs:
    base_activation_probability: np.ndarray
    recent_activation_probability: np.ndarray
    context_activation_probability: np.ndarray
    mean_intensity_prediction: np.ndarray
    context_intensity_prediction: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "base_activation_probability",
            "recent_activation_probability",
            "context_activation_probability",
            "mean_intensity_prediction",
            "context_intensity_prediction",
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def fit_fast_predictor(
    dataset: PredictorDataset,
    targets: PredictorTargets,
    config: PredictorConfig,
) -> FastPredictor:
    """Fit exactly two logistic models and one positive-population ridge model."""

    training = dataset.split_codes == 0
    if not np.any(training):
        raise PredictorFitError("training split has no examples")
    y_train = np.asarray(targets.activation[training], dtype=np.int64)
    if len(np.unique(y_train)) != 2:
        raise PredictorFitError("training activation targets must contain both classes")
    factor_train = dataset.factor_ids[training]
    factor_count = dataset.projected_factor_demand.shape[1]

    activation_rates = np.empty(factor_count, dtype=np.float64)
    for factor_id in range(factor_count):
        mask = factor_train == factor_id
        if not np.any(mask):
            raise PredictorFitError(f"factor {factor_id} has no training examples")
        activation_rates[factor_id] = float(np.mean(y_train[mask]))

    recent_state, recent_metadata = _fit_logistic(
        dataset.recent_features[training],
        y_train,
        dataset.recent_continuous_indices,
        config,
        "recent_activation",
    )
    context_state, context_metadata = _fit_logistic(
        dataset.context_features[training],
        y_train,
        dataset.context_continuous_indices,
        config,
        "context_activation",
    )

    positive_training = training & (targets.activation == 1)
    if not np.any(positive_training):
        raise PredictorFitError("conditional-intensity training population is empty")
    intensity_targets = targets.intensity[positive_training]
    if not np.all(np.isfinite(intensity_targets)):
        raise PredictorFitError("positive training intensity targets must be finite")
    global_mean = float(np.mean(intensity_targets))
    factor_means = np.empty(factor_count, dtype=np.float64)
    factor_available = np.ones(factor_count, dtype=np.bool_)
    fit_warnings: list[str] = []
    for factor_id in range(factor_count):
        mask = positive_training & (dataset.factor_ids == factor_id)
        if np.any(mask):
            factor_means[factor_id] = float(np.mean(targets.intensity[mask]))
        else:
            factor_means[factor_id] = global_mean
            factor_available[factor_id] = False
            fit_warnings.append(
                f"factor {factor_id} has no positive training activation; "
                "the global conditional mean is used"
            )

    intensity_preprocessing, transformed_positive = _fit_preprocessor(
        dataset.context_features[positive_training],
        dataset.context_continuous_indices,
    )
    ridge = Ridge(alpha=config.intensity_ridge_alpha, fit_intercept=True)
    ridge.fit(transformed_positive, intensity_targets)
    intensity_state = LinearModelState(
        coefficients=np.asarray(ridge.coef_, dtype=np.float64).reshape(1, -1),
        intercept=np.asarray([ridge.intercept_], dtype=np.float64),
        preprocessing=intensity_preprocessing,
    )
    return FastPredictor(
        recent_activation=recent_state,
        context_activation=context_state,
        context_intensity=intensity_state,
        activation_base_rates=activation_rates,
        intensity_factor_means=factor_means,
        intensity_factor_available=factor_available,
        global_intensity_mean=global_mean,
        convergence={
            "recent_activation": recent_metadata,
            "context_activation": context_metadata,
            "context_intensity": {"converged": True, "algorithm": "closed_form_ridge"},
        },
        warnings=tuple(fit_warnings),
    )


def run_fast_prediction(
    dataset: PredictorDataset, predictor: FastPredictor
) -> PredictorOutputs:
    """Generate all fixed-model outputs for every usable factor-window example."""

    base_probability = predictor.activation_base_rates[dataset.factor_ids]
    recent_probability = _logistic_predict(
        dataset.recent_features, predictor.recent_activation
    )
    context_probability = _logistic_predict(
        dataset.context_features, predictor.context_activation
    )
    mean_intensity = predictor.intensity_factor_means[dataset.factor_ids]
    intensity = _linear_predict(dataset.context_features, predictor.context_intensity)
    for name, values in (
        ("base activation", base_probability),
        ("recent activation", recent_probability),
        ("context activation", context_probability),
    ):
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise PredictorFitError(f"{name} probabilities must be finite and in [0, 1]")
    if not np.all(np.isfinite(mean_intensity)) or not np.all(np.isfinite(intensity)):
        raise PredictorFitError("intensity predictions must be finite")
    return PredictorOutputs(
        base_activation_probability=base_probability,
        recent_activation_probability=recent_probability,
        context_activation_probability=context_probability,
        mean_intensity_prediction=mean_intensity,
        context_intensity_prediction=intensity,
    )


def transform_with_preprocessor(
    features: np.ndarray, state: PreprocessorState
) -> np.ndarray:
    """Apply persisted train-only means and scales without an estimator object."""

    transformed = np.array(features, dtype=np.float64, copy=True)
    indices = state.continuous_indices
    transformed[:, indices] = (
        transformed[:, indices] - state.means
    ) / state.scales
    return transformed


def _fit_preprocessor(
    features: np.ndarray, continuous_indices: np.ndarray
) -> tuple[PreprocessorState, np.ndarray]:
    indices = np.asarray(continuous_indices, dtype=np.int64)
    scaler = StandardScaler()
    continuous = scaler.fit_transform(np.asarray(features[:, indices], dtype=np.float64))
    transformed = np.array(features, dtype=np.float64, copy=True)
    transformed[:, indices] = continuous
    state = PreprocessorState(
        continuous_indices=indices,
        means=np.asarray(scaler.mean_, dtype=np.float64),
        scales=np.asarray(scaler.scale_, dtype=np.float64),
    )
    return state, transformed


def _fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    continuous_indices: np.ndarray,
    config: PredictorConfig,
    name: str,
) -> tuple[LinearModelState, dict[str, object]]:
    preprocessing, transformed = _fit_preprocessor(features, continuous_indices)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight=None,
        max_iter=config.activation_max_iter,
        tol=config.activation_tolerance,
        random_state=config.fit_seed,
    )
    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always", ConvergenceWarning)
        model.fit(transformed, targets)
    convergence_messages = tuple(
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    )
    iteration_count = int(np.max(model.n_iter_))
    reached_limit = iteration_count >= config.activation_max_iter
    if reached_limit and not convergence_messages:
        convergence_messages = (
            f"{name} reached the configured iteration limit "
            f"({config.activation_max_iter})",
        )
    state = LinearModelState(
        coefficients=np.asarray(model.coef_, dtype=np.float64),
        intercept=np.asarray(model.intercept_, dtype=np.float64),
        preprocessing=preprocessing,
    )
    metadata: dict[str, object] = {
        "converged": not convergence_messages and not reached_limit,
        "iteration_count": iteration_count,
        "max_iter": config.activation_max_iter,
        "convergence_warnings": list(convergence_messages),
    }
    return state, metadata


def _linear_predict(features: np.ndarray, state: LinearModelState) -> np.ndarray:
    transformed = transform_with_preprocessor(features, state.preprocessing)
    return np.asarray(
        transformed @ state.coefficients.reshape(-1) + state.intercept[0],
        dtype=np.float64,
    )


def _logistic_predict(features: np.ndarray, state: LinearModelState) -> np.ndarray:
    scores = _linear_predict(features, state)
    probabilities = np.empty_like(scores)
    nonnegative = scores >= 0.0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
    exp_scores = np.exp(scores[~nonnegative])
    probabilities[~nonnegative] = exp_scores / (1.0 + exp_scores)
    return probabilities
