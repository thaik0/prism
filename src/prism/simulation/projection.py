"""Training-only factor calibration and record-demand projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


class ProjectionError(ValueError):
    """Raised when observable projection inputs violate their contract."""


@dataclass(frozen=True)
class ProjectionModel:
    membership_matrix: np.ndarray
    factor_coefficients: np.ndarray
    factor_residual_norms: np.ndarray
    residual_record_baseline: np.ndarray
    training_target_window_ids: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "membership_matrix",
            "factor_coefficients",
            "factor_residual_norms",
            "residual_record_baseline",
            "training_target_window_ids",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ProjectionResult:
    model: ProjectionModel
    observable_factor_demand: np.ndarray
    predicted_factor_demand: np.ndarray
    predicted_record_demand: np.ndarray
    cumulative_future_factor_demand: np.ndarray | None = None
    cumulative_future_record_demand: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "observable_factor_demand",
            "predicted_factor_demand",
            "predicted_record_demand",
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in (
            "cumulative_future_factor_demand",
            "cumulative_future_record_demand",
        ):
            raw = getattr(self, name)
            if raw is not None:
                value = np.array(raw, dtype=np.float64, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)


def fit_record_demand_projection(
    observable_demand: np.ndarray,
    membership_matrix: np.ndarray,
    activation_probability: np.ndarray,
    conditional_intensity: np.ndarray,
    prediction_available: np.ndarray,
    train_end: int,
    *,
    forecast_horizon_windows: int = 1,
    training_target_window_ids: np.ndarray | None = None,
) -> ProjectionResult:
    """Fit one nonnegative calibration per factor using training targets only."""

    X, membership, probability, intensity, available = _validate_inputs(
        observable_demand,
        membership_matrix,
        activation_probability,
        conditional_intensity,
        prediction_available,
        train_end,
    )
    factor_demand = X @ membership.T
    if (
        isinstance(forecast_horizon_windows, bool)
        or not isinstance(forecast_horizon_windows, int)
        or forecast_horizon_windows <= 0
    ):
        raise ProjectionError("forecast_horizon_windows must be a positive integer")
    if training_target_window_ids is None:
        training_targets = np.flatnonzero(available & (np.arange(len(X)) < train_end))
    else:
        training_targets = np.asarray(training_target_window_ids, dtype=np.int64)
        if training_targets.ndim != 1 or not np.all(available[training_targets]):
            raise ProjectionError("explicit training targets must be available")
    if not len(training_targets):
        raise ProjectionError("no available training target windows")
    if np.any(training_targets <= 0):
        raise ProjectionError("training targets require a preceding source window")

    factor_count = membership.shape[0]
    coefficients = np.empty((factor_count, 3), dtype=np.float64)
    residual_norms = np.empty(factor_count, dtype=np.float64)
    activation_intensity = probability * intensity
    cumulative_factor = _cumulative_future(factor_demand, forecast_horizon_windows)
    cumulative_record = _cumulative_future(X, forecast_horizon_windows)
    if np.any(training_targets + forecast_horizon_windows > len(X)):
        raise ProjectionError("training horizon crosses the chronological boundary")
    for factor_id in range(factor_count):
        design = np.column_stack(
            (
                factor_demand[training_targets - 1, factor_id],
                activation_intensity[training_targets, factor_id],
                np.ones(len(training_targets), dtype=np.float64),
            )
        )
        fitted, residual_norm = nnls(
            design, cumulative_factor[training_targets, factor_id]
        )
        coefficients[factor_id] = fitted
        residual_norms[factor_id] = residual_norm

    predicted_factor = _project_factor_demand(
        factor_demand, probability, intensity, available, coefficients
    )
    training_factor_projection = predicted_factor[training_targets] @ membership
    residual_baseline = np.mean(
        np.maximum(
            0.0,
            cumulative_record[training_targets] - training_factor_projection,
        ),
        axis=0,
    )
    model = ProjectionModel(
        membership_matrix=membership,
        factor_coefficients=coefficients,
        factor_residual_norms=residual_norms,
        residual_record_baseline=residual_baseline,
        training_target_window_ids=training_targets.astype(np.int64),
    )
    predicted_record = project_record_demand(predicted_factor, available, model)
    return ProjectionResult(
        model,
        factor_demand,
        predicted_factor,
        predicted_record,
        cumulative_factor,
        cumulative_record,
    )


def _cumulative_future(values: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for start in range(len(values) - horizon + 1):
        result[start] = np.sum(values[start : start + horizon], axis=0)
    return result


def project_record_demand(
    predicted_factor_demand: np.ndarray,
    prediction_available: np.ndarray,
    model: ProjectionModel,
) -> np.ndarray:
    """Project calibrated expected factor demand through memberships to records."""

    factor = np.asarray(predicted_factor_demand, dtype=np.float64)
    available = np.asarray(prediction_available, dtype=np.bool_)
    if factor.ndim != 2 or factor.shape[1] != model.membership_matrix.shape[0]:
        raise ProjectionError("predicted factor demand has incompatible shape")
    if available.shape != (factor.shape[0],):
        raise ProjectionError("prediction availability has incompatible shape")
    projected = np.full(
        (factor.shape[0], model.membership_matrix.shape[1]),
        np.nan,
        dtype=np.float64,
    )
    projected[available] = (
        model.residual_record_baseline
        + factor[available] @ model.membership_matrix
    )
    if np.any(~np.isfinite(projected[available])) or np.any(projected[available] < 0.0):
        raise ProjectionError("record-demand forecasts must be finite and nonnegative")
    return projected


def build_record_demand_variants(
    projection: ProjectionResult,
    activation_probability: np.ndarray,
    conditional_intensity: np.ndarray,
    prediction_available: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mechanically ablate original fitted terms without changing coefficients."""

    probability = np.asarray(activation_probability, dtype=np.float64)
    intensity = np.asarray(conditional_intensity, dtype=np.float64)
    available = np.asarray(prediction_available, dtype=np.bool_)
    factor_demand = projection.observable_factor_demand
    if probability.shape != factor_demand.shape or intensity.shape != factor_demand.shape:
        raise ProjectionError("ablation predictor values have incompatible shape")
    if available.shape != (factor_demand.shape[0],):
        raise ProjectionError("ablation prediction availability has incompatible shape")
    targets = np.flatnonzero(available)
    if np.any(targets <= 0):
        raise ProjectionError("ablation targets require preceding windows")
    coefficients = projection.model.factor_coefficients
    recent_factor = np.full_like(factor_demand, np.nan, dtype=np.float64)
    activation_factor = np.full_like(factor_demand, np.nan, dtype=np.float64)
    residual_factor = np.full_like(factor_demand, np.nan, dtype=np.float64)
    recent_factor[targets] = np.maximum(
        0.0,
        factor_demand[targets - 1] * coefficients[:, 0] + coefficients[:, 2],
    )
    activation_factor[targets] = np.maximum(
        0.0,
        probability[targets]
        * intensity[targets]
        * coefficients[:, 1]
        + coefficients[:, 2],
    )
    residual_factor[targets] = 0.0
    variants = {
        "predictive_greedy": projection.predicted_record_demand,
        "recent_state_only": project_record_demand(
            recent_factor, available, projection.model
        ),
        "activation_intensity_only": project_record_demand(
            activation_factor, available, projection.model
        ),
        "residual_baseline_only": project_record_demand(
            residual_factor, available, projection.model
        ),
    }
    for values in variants.values():
        values.setflags(write=False)
    return variants


def _project_factor_demand(
    factor_demand: np.ndarray,
    probability: np.ndarray,
    intensity: np.ndarray,
    available: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    predicted = np.full_like(factor_demand, np.nan, dtype=np.float64)
    targets = np.flatnonzero(available)
    for factor_id, (a_value, b_value, c_value) in enumerate(coefficients):
        predicted[targets, factor_id] = np.maximum(
            0.0,
            a_value * factor_demand[targets - 1, factor_id]
            + b_value * probability[targets, factor_id] * intensity[targets, factor_id]
            + c_value,
        )
    if np.any(~np.isfinite(predicted[available])) or np.any(predicted[available] < 0.0):
        raise ProjectionError("factor-demand forecasts must be finite and nonnegative")
    return predicted


def _validate_inputs(
    observable_demand: np.ndarray,
    membership_matrix: np.ndarray,
    activation_probability: np.ndarray,
    conditional_intensity: np.ndarray,
    prediction_available: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(observable_demand)
    membership = np.asarray(membership_matrix, dtype=np.float64)
    probability = np.asarray(activation_probability, dtype=np.float64)
    intensity = np.asarray(conditional_intensity, dtype=np.float64)
    available = np.asarray(prediction_available, dtype=np.bool_)
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number):
        raise ProjectionError("observable demand must be a numeric matrix")
    X = np.asarray(X, dtype=np.float64)
    if np.any(~np.isfinite(X)) or np.any(X < 0.0):
        raise ProjectionError("observable demand must be finite and nonnegative")
    if membership.ndim != 2 or membership.shape[1] != X.shape[1]:
        raise ProjectionError("membership matrix has incompatible record dimension")
    expected = (X.shape[0], membership.shape[0])
    if probability.shape != expected or intensity.shape != expected:
        raise ProjectionError("predictor matrices have incompatible dimensions")
    if available.shape != (X.shape[0],):
        raise ProjectionError("prediction availability must be a window vector")
    if (
        np.any(~np.isfinite(membership))
        or np.any(membership < 0.0)
        or np.any(~np.isfinite(probability[available]))
        or np.any((probability[available] < 0.0) | (probability[available] > 1.0))
        or np.any(~np.isfinite(intensity[available]))
    ):
        raise ProjectionError("membership and available predictor values are invalid")
    if isinstance(train_end, bool) or not isinstance(train_end, int):
        raise ProjectionError("train_end must be an integer")
    if train_end <= 1 or train_end > X.shape[0]:
        raise ProjectionError("train_end is outside source windows")
    return X, membership, probability, intensity, available
