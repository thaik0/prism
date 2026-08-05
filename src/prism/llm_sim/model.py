"""Training-only reuse of Prism's accepted structure, predictor, and projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prism.predictor.config import PredictorConfig, SplitBoundaries
from prism.predictor.features import PredictorDataset, WindowContext
from prism.predictor.models import fit_fast_predictor, run_fast_prediction
from prism.predictor.targets import PredictorTargets
from prism.simulation.projection import ProjectionResult, fit_record_demand_projection
from prism.structure.config import StructureLearnerConfig
from prism.structure.demand import DemandMatrix
from prism.structure.learner import LearnedStructure, fit_structure

from .config import IntegrationConfig, RequestSplit
from .demand import LogicalDemand


class LLMModelFitError(ValueError):
    """Raised when the accepted model contract cannot fit the real trace safely."""


@dataclass(frozen=True)
class DemandForecasts:
    predicted_record_demand: np.ndarray
    prediction_available: np.ndarray
    learned_membership: np.ndarray
    training_seen_indices: np.ndarray
    activation_threshold: np.ndarray
    fit_status: str
    structure_converged: bool
    structure_iterations: int

    def __post_init__(self) -> None:
        for name in (
            "predicted_record_demand",
            "prediction_available",
            "learned_membership",
            "training_seen_indices",
            "activation_threshold",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def fit_demand_forecasts(
    demand: LogicalDemand,
    split: RequestSplit,
    config: IntegrationConfig,
    *,
    allow_degenerate_smoke: bool = False,
) -> DemandForecasts:
    """Fit only training rows and emit one-window block forecasts.

    Observable factor activation is the accepted positive-demand condition.
    Conditional intensity is factor demand when active. No hidden label or
    future policy outcome enters this fit.
    """
    seen = np.flatnonzero(demand.training_seen)
    if len(seen) < config.structure_components:
        if allow_degenerate_smoke:
            return _degenerate_forecasts(demand, config.structure_components)
        raise LLMModelFitError(
            "training trace has fewer seen blocks than the accepted factor count"
        )
    train_matrix = DemandMatrix(
        X=demand.demand_matrix[: split.training_end, seen],
        window_ids=np.arange(split.training_end, dtype=np.int64),
        record_ids=np.arange(len(seen), dtype=np.int64),
    )
    learner_config = StructureLearnerConfig.from_json(
        config.project_root / "configs/milestone2_representative.json"
    )
    if learner_config.n_components != config.structure_components:
        raise LLMModelFitError("accepted learner factor count differs from integration config")
    structure = fit_structure(train_matrix, learner_config)
    if not structure.converged:
        raise LLMModelFitError(
            f"training-only structure fit did not converge: {structure.convergence_warnings}"
        )

    full_seen = DemandMatrix(
        X=demand.demand_matrix[:, seen],
        window_ids=np.arange(split.request_count, dtype=np.int64),
        record_ids=np.arange(len(seen), dtype=np.int64),
    )
    predictor_config = PredictorConfig.from_json(
        config.project_root / "configs/milestone3_predictor.json"
    )
    boundaries = predictor_config.split_boundaries(split.request_count)
    if (boundaries.train_end, boundaries.validation_end) != (
        split.training_end,
        split.validation_end,
    ):
        raise LLMModelFitError("accepted predictor fractions do not reproduce the frozen split")
    try:
        dataset = _build_neutral_dataset(full_seen, structure, boundaries)
        targets, thresholds = _observable_targets(dataset, split.training_end)
        predictor = fit_fast_predictor(dataset, targets, predictor_config)
    except ValueError as exc:
        if allow_degenerate_smoke:
            return _degenerate_forecasts(demand, config.structure_components)
        raise LLMModelFitError(f"accepted predictor fit failed: {exc}") from exc
    outputs = run_fast_prediction(dataset, predictor)
    probability, intensity, available = _window_predictions(dataset, outputs)
    projection = fit_record_demand_projection(
        observable_demand=full_seen.X,
        membership_matrix=structure.membership_matrix,
        activation_probability=probability,
        conditional_intensity=intensity,
        prediction_available=available,
        train_end=split.training_end,
    )
    expanded = np.zeros(demand.demand_matrix.shape, dtype=np.float64)
    expanded[:, seen] = np.nan_to_num(
        projection.predicted_record_demand, nan=0.0, posinf=0.0, neginf=0.0
    )
    expanded[:, demand.cold_start] = 0.0
    return DemandForecasts(
        predicted_record_demand=expanded,
        prediction_available=available,
        learned_membership=structure.membership_matrix,
        training_seen_indices=seen,
        activation_threshold=thresholds,
        fit_status="fitted_training_only",
        structure_converged=structure.converged,
        structure_iterations=structure.iteration_count,
    )


def _build_neutral_dataset(
    demand: DemandMatrix,
    structure: LearnedStructure,
    boundaries: SplitBoundaries,
) -> PredictorDataset:
    factor_demand = np.asarray(demand.X, dtype=np.float64) @ structure.membership_matrix.T
    factors = factor_demand.shape[1]
    recent_names = (
        "factor_demand_t",
        "factor_demand_t_minus_1",
        "factor_demand_t_minus_2",
        "factor_demand_delta_1",
        "factor_demand_mean_3",
        "session_count_t",
        "request_count_t",
        "access_count_t",
        *(f"factor_id_{factor_id}" for factor_id in range(factors)),
    )
    context_names = (
        *recent_names,
        *(f"factor_user_fraction_factor_{factor_id}_user_0" for factor_id in range(factors)),
        *(
            f"factor_request_type_fraction_factor_{factor_id}_request_type_llm_request"
            for factor_id in range(factors)
        ),
    )
    rows = (boundaries.num_windows - 3) * factors
    recent = np.zeros((rows, len(recent_names)), dtype=np.float64)
    context = np.zeros((rows, len(context_names)), dtype=np.float64)
    feature_windows = np.empty(rows, dtype=np.int64)
    target_windows = np.empty(rows, dtype=np.int64)
    factor_ids = np.empty(rows, dtype=np.int64)
    split_codes = np.empty(rows, dtype=np.int8)
    contexts = tuple(
        WindowContext(
            session_count=1,
            request_count=1,
            access_count=int(demand.X[window].sum()),
            user_fractions=np.ones(1, dtype=np.float64),
            request_type_fractions=np.ones(1, dtype=np.float64),
        )
        for window in range(boundaries.num_windows)
    )
    row = 0
    user_start = len(recent_names)
    type_start = user_start + factors
    for feature_window in range(2, boundaries.num_windows - 1):
        for factor_id in range(factors):
            d0 = factor_demand[feature_window, factor_id]
            d1 = factor_demand[feature_window - 1, factor_id]
            d2 = factor_demand[feature_window - 2, factor_id]
            recent[row, :8] = (
                d0, d1, d2, d0 - d1, (d0 + d1 + d2) / 3.0,
                1.0, 1.0, float(contexts[feature_window].access_count),
            )
            recent[row, 8 + factor_id] = 1.0
            context[row, : len(recent_names)] = recent[row]
            context[row, user_start + factor_id] = 1.0
            context[row, type_start + factor_id] = 1.0
            target = feature_window + 1
            feature_windows[row] = feature_window
            target_windows[row] = target
            factor_ids[row] = factor_id
            split_codes[row] = boundaries.split_code_for_target(target)
            row += 1
    continuous_recent = np.arange(8, dtype=np.int64)
    continuous_context = np.concatenate(
        (continuous_recent, np.arange(len(recent_names), len(context_names), dtype=np.int64))
    )
    return PredictorDataset(
        projected_factor_demand=factor_demand,
        recent_features=recent,
        context_features=context,
        recent_feature_names=recent_names,
        context_feature_names=context_names,
        recent_continuous_indices=continuous_recent,
        context_continuous_indices=continuous_context,
        feature_window_ids=feature_windows,
        target_window_ids=target_windows,
        factor_ids=factor_ids,
        split_codes=split_codes,
        user_ids=np.asarray([0], dtype=np.int64),
        request_types=("llm_request",),
        window_contexts=contexts,
        warnings=("anonymous user and llm_request context are neutral constants",),
    )


def _observable_targets(
    dataset: PredictorDataset, training_end: int
) -> tuple[PredictorTargets, np.ndarray]:
    values = dataset.projected_factor_demand[
        dataset.target_window_ids, dataset.factor_ids
    ]
    # Zero is the fixed, trace-native dormant boundary; no quantile or
    # test-informed threshold is selected.
    thresholds = np.zeros(dataset.projected_factor_demand.shape[1], dtype=np.float64)
    activation = (values > thresholds[dataset.factor_ids]).astype(np.int8)
    intensity = np.where(activation == 1, values, np.nan)
    training = dataset.target_window_ids < training_end
    if len(np.unique(activation[training])) != 2:
        raise LLMModelFitError("observable training activation has only one class")
    targets = PredictorTargets(
        activation=activation,
        intensity=intensity,
        eligible=np.ones(dataset.example_count, dtype=np.bool_),
        realized_next_window_accesses=values,
        learned_to_planted=np.arange(
            dataset.projected_factor_demand.shape[1], dtype=np.int64
        ),
        matching_report={"source": "observable_positive_factor_demand"},
    )
    return targets, thresholds


def _window_predictions(dataset: PredictorDataset, outputs):
    windows, factors = dataset.projected_factor_demand.shape
    probability = np.full((windows, factors), np.nan, dtype=np.float64)
    intensity = np.full((windows, factors), np.nan, dtype=np.float64)
    for row, (target, factor) in enumerate(
        zip(dataset.target_window_ids, dataset.factor_ids, strict=True)
    ):
        probability[target, factor] = outputs.context_activation_probability[row]
        intensity[target, factor] = outputs.context_intensity_prediction[row]
    available = np.all(np.isfinite(probability) & np.isfinite(intensity), axis=1)
    return probability, intensity, available


def _degenerate_forecasts(
    demand: LogicalDemand, factor_count: int
) -> DemandForecasts:
    return DemandForecasts(
        predicted_record_demand=np.zeros(demand.demand_matrix.shape, dtype=np.float64),
        prediction_available=np.zeros(len(demand.request_ids), dtype=np.bool_),
        learned_membership=np.zeros((factor_count, len(demand.block_ids)), dtype=np.float64),
        training_seen_indices=np.flatnonzero(demand.training_seen),
        activation_threshold=np.zeros(factor_count, dtype=np.float64),
        fit_status="degenerate_tiny_smoke_no_predictive_target",
        structure_converged=False,
        structure_iterations=0,
    )
