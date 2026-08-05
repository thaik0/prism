"""Held-out activation, calibration, intensity, and scientific-gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from prism.predictor.config import PredictorConfig
from prism.predictor.features import PredictorDataset
from prism.predictor.models import FastPredictor, PredictorOutputs
from prism.predictor.targets import PredictorTargets
from prism.workload.config import WorkloadConfig


@dataclass(frozen=True)
class PredictorEvaluation:
    report: dict[str, Any]

    @property
    def all_gates_passed(self) -> bool:
        return bool(self.report["scientific_gates"]["all_passed"])


def evaluate_fast_predictor(
    dataset: PredictorDataset,
    targets: PredictorTargets,
    predictor: FastPredictor,
    outputs: PredictorOutputs,
    config: PredictorConfig,
    source_config: WorkloadConfig,
) -> PredictorEvaluation:
    """Evaluate fixed models without fitting, selection, or threshold tuning."""

    warnings = list(dataset.warnings) + list(predictor.warnings)
    model_predictions = {
        "per_factor_constant": outputs.base_activation_probability,
        "recent_demand_logistic": outputs.recent_activation_probability,
        "context_plus_state_logistic": outputs.context_activation_probability,
    }
    activation_report: dict[str, Any] = {}
    calibration_report: dict[str, Any] = {}
    intensity_report: dict[str, Any] = {}
    for split_name, split_code in (("validation", 1), ("test", 2)):
        split_mask = dataset.split_codes == split_code
        populations = {
            "all_examples": split_mask,
            "hidden_eligible_subset": split_mask & targets.eligible,
        }
        activation_report[split_name] = {}
        for population_name, population_mask in populations.items():
            activation_report[split_name][population_name] = {}
            for model_name, predictions in model_predictions.items():
                activation_report[split_name][population_name][model_name] = (
                    _activation_metrics(
                        targets.activation,
                        predictions,
                        dataset.factor_ids,
                        population_mask,
                        f"{split_name} {population_name} {model_name}",
                        warnings,
                    )
                )
        calibration_report[split_name] = {
            model_name: calibration_table(
                targets.activation[split_mask],
                predictions[split_mask],
                config.calibration_bins,
            )
            for model_name, predictions in model_predictions.items()
        }

        positive_mask = split_mask & (targets.activation == 1)
        intensity_report[split_name] = {
            "per_factor_conditional_mean": _intensity_metrics(
                targets,
                outputs.mean_intensity_prediction,
                dataset.factor_ids,
                positive_mask,
                source_config,
                f"{split_name} per-factor conditional mean",
                warnings,
            ),
            "context_plus_state_ridge": _intensity_metrics(
                targets,
                outputs.context_intensity_prediction,
                dataset.factor_ids,
                positive_mask,
                source_config,
                f"{split_name} context-plus-state ridge",
                warnings,
            ),
        }

    gates = _scientific_gates(activation_report, intensity_report)
    if not gates["all_passed"]:
        _add_warning(warnings, "one or more untouched-test scientific gates failed")
    report = {
        "activation": activation_report,
        "calibration": calibration_report,
        "intensity": intensity_report,
        "scientific_gates": gates,
        "model_convergence": predictor.convergence,
        "warnings": warnings,
    }
    return PredictorEvaluation(report=report)


def calibration_table(
    targets: np.ndarray, predictions: np.ndarray, bin_count: int
) -> list[dict[str, Any]]:
    """Return fixed equal-width bins; the final bin contains probability 1.0."""

    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.float64)
    if len(y) != len(p):
        raise ValueError("calibration inputs must have equal length")
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("calibration predictions must be finite probabilities")
    bin_indices = np.minimum((p * bin_count).astype(np.int64), bin_count - 1)
    rows = []
    for index in range(bin_count):
        mask = bin_indices == index
        count = int(np.sum(mask))
        rows.append(
            {
                "lower_bound": index / bin_count,
                "upper_bound": (index + 1) / bin_count,
                "includes_upper_bound": index == bin_count - 1,
                "count": count,
                "mean_predicted_probability": float(np.mean(p[mask])) if count else None,
                "empirical_activation_rate": float(np.mean(y[mask])) if count else None,
            }
        )
    return rows


def _activation_metrics(
    all_targets: np.ndarray,
    all_predictions: np.ndarray,
    all_factor_ids: np.ndarray,
    mask: np.ndarray,
    metric_name: str,
    warnings: list[str],
) -> dict[str, Any]:
    pooled = _activation_metric_row(
        all_targets[mask], all_predictions[mask], metric_name, warnings
    )
    per_factor = []
    factor_count = int(np.max(all_factor_ids)) + 1
    for factor_id in range(factor_count):
        factor_mask = mask & (all_factor_ids == factor_id)
        per_factor.append(
            {
                "learned_factor_id": factor_id,
                **_activation_metric_row(
                    all_targets[factor_mask],
                    all_predictions[factor_mask],
                    f"{metric_name} factor {factor_id}",
                    warnings,
                ),
            }
        )
    return {"pooled": pooled, "per_factor": per_factor}


def _activation_metric_row(
    targets: np.ndarray,
    predictions: np.ndarray,
    metric_name: str,
    warnings: list[str],
) -> dict[str, Any]:
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.float64)
    count = len(y)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if count == 0:
        _add_warning(warnings, f"{metric_name} metrics are undefined for an empty population")
        return {
            "example_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "activation_rate": None,
            "brier_score": None,
            "binary_log_loss": None,
            "average_precision": None,
            "auroc": None,
        }
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{metric_name} predictions must be finite")
    average_precision: float | None
    auroc: float | None
    if positives == 0:
        average_precision = None
        _add_warning(warnings, f"{metric_name} average precision is undefined with no positives")
    else:
        average_precision = float(average_precision_score(y, p))
    if positives == 0 or negatives == 0:
        auroc = None
        _add_warning(warnings, f"{metric_name} AUROC is undefined without both classes")
    else:
        auroc = float(roc_auc_score(y, p))
    return {
        "example_count": count,
        "positive_count": positives,
        "negative_count": negatives,
        "activation_rate": positives / count,
        "brier_score": float(brier_score_loss(y, p)),
        "binary_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "average_precision": average_precision,
        "auroc": auroc,
    }


def _intensity_metrics(
    targets: PredictorTargets,
    all_predictions: np.ndarray,
    factor_ids: np.ndarray,
    mask: np.ndarray,
    source_config: WorkloadConfig,
    metric_name: str,
    warnings: list[str],
) -> dict[str, Any]:
    observed = targets.intensity[mask]
    predicted = all_predictions[mask]
    realized = targets.realized_next_window_accesses[mask]
    if len(observed) == 0:
        _add_warning(warnings, f"{metric_name} is undefined for no positive activations")
        pooled = {
            "positive_example_count": 0,
            "mae": None,
            "rmse": None,
            "latent_intensity_pearson_correlation": None,
            "realized_access_count_pearson_correlation": None,
        }
    else:
        pooled = {
            "positive_example_count": len(observed),
            "mae": float(mean_absolute_error(observed, predicted)),
            "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
            "latent_intensity_pearson_correlation": _pearson(
                predicted,
                observed,
                f"{metric_name} latent-intensity correlation",
                warnings,
            ),
            "realized_access_count_pearson_correlation": _pearson(
                predicted,
                realized,
                f"{metric_name} realized-access correlation",
                warnings,
            ),
        }
    per_factor = []
    factor_count = int(np.max(factor_ids)) + 1
    for factor_id in range(factor_count):
        factor_mask = mask & (factor_ids == factor_id)
        factor_observed = targets.intensity[factor_mask]
        factor_predicted = all_predictions[factor_mask]
        if len(factor_observed):
            mae = float(mean_absolute_error(factor_observed, factor_predicted))
            rmse = float(math.sqrt(mean_squared_error(factor_observed, factor_predicted)))
        else:
            mae = None
            rmse = None
            _add_warning(
                warnings,
                f"{metric_name} factor {factor_id} has no positive examples",
            )
        per_factor.append(
            {
                "learned_factor_id": factor_id,
                "positive_example_count": len(factor_observed),
                "mae": mae,
                "rmse": rmse,
            }
        )
    below = int(np.sum(predicted < source_config.burst_intensity_min))
    above = int(np.sum(predicted > source_config.burst_intensity_max))
    if below:
        _add_warning(warnings, f"{metric_name} has {below} prediction(s) below intensity bounds")
    if above:
        _add_warning(warnings, f"{metric_name} has {above} prediction(s) above intensity bounds")
    return {
        "pooled": pooled,
        "per_factor": per_factor,
        "raw_prediction_bounds": {
            "configured_minimum": source_config.burst_intensity_min,
            "configured_maximum": source_config.burst_intensity_max,
            "below_minimum_count": below,
            "above_maximum_count": above,
            "minimum_prediction": float(np.min(predicted)) if len(predicted) else None,
            "maximum_prediction": float(np.max(predicted)) if len(predicted) else None,
        },
        "realized_access_correlation_note": (
            "Predicted latent intensity and realized working-set access count have "
            "different units; correlation is diagnostic, not a regression loss."
        ),
    }


def _pearson(
    left: np.ndarray, right: np.ndarray, metric_name: str, warnings: list[str]
) -> float | None:
    if len(left) < 2:
        _add_warning(warnings, f"{metric_name} is undefined with fewer than two examples")
        return None
    left_dev = left - np.mean(left)
    right_dev = right - np.mean(right)
    denominator = float(np.linalg.norm(left_dev) * np.linalg.norm(right_dev))
    if denominator == 0.0:
        _add_warning(warnings, f"{metric_name} is undefined because an input has zero variance")
        return None
    value = float(np.dot(left_dev, right_dev) / denominator)
    return min(1.0, max(-1.0, value))


def _scientific_gates(
    activation: dict[str, Any], intensity: dict[str, Any]
) -> dict[str, Any]:
    test_all = activation["test"]["all_examples"]
    test_eligible = activation["test"]["hidden_eligible_subset"]
    base = test_all["per_factor_constant"]["pooled"]
    context_all = test_all["context_plus_state_logistic"]["pooled"]
    recent_eligible = test_eligible["recent_demand_logistic"]["pooled"]
    context_eligible = test_eligible["context_plus_state_logistic"]["pooled"]
    mean_intensity = intensity["test"]["per_factor_conditional_mean"]["pooled"]
    ridge_intensity = intensity["test"]["context_plus_state_ridge"]["pooled"]
    gate1 = _activation_gate(context_all, base, "context", "constant")
    gate2 = _activation_gate(context_eligible, recent_eligible, "context", "recent")
    intensity_valid = (
        ridge_intensity["positive_example_count"] > 0
        and ridge_intensity["rmse"] is not None
        and mean_intensity["rmse"] is not None
        and math.isfinite(ridge_intensity["rmse"])
        and math.isfinite(mean_intensity["rmse"])
    )
    gate3 = {
        "name": "useful conditional-intensity prediction",
        "context_ridge_rmse": ridge_intensity["rmse"],
        "per_factor_mean_rmse": mean_intensity["rmse"],
        "improvement": (
            mean_intensity["rmse"] - ridge_intensity["rmse"]
            if intensity_valid
            else None
        ),
        "population_valid": intensity_valid,
        "passed": bool(
            intensity_valid
            and ridge_intensity["rmse"] < mean_intensity["rmse"]
        ),
    }
    return {
        "gate_1": gate1,
        "gate_2": gate2,
        "gate_3": gate3,
        "all_passed": bool(gate1["passed"] and gate2["passed"] and gate3["passed"]),
    }


def _activation_gate(
    candidate: dict[str, Any], baseline: dict[str, Any], candidate_name: str, baseline_name: str
) -> dict[str, Any]:
    valid = (
        candidate["example_count"] > 0
        and candidate["positive_count"] > 0
        and candidate["negative_count"] > 0
        and candidate["brier_score"] is not None
        and baseline["brier_score"] is not None
        and math.isfinite(candidate["brier_score"])
        and math.isfinite(baseline["brier_score"])
    )
    return {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "candidate_brier_score": candidate["brier_score"],
        "baseline_brier_score": baseline["brier_score"],
        "improvement": (
            baseline["brier_score"] - candidate["brier_score"] if valid else None
        ),
        "population_valid": valid,
        "passed": bool(valid and candidate["brier_score"] < baseline["brier_score"]),
    }


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)
