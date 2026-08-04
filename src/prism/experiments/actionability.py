"""Matched-horizon foundations for Milestone 5.5 actionability experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from prism.predictor.config import PredictorConfig
from prism.predictor.evaluate import PredictorEvaluation, evaluate_fast_predictor
from prism.predictor.features import PredictorDataset
from prism.predictor.features import build_predictor_dataset
from prism.predictor.models import (
    FastPredictor,
    PredictorOutputs,
    fit_fast_predictor,
    run_fast_prediction,
)
from prism.predictor.persistence import _write_bundle, _write_predictions
from prism.predictor.targets import PredictorTargets, build_predictor_targets
from prism.structure.config import StructureLearnerConfig
from prism.structure.demand import DemandMatrix, build_demand_matrix
from prism.structure.learner import LearnedStructure, fit_structure
from prism.workload.config import WorkloadConfig


@dataclass(frozen=True)
class CommonWindowProtocol:
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    trace_end: int
    h_max: int = 4

    def __post_init__(self) -> None:
        values = (
            self.train_end,
            self.validation_start,
            self.validation_end,
            self.test_start,
            self.trace_end,
            self.h_max,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("common-window values must be integers")
        if not (
            0 < self.train_end == self.validation_start
            < self.validation_end == self.test_start
            < self.trace_end
        ):
            raise ValueError("common chronological boundaries are invalid")
        if self.h_max <= 0:
            raise ValueError("h_max must be positive")

    @property
    def training_target_end(self) -> int:
        return self.train_end - self.h_max + 1

    @property
    def validation_evaluation_end(self) -> int:
        return self.validation_end - self.h_max + 1

    @property
    def test_evaluation_end(self) -> int:
        return self.trace_end - self.h_max + 1

    def target_is_common(self, target: int) -> bool:
        return (
            3 <= target < self.training_target_end
            or self.validation_start <= target < self.validation_evaluation_end
            or self.test_start <= target < self.test_evaluation_end
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "h_max": self.h_max,
            "train_end": self.train_end,
            "training_target_windows": [3, self.training_target_end],
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "validation_evaluation_windows": [
                self.validation_start,
                self.validation_evaluation_end,
            ],
            "validation_carry_only_windows": [
                self.validation_evaluation_end,
                self.validation_end,
            ],
            "test_start": self.test_start,
            "trace_end": self.trace_end,
            "test_evaluation_windows": [self.test_start, self.test_evaluation_end],
            "final_excluded_tail_windows": [self.test_evaluation_end, self.trace_end],
        }


@dataclass(frozen=True)
class HorizonPredictorRun:
    demand_matrix: DemandMatrix
    training_structure: LearnedStructure
    dataset: PredictorDataset
    targets: PredictorTargets
    predictor: FastPredictor
    outputs: PredictorOutputs
    evaluation: PredictorEvaluation
    config_artifact: dict[str, Any]


def subset_common_dataset(
    dataset: PredictorDataset, protocol: CommonWindowProtocol
) -> PredictorDataset:
    """Return identical horizon-safe source/target rows for every horizon."""

    mask = np.asarray(
        [protocol.target_is_common(int(target)) for target in dataset.target_window_ids],
        dtype=np.bool_,
    )
    if not np.any(mask):
        raise ValueError("common-window protocol selected no predictor examples")
    split_codes = np.asarray(
        [
            0 if target < protocol.train_end
            else 1 if target < protocol.validation_end
            else 2
            for target in dataset.target_window_ids[mask]
        ],
        dtype=np.int8,
    )
    return PredictorDataset(
        projected_factor_demand=dataset.projected_factor_demand,
        recent_features=dataset.recent_features[mask],
        context_features=dataset.context_features[mask],
        recent_feature_names=dataset.recent_feature_names,
        context_feature_names=dataset.context_feature_names,
        recent_continuous_indices=dataset.recent_continuous_indices,
        context_continuous_indices=dataset.context_continuous_indices,
        feature_window_ids=dataset.feature_window_ids[mask],
        target_window_ids=dataset.target_window_ids[mask],
        factor_ids=dataset.factor_ids[mask],
        split_codes=split_codes,
        user_ids=dataset.user_ids,
        request_types=dataset.request_types,
        window_contexts=dataset.window_contexts,
        warnings=dataset.warnings,
    )


def build_horizon_targets(
    run_dir: str | Path,
    learned_membership: np.ndarray,
    record_ids: np.ndarray,
    dataset: PredictorDataset,
    horizon: int,
) -> PredictorTargets:
    """Build binary any-start and summed-intensity labels over one horizon."""

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("forecast horizon must be a positive integer")
    one_window = build_predictor_targets(
        run_dir, learned_membership, record_ids, dataset
    )
    hidden_path = Path(run_dir) / "hidden_ground_truth.json"
    hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
    raw_bursts = hidden.get("bursts")
    raw_counts = hidden.get("access_source_counts_by_window")
    if not isinstance(raw_bursts, list) or not isinstance(raw_counts, list):
        raise ValueError("hidden horizon targets are incomplete")
    bursts_by_factor: dict[int, list[dict[str, Any]]] = {}
    for burst in raw_bursts:
        if not isinstance(burst, dict):
            raise ValueError("hidden bursts must be objects")
        bursts_by_factor.setdefault(int(burst["working_set_id"]), []).append(burst)
    factor_count = learned_membership.shape[0]
    source = np.zeros((len(raw_counts), factor_count), dtype=np.float64)
    for window_id, row in enumerate(raw_counts):
        counts = row["working_set_access_counts"]
        for factor_id, item in enumerate(counts):
            source[window_id, factor_id] = int(item["access_count"])

    activation = np.zeros(dataset.example_count, dtype=np.int8)
    intensity = np.full(dataset.example_count, np.nan, dtype=np.float64)
    eligible = np.ones(dataset.example_count, dtype=np.bool_)
    realized = np.zeros(dataset.example_count, dtype=np.float64)
    for index, (target, learned_factor) in enumerate(
        zip(dataset.target_window_ids, dataset.factor_ids, strict=True)
    ):
        start = int(target)
        end = start + horizon
        if end > source.shape[0]:
            raise ValueError("horizon target crosses the trace boundary")
        planted = int(one_window.learned_to_planted[int(learned_factor)])
        starts = [
            burst for burst in bursts_by_factor.get(planted, [])
            if start <= int(burst["start_window"]) < end
        ]
        if starts:
            activation[index] = 1
            intensity[index] = sum(float(burst["intensity"]) for burst in starts)
        realized[index] = float(np.sum(source[start:end, planted]))
    return PredictorTargets(
        activation=activation,
        intensity=intensity,
        eligible=eligible,
        realized_next_window_accesses=realized,
        learned_to_planted=one_window.learned_to_planted,
        matching_report=one_window.matching_report,
    )


def cumulative_future(values: np.ndarray, horizon: int) -> np.ndarray:
    """Return full-shape cumulative future values with an unavailable NaN tail."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("cumulative values must be a matrix")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    result = np.full(array.shape, np.nan, dtype=np.float64)
    for start in range(0, len(array) - horizon + 1):
        result[start] = np.sum(array[start : start + horizon], axis=0)
    return result


def run_horizon_predictor(
    run_dir: str | Path,
    structure_config_path: str | Path,
    predictor_config_path: str | Path,
    output_dir: str | Path,
    horizon: int,
    protocol: CommonWindowProtocol,
) -> HorizonPredictorRun:
    """Fit the accepted model family on one common matched-horizon population."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("horizon predictor output directory must be empty")
    source = Path(run_dir)
    predictor_config = PredictorConfig.from_json(predictor_config_path)
    structure_config = StructureLearnerConfig.from_json(structure_config_path)
    source_config = WorkloadConfig.from_json(source / "config.json")
    demand = build_demand_matrix(source)
    boundaries = predictor_config.split_boundaries(demand.X.shape[0])
    if (
        boundaries.train_end != protocol.train_end
        or boundaries.validation_end != protocol.validation_end
        or boundaries.num_windows != protocol.trace_end
    ):
        raise ValueError("predictor splits do not match the common-window protocol")
    training_demand = DemandMatrix(
        X=demand.X[: protocol.train_end],
        window_ids=demand.window_ids[: protocol.train_end],
        record_ids=demand.record_ids,
    )
    structure_config.validate_representative_factor_count(source_config.num_working_sets)
    training_structure = fit_structure(training_demand, structure_config)
    base_dataset = build_predictor_dataset(
        source,
        demand,
        training_structure.membership_matrix,
        boundaries,
        source_config,
    )
    dataset = subset_common_dataset(base_dataset, protocol)
    targets = build_horizon_targets(
        source,
        training_structure.membership_matrix,
        demand.record_ids,
        dataset,
        horizon,
    )
    predictor = fit_fast_predictor(dataset, targets, predictor_config)
    outputs = run_fast_prediction(dataset, predictor)
    evaluation = evaluate_fast_predictor(
        dataset,
        targets,
        predictor,
        outputs,
        predictor_config,
        source_config,
    )
    config_artifact = {
        "schema_version": 2,
        "forecast_horizon_windows": horizon,
        "common_eligible_window_protocol": protocol.to_dict(),
        "resolved_predictor_configuration": predictor_config.to_resolved_dict(),
        "resolved_structure_configuration": structure_config.to_resolved_dict(),
        "source_artifact_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "observable_events.jsonl",
                "hidden_ground_truth.json",
                "summary.json",
            )
        },
        "feature_schema": {
            "recent_feature_names": list(dataset.recent_feature_names),
            "context_feature_names": list(dataset.context_feature_names),
            "recent_continuous_column_indices": dataset.recent_continuous_indices.tolist(),
            "context_continuous_column_indices": dataset.context_continuous_indices.tolist(),
            "example_order": "common target windows ascending, then learned factor ID",
        },
        "example_counts": {
            "training": int(np.sum(dataset.split_codes == 0)),
            "validation": int(np.sum(dataset.split_codes == 1)),
            "test": int(np.sum(dataset.split_codes == 2)),
        },
    }
    report = {
        "schema_version": 2,
        "forecast_horizon_windows": horizon,
        "common_eligible_window_protocol": protocol.to_dict(),
        "training_only_structure": {
            "converged": training_structure.converged,
            "iteration_count": training_structure.iteration_count,
            "matching_and_recovery": targets.matching_report,
        },
        "target_counts": {
            "positive_activation": int(np.sum(targets.activation)),
            "positive_intensity": int(np.sum(np.isfinite(targets.intensity))),
        },
        **evaluation.report,
    }
    evaluation = PredictorEvaluation(report)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "predictor_config.json", config_artifact)
    _write_bundle(destination / "predictor_bundle.npz", training_structure, dataset, predictor)
    _write_predictions(destination / "predictions.npz", dataset, outputs)
    _write_json(destination / "evaluation_report.json", report)
    return HorizonPredictorRun(
        demand,
        training_structure,
        dataset,
        targets,
        predictor,
        outputs,
        evaluation,
        config_artifact,
    )


def predictor_window_matrices(
    run: HorizonPredictorRun,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align primary predictions to full window-by-factor matrices."""

    window_count, factor_count = run.dataset.projected_factor_demand.shape
    probability = np.full((window_count, factor_count), np.nan, dtype=np.float64)
    intensity = np.full((window_count, factor_count), np.nan, dtype=np.float64)
    available = np.zeros(window_count, dtype=np.bool_)
    for index, (target, factor) in enumerate(
        zip(run.dataset.target_window_ids, run.dataset.factor_ids, strict=True)
    ):
        probability[int(target), int(factor)] = (
            run.outputs.context_activation_probability[index]
        )
        intensity[int(target), int(factor)] = (
            run.outputs.context_intensity_prediction[index]
        )
        available[int(target)] = True
    if np.any(~np.isfinite(probability[available])) or np.any(~np.isfinite(intensity[available])):
        raise ValueError("horizon predictor matrices are incomplete")
    return probability, intensity, available


def regime_sparsity_diagnostics(
    hidden: Mapping[str, Any],
    num_windows: int,
    num_working_sets: int,
    workload_validation_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize realized burst sparsity without affecting generation."""

    bursts = [dict(row) for row in hidden.get("bursts", [])]
    by_factor = {factor: [] for factor in range(num_working_sets)}
    active_counts = np.zeros(num_windows, dtype=np.int64)
    for burst in bursts:
        factor = int(burst["working_set_id"])
        by_factor[factor].append(burst)
        start = int(burst["start_window"])
        end = min(num_windows, int(burst["end_window_exclusive"]))
        active_counts[start:end] += 1
    dormant: list[int] = []
    per_factor = []
    fewer_than_two = []
    for factor in range(num_working_sets):
        ordered = sorted(by_factor[factor], key=lambda row: int(row["start_window"]))
        intervals = [
            int(current["start_window"]) - int(previous["end_window_exclusive"])
            for previous, current in zip(ordered, ordered[1:])
        ]
        dormant.extend(intervals)
        if len(ordered) < 2:
            fewer_than_two.append(factor)
        per_factor.append({
            "working_set_id": factor,
            "burst_start_count": len(ordered),
            "dormant_interval_count": len(intervals),
            "dormant_interval_mean": float(np.mean(intervals)) if intervals else None,
            "dormant_interval_median": float(np.median(intervals)) if intervals else None,
        })
    trials = [dict(row) for row in hidden.get("activation_trials", [])]
    precursor_trials = [row for row in trials if float(row["previous_window_precursor_score"]) > 0]
    precursor_successes = [row for row in precursor_trials if row["activated"]]
    spontaneous = [
        row for row in trials
        if row["activated"] and float(row["contextual_probability"]) == 0.0
    ]
    demonstration = workload_validation_report["demonstration_checks"]
    return {
        "total_burst_starts": len(bursts),
        "burst_starts_per_100_windows": 100.0 * len(bursts) / num_windows,
        "fraction_windows_with_active_burst": float(np.mean(active_counts >= 1)),
        "mean_active_working_set_count": float(np.mean(active_counts)),
        "maximum_active_working_set_count": int(np.max(active_counts)),
        "fraction_windows_with_at_least_two_active": float(np.mean(active_counts >= 2)),
        "fraction_windows_with_at_least_three_active": float(np.mean(active_counts >= 3)),
        "pooled_dormant_intervals": {
            "count": len(dormant),
            "mean": float(np.mean(dormant)) if dormant else None,
            "median": float(np.median(dormant)) if dormant else None,
        },
        "working_sets_with_fewer_than_two_bursts": fewer_than_two,
        "per_working_set": per_factor,
        "precursor_demonstration_counts": {
            "clear_precursor_then_burst": demonstration["clear_precursor_followed_by_burst"]["count"],
            "clear_precursor_no_burst": demonstration["clear_precursor_followed_by_no_burst"]["count"],
            "no_clear_precursor_then_burst": demonstration["no_clear_precursor_followed_by_burst"]["count"],
        },
        "precursor_to_burst_rate": len(precursor_successes) / len(precursor_trials) if precursor_trials else None,
        "spontaneous_start_count": len(spontaneous),
        "spontaneous_start_definition": "successful start with zero contextual activation probability",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
