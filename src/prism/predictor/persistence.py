"""Deterministic end-to-end orchestration and four-artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy
import sklearn

from prism.predictor.config import PredictorConfig
from prism.predictor.evaluate import PredictorEvaluation, evaluate_fast_predictor
from prism.predictor.features import PredictorDataset, build_predictor_dataset
from prism.predictor.models import (
    FastPredictor,
    PredictorOutputs,
    fit_fast_predictor,
    run_fast_prediction,
)
from prism.predictor.targets import PredictorTargets, build_predictor_targets
from prism.structure.config import StructureLearnerConfig
from prism.structure.demand import DemandMatrix, build_demand_matrix
from prism.structure.evaluate import SOURCE_ARTIFACT_FILENAMES
from prism.structure.learner import LearnedStructure, fit_structure
from prism.workload.config import WorkloadConfig


PREDICTOR_SCHEMA_VERSION = 1
PREDICTOR_ARTIFACT_FILENAMES = (
    "predictor_config.json",
    "predictor_bundle.npz",
    "predictions.npz",
    "evaluation_report.json",
)


class PredictorOutputDirectoryError(ValueError):
    """Raised when predictor artifacts cannot be written safely."""


@dataclass(frozen=True)
class FastPredictionRun:
    demand_matrix: DemandMatrix
    training_structure: LearnedStructure
    dataset: PredictorDataset
    targets: PredictorTargets
    predictor: FastPredictor
    outputs: PredictorOutputs
    evaluation: PredictorEvaluation
    predictor_config_artifact: dict[str, Any]


def run_predictor_experiment(
    run_dir: str | Path,
    structure_config_path: str | Path,
    predictor_config_path: str | Path,
    output_dir: str | Path,
) -> FastPredictionRun:
    """Build, fit, evaluate, and persist one complete Milestone 3 experiment."""

    destination = Path(output_dir)
    _validate_output_directory(destination)
    source = Path(run_dir)
    predictor_config = PredictorConfig.from_json(predictor_config_path)
    structure_config = StructureLearnerConfig.from_json(structure_config_path)
    source_config = WorkloadConfig.from_json(source / "config.json")
    demand = build_demand_matrix(source)
    boundaries = predictor_config.split_boundaries(demand.X.shape[0])
    structure_config.validate_dimensions(boundaries.train_end, demand.X.shape[1])
    structure_config.validate_representative_factor_count(
        source_config.num_working_sets
    )

    training_demand = DemandMatrix(
        X=demand.X[: boundaries.train_end],
        window_ids=demand.window_ids[: boundaries.train_end],
        record_ids=demand.record_ids,
    )
    training_structure = fit_structure(training_demand, structure_config)
    dataset = build_predictor_dataset(
        source,
        demand,
        training_structure.membership_matrix,
        boundaries,
        source_config,
    )
    targets = build_predictor_targets(
        source,
        training_structure.membership_matrix,
        demand.record_ids,
        dataset,
    )
    predictor = fit_fast_predictor(dataset, targets, predictor_config)
    outputs = run_fast_prediction(dataset, predictor)
    evaluation = evaluate_fast_predictor(
        dataset, targets, predictor, outputs, predictor_config, source_config
    )

    source_hashes = _hash_sources(source)
    split_counts = _split_counts(dataset, targets)
    training_reconstruction = (
        training_structure.activation_matrix @ training_structure.membership_matrix
    )
    training_X = np.asarray(training_demand.X, dtype=np.float64)
    reconstruction_error = float(
        np.linalg.norm(training_X - training_reconstruction, ord="fro")
    )
    training_norm = float(np.linalg.norm(training_X, ord="fro"))
    config_artifact = {
        "schema_version": PREDICTOR_SCHEMA_VERSION,
        "resolved_predictor_configuration": predictor_config.to_resolved_dict(),
        "resolved_structure_configuration": structure_config.to_resolved_dict(),
        "source_artifact_sha256": source_hashes,
        "structure_configuration_sha256": _sha256(Path(structure_config_path)),
        "predictor_configuration_sha256": _sha256(Path(predictor_config_path)),
        "split_boundaries": boundaries.to_dict(),
        "training_demand_shape": list(training_demand.X.shape),
        "feature_schema": {
            "recent_feature_names": list(dataset.recent_feature_names),
            "recent_continuous_column_indices": (
                dataset.recent_continuous_indices.tolist()
            ),
            "context_feature_names": list(dataset.context_feature_names),
            "context_continuous_column_indices": (
                dataset.context_continuous_indices.tolist()
            ),
            "example_order": "ascending feature window, then learned factor ID",
            "request_type_order": list(dataset.request_types),
            "user_id_order": dataset.user_ids.tolist(),
        },
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": _distribution_version(
                "scikit-learn", sklearn.__version__
            ),
        },
    }
    report = {
        "schema_version": PREDICTOR_SCHEMA_VERSION,
        "source_artifact_sha256": source_hashes,
        "split_boundaries": boundaries.to_dict(),
        "split_counts": split_counts,
        "training_only_structure": {
            "demand_shape": list(training_demand.X.shape),
            "converged": training_structure.converged,
            "iteration_count": training_structure.iteration_count,
            "max_iter": structure_config.max_iter,
            "convergence_warnings": list(
                training_structure.convergence_warnings
            ),
            "sklearn_reconstruction_error": (
                training_structure.sklearn_reconstruction_error
            ),
            "absolute_frobenius_error": reconstruction_error,
            "demand_matrix_frobenius_norm": training_norm,
            "normalized_frobenius_error": (
                reconstruction_error / training_norm if training_norm else None
            ),
            "matching_and_recovery": targets.matching_report,
        },
        **evaluation.report,
    }
    combined_warnings = list(report["warnings"])
    for warning in (
        *training_structure.warnings,
        *training_structure.convergence_warnings,
        *targets.matching_report["warnings"],
    ):
        if warning not in combined_warnings:
            combined_warnings.append(warning)
    report["warnings"] = combined_warnings
    evaluation = PredictorEvaluation(report=report)

    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "predictor_config.json", config_artifact)
    _write_bundle(
        destination / "predictor_bundle.npz",
        training_structure,
        dataset,
        predictor,
    )
    _write_predictions(destination / "predictions.npz", dataset, outputs)
    _write_json(destination / "evaluation_report.json", report)
    return FastPredictionRun(
        demand_matrix=demand,
        training_structure=training_structure,
        dataset=dataset,
        targets=targets,
        predictor=predictor,
        outputs=outputs,
        evaluation=evaluation,
        predictor_config_artifact=config_artifact,
    )


def _write_bundle(
    path: Path,
    structure: LearnedStructure,
    dataset: PredictorDataset,
    predictor: FastPredictor,
) -> None:
    recent = predictor.recent_activation
    context = predictor.context_activation
    intensity = predictor.context_intensity
    np.savez(
        path,
        membership_matrix=structure.membership_matrix,
        record_ids=structure.record_ids,
        factor_ids=structure.factor_ids,
        recent_feature_names=np.asarray(dataset.recent_feature_names),
        recent_scaler_means=recent.preprocessing.means,
        recent_scaler_scales=recent.preprocessing.scales,
        recent_continuous_column_indices=recent.preprocessing.continuous_indices,
        recent_logistic_coefficients=recent.coefficients,
        recent_logistic_intercept=recent.intercept,
        context_feature_names=np.asarray(dataset.context_feature_names),
        context_scaler_means=context.preprocessing.means,
        context_scaler_scales=context.preprocessing.scales,
        context_continuous_column_indices=context.preprocessing.continuous_indices,
        context_logistic_coefficients=context.coefficients,
        context_logistic_intercept=context.intercept,
        intensity_scaler_means=intensity.preprocessing.means,
        intensity_scaler_scales=intensity.preprocessing.scales,
        intensity_continuous_column_indices=(
            intensity.preprocessing.continuous_indices
        ),
        intensity_ridge_coefficients=intensity.coefficients,
        intensity_ridge_intercept=intensity.intercept,
        activation_base_rates=predictor.activation_base_rates,
        intensity_factor_means=predictor.intensity_factor_means,
        intensity_factor_available=predictor.intensity_factor_available,
        global_intensity_mean=np.asarray([predictor.global_intensity_mean]),
        user_ids=dataset.user_ids,
        request_types=np.asarray(dataset.request_types),
    )


def _write_predictions(
    path: Path, dataset: PredictorDataset, outputs: PredictorOutputs
) -> None:
    np.savez(
        path,
        feature_window_id=dataset.feature_window_ids,
        target_window_id=dataset.target_window_ids,
        learned_factor_id=dataset.factor_ids,
        split_code=dataset.split_codes,
        base_rate_activation_probability=outputs.base_activation_probability,
        recent_demand_activation_probability=(
            outputs.recent_activation_probability
        ),
        context_plus_state_activation_probability=(
            outputs.context_activation_probability
        ),
        per_factor_mean_intensity_prediction=outputs.mean_intensity_prediction,
        context_plus_state_intensity_prediction=(
            outputs.context_intensity_prediction
        ),
    )


def _split_counts(
    dataset: PredictorDataset, targets: PredictorTargets
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    factor_count = dataset.projected_factor_demand.shape[1]
    for split_name, split_code in (("training", 0), ("validation", 1), ("test", 2)):
        mask = dataset.split_codes == split_code
        result[split_name] = {
            "example_count": int(np.sum(mask)),
            "feature_window_count": int(len(np.unique(dataset.feature_window_ids[mask]))),
            "positive_activation_count": int(np.sum(targets.activation[mask])),
            "eligible_count": int(np.sum(targets.eligible[mask])),
            "positive_intensity_count": int(
                np.sum(np.isfinite(targets.intensity[mask]))
            ),
            "by_factor": [
                {
                    "learned_factor_id": factor_id,
                    "example_count": int(
                        np.sum(mask & (dataset.factor_ids == factor_id))
                    ),
                    "positive_activation_count": int(
                        np.sum(
                            targets.activation[
                                mask & (dataset.factor_ids == factor_id)
                            ]
                        )
                    ),
                    "eligible_count": int(
                        np.sum(
                            targets.eligible[
                                mask & (dataset.factor_ids == factor_id)
                            ]
                        )
                    ),
                }
                for factor_id in range(factor_count)
            ],
        }
    return result


def _validate_output_directory(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise PredictorOutputDirectoryError(
                f"output path exists and is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise PredictorOutputDirectoryError(
                f"output directory must be empty: {destination}"
            )


def _hash_sources(run_dir: Path) -> dict[str, str]:
    return {
        filename: _sha256(run_dir / filename)
        for filename in SOURCE_ARTIFACT_FILENAMES
    }


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing required file: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _distribution_version(distribution: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return fallback
