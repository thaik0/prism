"""Deterministic orchestration and persistence for Milestone 2."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy
import sklearn

from prism.structure.config import StructureLearnerConfig
from prism.structure.demand import DemandMatrix, DemandMatrixError, build_demand_matrix
from prism.structure.evaluate import RecoveryEvaluation, evaluate_recovery
from prism.structure.learner import LearnedStructure, fit_structure
from prism.workload.config import WorkloadConfig, WorkloadConfigError


LEARNER_SCHEMA_VERSION = 1
STRUCTURE_ARTIFACT_FILENAMES = (
    "learner_config.json",
    "demand_matrix.npz",
    "learned_structure.npz",
    "recovery_report.json",
)


class StructureOutputDirectoryError(ValueError):
    """Raised when Milestone 2 output cannot be written safely."""


@dataclass(frozen=True)
class StructureRunResult:
    """In-memory result corresponding to the four persisted artifacts."""

    demand_matrix: DemandMatrix
    learned_structure: LearnedStructure
    recovery_evaluation: RecoveryEvaluation
    learner_config_artifact: dict[str, Any]


def run_structure_recovery(
    run_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> StructureRunResult:
    """Build, fit, evaluate, and persist one complete Milestone 2 run."""

    destination = Path(output_dir)
    _validate_output_directory(destination)

    source = Path(run_dir)
    demand = build_demand_matrix(source)
    learner_config = StructureLearnerConfig.from_json(config_path)
    try:
        source_config = WorkloadConfig.from_json(source / "config.json")
    except WorkloadConfigError as error:
        raise DemandMatrixError(f"source config is invalid: {error}") from error
    learner_config.validate_dimensions(*demand.X.shape)
    learner_config.validate_representative_factor_count(
        source_config.num_working_sets
    )

    learned = fit_structure(demand, learner_config)
    evaluation = evaluate_recovery(source, demand, learned, learner_config)
    learner_artifact = {
        "schema_version": LEARNER_SCHEMA_VERSION,
        "resolved_learner_configuration": learner_config.to_resolved_dict(),
        "source_artifact_sha256": evaluation.report[
            "source_artifact_sha256"
        ],
        "source_dimensions": {
            "num_windows": demand.X.shape[0],
            "num_records": demand.X.shape[1],
            "num_working_sets": source_config.num_working_sets,
            "observable_event_count": demand.event_count,
        },
        "resolved_factor_count": learner_config.n_components,
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": _distribution_version("scikit-learn", sklearn.__version__),
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "learner_config.json", learner_artifact)
    np.savez(
        destination / "demand_matrix.npz",
        X=demand.X,
        window_ids=demand.window_ids,
        record_ids=demand.record_ids,
    )
    np.savez(
        destination / "learned_structure.npz",
        activation_matrix=learned.activation_matrix,
        membership_matrix=learned.membership_matrix,
        factor_ids=learned.factor_ids,
        window_ids=learned.window_ids,
        record_ids=learned.record_ids,
    )
    _write_json(destination / "recovery_report.json", evaluation.report)
    return StructureRunResult(
        demand_matrix=demand,
        learned_structure=learned,
        recovery_evaluation=evaluation,
        learner_config_artifact=learner_artifact,
    )


def _validate_output_directory(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise StructureOutputDirectoryError(
                f"output path exists and is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise StructureOutputDirectoryError(
                f"output directory must be empty: {destination}"
            )


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
