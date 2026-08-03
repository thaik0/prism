"""Sequential whole-pipeline execution, status indexing, and verified resume."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
from typing import Any, Callable

import numpy as np
import scipy
import sklearn

from prism.experiments.config import ExperimentManifest, load_manifest
from prism.experiments.materialize import (
    resolve_simulation_config,
    resolve_workload_config,
)
from prism.predictor import run_predictor_experiment
from prism.simulation import run_simulated_evaluation
from prism.structure import run_structure_recovery
from prism.workload import generate_workload, persist_workload
from prism.workload.validate import validate_workload_run, write_validation_report


class ExperimentRunError(ValueError):
    """Raised for unsafe output or invalid resume state."""


@dataclass(frozen=True)
class ExperimentExecution:
    output_dir: Path
    completed_count: int
    failed_count: int
    reused_experiment_ids: tuple[str, ...]


def run_experiments(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    experiment_id: str | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ExperimentExecution:
    """Run selected frozen experiments sequentially and never hide a failure."""

    manifest = load_manifest(manifest_path)
    destination = Path(output_dir)
    emit = progress if progress is not None else lambda _: None
    selected = _selected_ids(manifest, experiment_id)
    if resume:
        index = _load_resume_state(destination, manifest)
    else:
        _require_empty_output(destination)
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "experiment_manifest.json", _resolved_manifest(manifest))
        index = _initial_index(manifest)
        _write_json(destination / "experiment_index.json", index)

    reused: list[str] = []
    entries = {entry["experiment_id"]: entry for entry in index["runs"]}
    for current_id in selected:
        entry = entries[current_id]
        if resume and entry["status"] == "completed":
            _validate_completed_run(destination, entry, manifest.sha256)
            reused.append(current_id)
            emit(f"Reusing hash-verified completed run: {current_id}")
            continue
        run_dir = destination / "runs" / current_id
        if run_dir.exists():
            _require_run_child(destination, run_dir)
            shutil.rmtree(run_dir)
        variant_id, seed = _parse_experiment_id(current_id)
        entry.update(
            {
                "status": "running",
                "stage_reached": "initializing",
                "resolved_configuration_sha256": {},
                "source_artifact_sha256": {},
                "artifact_sha256": {},
                "scientific_gate_outcomes": {},
                "failure": None,
            }
        )
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "run_status.json", entry)
        _write_json(destination / "experiment_index.json", index)
        emit(f"Running {current_id}")
        try:
            result = _execute_run(manifest, variant_id, seed, run_dir, entry)
            entry.update(result)
            entry["status"] = "completed"
            entry["stage_reached"] = "completed"
            entry["failure"] = None
        except Exception as error:  # status contract deliberately captures exact failures
            entry["status"] = "failed"
            entry["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            entry["artifact_sha256"] = _hash_run_artifacts(run_dir)
            emit(f"Failed {current_id}: {type(error).__name__}: {error}")
        _write_json(run_dir / "run_status.json", entry)
        _write_json(destination / "experiment_index.json", index)

    _write_completion_summary(destination, manifest, index)
    completed = sum(entry["status"] == "completed" for entry in index["runs"])
    failed = sum(entry["status"] == "failed" for entry in index["runs"])
    return ExperimentExecution(destination, completed, failed, tuple(reused))


def _execute_run(
    manifest: ExperimentManifest,
    variant_id: str,
    seed: int,
    run_dir: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    variant = manifest.variant(variant_id)
    workload_config = resolve_workload_config(manifest, variant, seed)
    workload_config_path = run_dir / "resolved_workload_config.json"
    _write_json(workload_config_path, workload_config.to_dict())
    entry["stage_reached"] = "workload_generation"
    _write_json(run_dir / "run_status.json", entry)
    workload_result = generate_workload(workload_config)
    workload_dir = run_dir / "workload"
    persist_workload(workload_result, workload_dir)
    workload_validation = validate_workload_run(workload_dir)
    write_validation_report(
        workload_validation, workload_dir / "workload_validation.json"
    )

    simulation_config = resolve_simulation_config(
        variant, workload_result.hidden_ground_truth.record_sizes_bytes
    )
    simulation_config_path = run_dir / "resolved_simulation_config.json"
    _write_json(simulation_config_path, simulation_config.to_dict())
    resolved_hashes = {
        "resolved_workload_config.json": _sha256(workload_config_path),
        "resolved_simulation_config.json": _sha256(simulation_config_path),
        "structure_config": _sha256(manifest.structure_config),
        "predictor_config": _sha256(manifest.predictor_config),
    }
    entry["resolved_configuration_sha256"] = resolved_hashes
    entry["source_artifact_sha256"] = workload_validation.report[
        "source_artifact_sha256"
    ]

    entry["stage_reached"] = "structure"
    _write_json(run_dir / "run_status.json", entry)
    structure = run_structure_recovery(
        workload_dir, manifest.structure_config, run_dir / "structure"
    )
    if not structure.learned_structure.converged:
        raise ExperimentRunError("NMF did not converge")

    entry["stage_reached"] = "predictor"
    _write_json(run_dir / "run_status.json", entry)
    predictor = run_predictor_experiment(
        workload_dir,
        manifest.structure_config,
        manifest.predictor_config,
        run_dir / "predictor",
    )
    if not predictor.training_structure.converged:
        raise ExperimentRunError("predictor training NMF did not converge")
    if not predictor.predictor.converged:
        raise ExperimentRunError("predictor model did not converge")

    entry["stage_reached"] = "simulation"
    _write_json(run_dir / "run_status.json", entry)
    simulation = run_simulated_evaluation(
        workload_dir,
        run_dir / "predictor",
        simulation_config_path,
        run_dir / "simulation",
        require_scientific_gates=False,
    )
    controller = simulation.evaluation_report["controller_diagnostics"]
    if simulation.replay.capacity_violations:
        raise ExperimentRunError("simulation capacity invariant failed")
    if not controller["all_exact_windows_optimal"]:
        raise ExperimentRunError("one or more exact solves were not optimal")

    scientific = {
        "workload_demonstrations": workload_validation.demonstrations_passed,
        "workload_intensity_signal": workload_validation.intensity_signal_passed,
        "structure_recovery": structure.recovery_evaluation.representative_gate_passed,
        "predictor": predictor.evaluation.report["scientific_gates"],
        "simulation": simulation.evaluation_report["scientific_gates"],
    }
    artifacts = _hash_run_artifacts(run_dir)
    return {
        "resolved_configuration_sha256": resolved_hashes,
        "source_artifact_sha256": workload_validation.report[
            "source_artifact_sha256"
        ],
        "artifact_sha256": artifacts,
        "scientific_gate_outcomes": scientific,
    }


def _initial_index(manifest: ExperimentManifest) -> dict[str, Any]:
    runs = []
    for experiment_id in manifest.experiment_ids:
        variant_id, seed = _parse_experiment_id(experiment_id)
        runs.append(
            {
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                "seed": seed,
                "status": "pending",
                "stage_reached": "pending",
                "resolved_configuration_sha256": {},
                "source_artifact_sha256": {},
                "artifact_sha256": {},
                "scientific_gate_outcomes": {},
                "stage_directories": {
                    "workload": f"runs/{experiment_id}/workload",
                    "structure": f"runs/{experiment_id}/structure",
                    "predictor": f"runs/{experiment_id}/predictor",
                    "simulation": f"runs/{experiment_id}/simulation",
                },
                "failure": None,
            }
        )
    return {
        "schema_version": 1,
        "source_manifest_sha256": manifest.sha256,
        "expected_run_count": len(manifest.experiment_ids),
        "runs": runs,
    }


def _resolved_manifest(manifest: ExperimentManifest) -> dict[str, Any]:
    return {
        **manifest.raw,
        "source_manifest_sha256": manifest.sha256,
        "base_configuration_sha256": {
            "workload": _sha256(manifest.base_workload_config),
            "structure": _sha256(manifest.structure_config),
            "predictor": _sha256(manifest.predictor_config),
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


def _load_resume_state(
    destination: Path, manifest: ExperimentManifest
) -> dict[str, Any]:
    if not destination.is_dir():
        raise ExperimentRunError("resume output directory does not exist")
    persisted = _load_json(destination / "experiment_manifest.json")
    if persisted.get("source_manifest_sha256") != manifest.sha256:
        raise ExperimentRunError("resume manifest hash does not match")
    index = _load_json(destination / "experiment_index.json")
    expected = list(manifest.experiment_ids)
    actual = [entry.get("experiment_id") for entry in index.get("runs", [])]
    if actual != expected:
        raise ExperimentRunError("resume experiment index is incomplete or unordered")
    return index


def _validate_completed_run(
    destination: Path, entry: dict[str, Any], manifest_hash: str
) -> None:
    run_dir = destination / "runs" / entry["experiment_id"]
    status = _load_json(run_dir / "run_status.json")
    if status.get("status") != "completed" or status.get("experiment_id") != entry["experiment_id"]:
        raise ExperimentRunError(f"completed run status is invalid: {entry['experiment_id']}")
    if status != entry:
        raise ExperimentRunError(f"run status and index disagree: {entry['experiment_id']}")
    expected = entry.get("artifact_sha256")
    if not isinstance(expected, dict) or expected != _hash_run_artifacts(run_dir):
        raise ExperimentRunError(f"completed run artifact hash mismatch: {entry['experiment_id']}")
    if not manifest_hash:
        raise ExperimentRunError("manifest hash is missing")


def _write_completion_summary(
    destination: Path, manifest: ExperimentManifest, index: dict[str, Any]
) -> None:
    from prism.experiments.aggregate import write_aggregate_outputs

    write_aggregate_outputs(destination, manifest)


def _hash_run_artifacts(run_dir: Path) -> dict[str, str]:
    result = {}
    if not run_dir.is_dir():
        return result
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "run_status.json":
            result[path.relative_to(run_dir).as_posix()] = _sha256(path)
    return result


def _selected_ids(
    manifest: ExperimentManifest, experiment_id: str | None
) -> tuple[str, ...]:
    if experiment_id is None:
        return manifest.experiment_ids
    if experiment_id not in manifest.experiment_ids:
        raise ExperimentRunError(f"unknown experiment ID: {experiment_id}")
    return (experiment_id,)


def _parse_experiment_id(experiment_id: str) -> tuple[str, int]:
    marker = "__seed_"
    if marker not in experiment_id:
        raise ExperimentRunError(f"invalid experiment ID: {experiment_id}")
    variant, raw_seed = experiment_id.rsplit(marker, 1)
    try:
        seed = int(raw_seed)
    except ValueError as error:
        raise ExperimentRunError(f"invalid experiment ID: {experiment_id}") from error
    return variant, seed


def _require_empty_output(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise ExperimentRunError("output path exists and is not a directory")
        if any(destination.iterdir()):
            raise ExperimentRunError("output directory must be empty without --resume")


def _require_run_child(destination: Path, run_dir: Path) -> None:
    runs_root = (destination / "runs").resolve()
    resolved = run_dir.resolve()
    if resolved.parent != runs_root or not run_dir.name:
        raise ExperimentRunError("refusing to replace an unsafe run directory")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentRunError(f"missing required experiment artifact: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentRunError(f"experiment artifact must be an object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _distribution_version(distribution: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return fallback
