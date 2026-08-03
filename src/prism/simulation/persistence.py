"""Strict frozen-input orchestration and deterministic four-artifact output."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
from typing import Any, Mapping

import numpy as np
import scipy
import sklearn

from prism.predictor.persistence import PREDICTOR_ARTIFACT_FILENAMES
from prism.simulation.config import SimulationConfig
from prism.simulation.diagnostics import evaluate_causal_diagnostics
from prism.simulation.evaluate import (
    evaluate_policy_results,
    evaluate_projection_diagnostics,
)
from prism.simulation.projection import (
    ProjectionResult,
    build_record_demand_variants,
    fit_record_demand_projection,
)
from prism.simulation.replay import (
    POLICY_DISPLAY_NAMES,
    POLICY_NAMES,
    PolicyReplayResult,
    run_policy_replay,
)
from prism.structure.demand import DemandMatrix, build_demand_matrix
from prism.structure.evaluate import SOURCE_ARTIFACT_FILENAMES
from prism.workload.config import WorkloadConfig
from prism.workload.models import OBSERVABLE_EVENT_FIELDS, ObservableEvent
from prism.workload.validate import validate_workload_run


SIMULATION_SCHEMA_VERSION = 1
SIMULATION_ARTIFACT_FILENAMES = (
    "simulation_config.json",
    "projection_model.npz",
    "policy_traces.npz",
    "evaluation_report.json",
)
PREDICTOR_BUNDLE_ARRAYS = frozenset(
    {
        "membership_matrix",
        "record_ids",
        "factor_ids",
        "recent_feature_names",
        "recent_scaler_means",
        "recent_scaler_scales",
        "recent_continuous_column_indices",
        "recent_logistic_coefficients",
        "recent_logistic_intercept",
        "context_feature_names",
        "context_scaler_means",
        "context_scaler_scales",
        "context_continuous_column_indices",
        "context_logistic_coefficients",
        "context_logistic_intercept",
        "intensity_scaler_means",
        "intensity_scaler_scales",
        "intensity_continuous_column_indices",
        "intensity_ridge_coefficients",
        "intensity_ridge_intercept",
        "activation_base_rates",
        "intensity_factor_means",
        "global_intensity_mean",
        "user_ids",
        "request_types",
    }
)
PREDICTION_ARRAYS = frozenset(
    {
        "feature_window_id",
        "target_window_id",
        "learned_factor_id",
        "split_code",
        "base_rate_activation_probability",
        "recent_demand_activation_probability",
        "context_plus_state_activation_probability",
        "per_factor_mean_intensity_prediction",
        "context_plus_state_intensity_prediction",
    }
)
PROJECTION_ARRAYS = frozenset(
    {
        "membership_matrix",
        "record_ids",
        "factor_ids",
        "factor_calibration_coefficients",
        "factor_calibration_residual_norms",
        "residual_record_baseline",
        "training_target_window_ids",
    }
)
POLICY_TRACE_ARRAYS = frozenset(
    {
        "policy_names",
        "test_window_ids",
        "observable_event_indices",
        "per_event_tier_cost",
        "per_event_hit_indicator",
        "per_window_access_count",
        "per_window_hit_count",
        "per_window_miss_count",
        "per_window_access_cost",
        "per_window_promotion_cost",
        "per_window_combined_cost",
        "per_window_promotion_count",
        "per_window_eviction_count",
        "per_window_bytes_promoted",
        "per_window_bytes_evicted",
        "per_window_end_occupancy_bytes",
        "per_window_end_resident_record_count",
        "final_resident_record_indicator",
        "previous_target_record_indicator",
        "pre_window_resident_record_indicator",
        "per_window_promotion_record_indicator",
    }
)


class SimulationInputError(ValueError):
    """Raised when frozen workload or predictor artifacts do not correspond."""


class SimulationOutputDirectoryError(ValueError):
    """Raised when Milestone 4 output cannot be written safely."""


@dataclass(frozen=True)
class SimulationRun:
    config: SimulationConfig
    projection: ProjectionResult
    replay: PolicyReplayResult
    evaluation_report: dict[str, Any]
    simulation_config_artifact: dict[str, Any]


@dataclass(frozen=True)
class _FrozenInputs:
    source_config: WorkloadConfig
    demand: DemandMatrix
    events: tuple[ObservableEvent, ...]
    record_sizes: np.ndarray
    membership_matrix: np.ndarray
    factor_ids: np.ndarray
    activation_probability: np.ndarray
    conditional_intensity: np.ndarray
    prediction_available: np.ndarray
    train_end: int
    validation_end: int
    bursts: tuple[dict[str, Any], ...]
    hidden_ground_truth: dict[str, Any]
    source_hashes: dict[str, str]
    predictor_hashes: dict[str, str]


def run_simulated_evaluation(
    run_dir: str | Path,
    predictor_run_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    require_scientific_gates: bool = True,
) -> SimulationRun:
    """Validate frozen inputs, fit projection, replay, evaluate, and persist."""

    destination = Path(output_dir)
    _validate_output_directory(destination)
    config = SimulationConfig.from_json(config_path)
    frozen = _load_frozen_inputs(
        Path(run_dir),
        Path(predictor_run_dir),
        require_scientific_gates=require_scientific_gates,
    )
    config.validate_record_sizes(frozen.record_sizes.tolist())
    projection = fit_record_demand_projection(
        frozen.demand.X,
        frozen.membership_matrix,
        frozen.activation_probability,
        frozen.conditional_intensity,
        frozen.prediction_available,
        frozen.train_end,
    )
    projection_diagnostics = evaluate_projection_diagnostics(
        projection,
        frozen.demand.X,
        frozen.prediction_available,
        frozen.train_end,
        frozen.validation_end,
    )
    forecast_variants = build_record_demand_variants(
        projection,
        frozen.activation_probability,
        frozen.conditional_intensity,
        frozen.prediction_available,
    )
    replay = run_policy_replay(
        frozen.events,
        frozen.demand.X,
        frozen.demand.record_ids,
        frozen.record_sizes,
        projection.predicted_record_demand,
        frozen.prediction_available,
        config,
        frozen.train_end,
        frozen.validation_end,
        forecast_variants=forecast_variants,
    )
    policy_evaluation = evaluate_policy_results(replay, frozen.bursts)
    causal_diagnostics = evaluate_causal_diagnostics(
        replay,
        frozen.demand.X,
        frozen.demand.record_ids,
        frozen.record_sizes,
        frozen.hidden_ground_truth,
        config,
    )
    warnings = list(projection_diagnostics["warnings"])
    for warning in policy_evaluation["warnings"]:
        if warning not in warnings:
            warnings.append(warning)
    report = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "source_artifact_sha256": frozen.source_hashes,
        "predictor_artifact_sha256": frozen.predictor_hashes,
        "projection": {
            "coefficients": [
                {
                    "learned_factor_id": int(factor_id),
                    "a_recent_factor_demand": float(
                        projection.model.factor_coefficients[index, 0]
                    ),
                    "b_activation_intensity": float(
                        projection.model.factor_coefficients[index, 1]
                    ),
                    "c_constant": float(
                        projection.model.factor_coefficients[index, 2]
                    ),
                    "residual_norm": float(
                        projection.model.factor_residual_norms[index]
                    ),
                }
                for index, factor_id in enumerate(frozen.factor_ids)
            ],
            "residual_record_baseline_summary": _numeric_summary(
                projection.model.residual_record_baseline
            ),
            "training_target_window_count": len(
                projection.model.training_target_window_ids
            ),
            "diagnostics": projection_diagnostics,
        },
        "simulator_invariants": {
            "policy_count": len(POLICY_NAMES),
            "identical_test_event_count_per_policy": True,
            "independent_policy_state": True,
            "capacity_violations": replay.capacity_violations,
            "validation_costs_excluded_from_primary_metrics": True,
            "validation_state_carried_into_test": True,
            "scientific_gates_required_for_input": require_scientific_gates,
        },
        "policy_metrics": replay.policy_metrics,
        "causal_diagnostics": causal_diagnostics,
        **policy_evaluation,
        "warnings": warnings,
    }
    static = _static_metadata(frozen.record_sizes, config)
    config_artifact = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "resolved_simulation_configuration": config.to_dict(),
        "derived_static_metadata": static,
        "source_artifact_sha256": frozen.source_hashes,
        "predictor_artifact_sha256": frozen.predictor_hashes,
        "simulation_configuration_sha256": _sha256(Path(config_path)),
        "split_boundaries": {
            "training_windows": [0, frozen.train_end],
            "validation_windows": [frozen.train_end, frozen.validation_end],
            "test_windows": [frozen.validation_end, frozen.demand.X.shape[0]],
        },
        "policy_names": list(POLICY_NAMES),
        "policy_display_names": {
            name: POLICY_DISPLAY_NAMES[name] for name in POLICY_NAMES
        },
        "fixed_semantics": {
            "window_boundary_policies": [
                "recent_demand_greedy",
                "predictive_greedy",
                "training_popularity_static",
                "validation_final_frozen",
                "recent_state_only",
                "activation_intensity_only",
                "residual_baseline_only",
                "oracle_greedy",
                "oracle_exact",
            ],
            "event_level_policies": ["lru", "lfu"],
            "greedy_tie_breaking": [
                "descending benefit density",
                "descending total benefit",
                "ascending record size",
                "ascending record ID",
            ],
            "migration_order": "ascending record ID",
            "exact_solver": "scipy.optimize.milp binary variables",
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
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "simulation_config.json", config_artifact)
    _write_projection(destination / "projection_model.npz", frozen, projection)
    _write_policy_traces(destination / "policy_traces.npz", replay)
    _write_json(destination / "evaluation_report.json", report)
    return SimulationRun(config, projection, replay, report, config_artifact)


def _load_frozen_inputs(
    source: Path,
    predictor: Path,
    *,
    require_scientific_gates: bool = True,
) -> _FrozenInputs:
    workload_validation = validate_workload_run(source)
    if require_scientific_gates and not workload_validation.demonstrations_passed:
        raise SimulationInputError("source workload demonstration gate did not pass")
    if require_scientific_gates and not workload_validation.intensity_signal_passed:
        raise SimulationInputError("source workload intensity gate did not pass")
    source_config = WorkloadConfig.from_json(source / "config.json")
    demand = build_demand_matrix(source)
    source_hashes = {
        filename: _sha256(source / filename)
        for filename in SOURCE_ARTIFACT_FILENAMES
    }
    predictor_files = {path.name for path in predictor.iterdir() if path.is_file()} if predictor.is_dir() else set()
    if predictor_files != set(PREDICTOR_ARTIFACT_FILENAMES):
        raise SimulationInputError(
            "predictor run must contain exactly the four Milestone 3 artifacts"
        )
    predictor_hashes = {
        filename: _sha256(predictor / filename)
        for filename in PREDICTOR_ARTIFACT_FILENAMES
    }
    predictor_config = _load_json(predictor / "predictor_config.json")
    predictor_report = _load_json(predictor / "evaluation_report.json")
    for artifact_name, artifact in (
        ("predictor_config.json", predictor_config),
        ("evaluation_report.json", predictor_report),
    ):
        if artifact.get("source_artifact_sha256") != source_hashes:
            raise SimulationInputError(
                f"{artifact_name} source hashes do not match the supplied workload"
            )
    gates = predictor_report.get("scientific_gates")
    if not isinstance(gates, dict):
        raise SimulationInputError("predictor scientific gate report is missing")
    if require_scientific_gates and not gates.get("all_passed"):
        raise SimulationInputError("predictor scientific gates did not all pass")
    boundaries = predictor_config.get("split_boundaries")
    if not isinstance(boundaries, dict):
        raise SimulationInputError("predictor split boundaries are missing")
    if boundaries != predictor_report.get("split_boundaries"):
        raise SimulationInputError("predictor artifacts disagree on split boundaries")
    train_end = _integer("train_end", boundaries.get("train_end"), minimum=1)
    validation_end = _integer(
        "validation_end", boundaries.get("validation_end"), minimum=train_end + 1
    )
    if validation_end >= demand.X.shape[0]:
        raise SimulationInputError("predictor split boundaries exceed source windows")

    with np.load(predictor / "predictor_bundle.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTOR_BUNDLE_ARRAYS):
            raise SimulationInputError("predictor bundle has an invalid array allowlist")
        membership = np.array(archive["membership_matrix"], dtype=np.float64)
        record_ids = np.array(archive["record_ids"], dtype=np.int64)
        factor_ids = np.array(archive["factor_ids"], dtype=np.int64)
    if not np.array_equal(record_ids, demand.record_ids):
        raise SimulationInputError("predictor record IDs do not match the workload")
    expected_factor_ids = np.arange(source_config.num_working_sets, dtype=np.int64)
    if not np.array_equal(factor_ids, expected_factor_ids):
        raise SimulationInputError("predictor factor IDs do not match the workload")
    if membership.shape != (len(factor_ids), len(record_ids)):
        raise SimulationInputError("predictor membership dimensions are invalid")
    if np.any(~np.isfinite(membership)) or np.any(membership < 0.0):
        raise SimulationInputError("predictor membership values are invalid")

    probability, intensity, available = _load_prediction_matrices(
        predictor / "predictions.npz",
        demand.X.shape[0],
        factor_ids,
        train_end,
        validation_end,
    )
    events, record_sizes = _load_observable_events(
        source / "observable_events.jsonl", demand.record_ids
    )
    hidden = _load_json(source / "hidden_ground_truth.json")
    raw_bursts = hidden.get("bursts")
    if not isinstance(raw_bursts, list):
        raise SimulationInputError("hidden bursts must be a list")
    return _FrozenInputs(
        source_config,
        demand,
        events,
        record_sizes,
        membership,
        factor_ids,
        probability,
        intensity,
        available,
        train_end,
        validation_end,
        tuple(dict(burst) for burst in raw_bursts),
        hidden,
        source_hashes,
        predictor_hashes,
    )


def _load_prediction_matrices(
    path: Path,
    num_windows: int,
    factor_ids: np.ndarray,
    train_end: int,
    validation_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(PREDICTION_ARRAYS):
            raise SimulationInputError("predictor predictions have an invalid array allowlist")
        feature_windows = np.array(archive["feature_window_id"], dtype=np.int64)
        target_windows = np.array(archive["target_window_id"], dtype=np.int64)
        row_factors = np.array(archive["learned_factor_id"], dtype=np.int64)
        split_codes = np.array(archive["split_code"], dtype=np.int8)
        probabilities = np.array(
            archive["context_plus_state_activation_probability"], dtype=np.float64
        )
        intensities = np.array(
            archive["context_plus_state_intensity_prediction"], dtype=np.float64
        )
    lengths = {
        len(feature_windows),
        len(target_windows),
        len(row_factors),
        len(split_codes),
        len(probabilities),
        len(intensities),
    }
    if len(lengths) != 1:
        raise SimulationInputError("predictor prediction arrays have unequal lengths")
    factor_index = {int(factor_id): index for index, factor_id in enumerate(factor_ids)}
    probability = np.full((num_windows, len(factor_ids)), np.nan, dtype=np.float64)
    intensity = np.full_like(probability, np.nan)
    seen = np.zeros_like(probability, dtype=np.bool_)
    for feature, target, factor, split, activation, conditional in zip(
        feature_windows,
        target_windows,
        row_factors,
        split_codes,
        probabilities,
        intensities,
        strict=True,
    ):
        if target != feature + 1 or not 0 <= target < num_windows:
            raise SimulationInputError("predictor feature/target windows are invalid")
        if int(factor) not in factor_index:
            raise SimulationInputError("predictor row references an unknown factor")
        expected_split = 0 if target < train_end else 1 if target < validation_end else 2
        if int(split) != expected_split:
            raise SimulationInputError("predictor split codes are inconsistent")
        column = factor_index[int(factor)]
        if seen[target, column]:
            raise SimulationInputError("predictor contains a duplicate factor-window row")
        if not np.isfinite(activation) or not 0.0 <= activation <= 1.0:
            raise SimulationInputError("predictor activation probabilities are invalid")
        if not np.isfinite(conditional):
            raise SimulationInputError("predictor intensity predictions are invalid")
        seen[target, column] = True
        probability[target, column] = activation
        intensity[target, column] = conditional
    available = np.all(seen, axis=1)
    expected_available = np.arange(num_windows) >= 3
    if not np.array_equal(available, expected_available):
        raise SimulationInputError("predictor factor/window coverage is incomplete")
    return probability, intensity, available


def _load_observable_events(
    path: Path, record_ids: np.ndarray
) -> tuple[tuple[ObservableEvent, ...], np.ndarray]:
    record_index = {int(record_id): index for index, record_id in enumerate(record_ids)}
    sizes = np.full(len(record_ids), -1, dtype=np.int64)
    events: list[ObservableEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line, object_pairs_hook=_unique_object)
            except json.JSONDecodeError as error:
                raise SimulationInputError(
                    f"malformed observable event line {line_number}: {error.msg}"
                ) from error
            if not isinstance(raw, dict) or set(raw) != set(OBSERVABLE_EVENT_FIELDS):
                raise SimulationInputError("observable event fields are invalid")
            event = ObservableEvent(**raw)
            if event.record_id not in record_index:
                raise SimulationInputError("observable event record ID is invalid")
            index = record_index[event.record_id]
            if sizes[index] not in (-1, event.record_size_bytes):
                raise SimulationInputError("observable record size changes across events")
            sizes[index] = event.record_size_bytes
            events.append(event)
    if np.any(sizes <= 0):
        raise SimulationInputError(
            "every source record must expose its static size in observable events"
        )
    return tuple(events), sizes


def _write_projection(
    path: Path, frozen: _FrozenInputs, projection: ProjectionResult
) -> None:
    np.savez(
        path,
        membership_matrix=projection.model.membership_matrix,
        record_ids=frozen.demand.record_ids,
        factor_ids=frozen.factor_ids,
        factor_calibration_coefficients=projection.model.factor_coefficients,
        factor_calibration_residual_norms=projection.model.factor_residual_norms,
        residual_record_baseline=projection.model.residual_record_baseline,
        training_target_window_ids=projection.model.training_target_window_ids,
    )


def _write_policy_traces(path: Path, replay: PolicyReplayResult) -> None:
    arrays: dict[str, np.ndarray] = {
        "policy_names": np.asarray(replay.policy_names),
        "test_window_ids": replay.test_window_ids,
        "observable_event_indices": replay.test_event_indices,
        "per_event_tier_cost": replay.per_event_tier_cost,
        "per_event_hit_indicator": replay.per_event_hit,
        "final_resident_record_indicator": replay.final_resident_indicator,
        "previous_target_record_indicator": replay.previous_target_indicator,
        "pre_window_resident_record_indicator": replay.pre_window_resident_indicator,
        "per_window_promotion_record_indicator": replay.per_window_promotion_indicator,
    }
    for name, values in replay.per_window.items():
        arrays[f"per_window_{name}"] = values
    if set(arrays) != set(POLICY_TRACE_ARRAYS):
        raise SimulationInputError("internal policy trace array allowlist is invalid")
    np.savez(path, **arrays)


def _static_metadata(record_sizes: np.ndarray, config: SimulationConfig) -> dict[str, Any]:
    total = int(np.sum(record_sizes, dtype=np.int64))
    median = float(statistics.median(record_sizes.tolist()))
    return {
        "record_count": len(record_sizes),
        "total_record_bytes": total,
        "median_record_size_bytes": median,
        "resolved_fast_capacity_bytes": config.fast_capacity_bytes,
        "capacity_fraction_of_total_bytes": config.fast_capacity_bytes / total,
        "median_record_promotion_cost": median * config.promotion_cost_per_byte,
        "representative_formula_values": {
            "floor_quarter_total_bytes": int(0.25 * total),
            "two_saved_reads_promotion_cost_per_byte": (
                2.0 * (config.slow_read_cost - config.fast_read_cost) / median
            ),
        },
    }


def _numeric_summary(values: np.ndarray) -> dict[str, int | float]:
    return {
        "count": len(values),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _validate_output_directory(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise SimulationOutputDirectoryError(
                f"output path exists and is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise SimulationOutputDirectoryError(
                f"output directory must be empty: {destination}"
            )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SimulationInputError(f"missing required artifact: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise SimulationInputError(f"malformed JSON in {path.name}: {error.msg}") from error
    if not isinstance(value, dict):
        raise SimulationInputError(f"{path.name} root must be an object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SimulationInputError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SimulationInputError(f"{name} must be an integer at least {minimum}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise SimulationInputError(f"missing required artifact: {path.name}")
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
