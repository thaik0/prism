"""Controlled structural-recovery evaluation using simulator-only truth."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from prism.structure.config import StructureLearnerConfig
from prism.structure.demand import DemandMatrix
from prism.structure.learner import LearnedStructure


RECOVERY_SCHEMA_VERSION = 1
SOURCE_ARTIFACT_FILENAMES = (
    "config.json",
    "observable_events.jsonl",
    "hidden_ground_truth.json",
    "summary.json",
)


class RecoveryEvaluationError(ValueError):
    """Raised when hidden evaluation truth is invalid or incompatible."""


@dataclass(frozen=True)
class RecoveryEvaluation:
    """One deterministic recovery report."""

    report: dict[str, Any]

    @property
    def representative_gate_passed(self) -> bool:
        return bool(self.report["representative_gate"]["passed"])


def evaluate_recovery(
    run_dir: str | Path,
    demand_matrix: DemandMatrix,
    learned_structure: LearnedStructure,
    config: StructureLearnerConfig,
) -> RecoveryEvaluation:
    """Evaluate learned structure against hidden truth only after fitting."""

    if not isinstance(demand_matrix, DemandMatrix):
        raise TypeError("demand_matrix must be a DemandMatrix")
    if not isinstance(learned_structure, LearnedStructure):
        raise TypeError("learned_structure must be a LearnedStructure")
    if not isinstance(config, StructureLearnerConfig):
        raise TypeError("config must be a StructureLearnerConfig")
    _validate_learned_contract(demand_matrix, learned_structure, config)

    source = Path(run_dir)
    hidden = _load_json(source / "hidden_ground_truth.json")
    planted_membership, true_activation = _load_evaluation_truth(
        hidden,
        learned_structure.record_ids,
        learned_structure.window_ids,
        config.n_components,
    )

    similarity, assignment, matching_warnings = match_factors(
        learned_structure.membership_matrix, planted_membership
    )
    fuzzy_per_factor = [
        {
            "learned_factor_id": learned_factor,
            "planted_factor_id": planted_factor,
            "cosine_similarity": float(similarity[learned_factor, planted_factor]),
        }
        for learned_factor, planted_factor in assignment
    ]
    fuzzy_values = [row["cosine_similarity"] for row in fuzzy_per_factor]

    support_rows = [
        _support_metrics(
            learned_factor,
            planted_factor,
            learned_structure.membership_matrix[learned_factor],
            planted_membership[planted_factor],
            learned_structure.record_ids,
        )
        for learned_factor, planted_factor in assignment
    ]
    support_aggregate = _aggregate_support_metrics(support_rows)

    evaluation_warnings = list(learned_structure.warnings)
    evaluation_warnings.extend(matching_warnings)
    if not learned_structure.converged:
        evaluation_warnings.append("NMF did not converge within the configured limit")
    for warning in learned_structure.convergence_warnings:
        _add_warning(evaluation_warnings, warning)

    reconstruction = (
        learned_structure.activation_matrix
        @ learned_structure.membership_matrix
    )
    X_float = np.asarray(demand_matrix.X, dtype=np.float64)
    absolute_error = float(np.linalg.norm(X_float - reconstruction, ord="fro"))
    matrix_norm = float(np.linalg.norm(X_float, ord="fro"))
    if matrix_norm == 0.0:
        normalized_error: float | None = None
        _add_warning(
            evaluation_warnings,
            "normalized reconstruction error is undefined for an all-zero demand matrix",
        )
    else:
        normalized_error = absolute_error / matrix_norm

    alignment_rows: list[dict[str, Any]] = []
    alignment_values: list[float] = []
    for learned_factor, planted_factor in assignment:
        true_signal = true_activation[:, planted_factor]
        if float(np.linalg.norm(true_signal)) == 0.0:
            similarity_value = None
            _add_warning(
                evaluation_warnings,
                f"planted factor {planted_factor} generated zero working-set "
                "accesses; activation alignment is undefined",
            )
        else:
            similarity_value = _cosine(
                learned_structure.activation_matrix[:, learned_factor], true_signal
            )
            alignment_values.append(similarity_value)
        alignment_rows.append(
            {
                "learned_factor_id": learned_factor,
                "planted_factor_id": planted_factor,
                "cosine_similarity": similarity_value,
            }
        )

    source_hashes = _hash_sources(source)
    report = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "source_artifact_sha256": source_hashes,
        "convergence": {
            "converged": learned_structure.converged,
            "iteration_count": learned_structure.iteration_count,
            "max_iter": config.max_iter,
            "convergence_warnings": list(
                learned_structure.convergence_warnings
            ),
        },
        "reconstruction": {
            "absolute_frobenius_error": absolute_error,
            "demand_matrix_frobenius_norm": matrix_norm,
            "normalized_frobenius_error": normalized_error,
            "sklearn_reconstruction_error": (
                learned_structure.sklearn_reconstruction_error
            ),
            "sklearn_absolute_error_difference": abs(
                learned_structure.sklearn_reconstruction_error - absolute_error
            ),
        },
        "pairwise_membership_cosine_similarity": similarity.tolist(),
        "optimal_assignment": fuzzy_per_factor,
        "fuzzy_membership_recovery": {
            "per_factor": fuzzy_per_factor,
            "aggregate": _descriptive_statistics(fuzzy_values),
        },
        "support_recovery": {
            "per_factor": support_rows,
            "aggregate": support_aggregate,
        },
        "activation_alignment": {
            "per_factor": alignment_rows,
            "aggregate": _descriptive_statistics(alignment_values),
        },
        "representative_gate": {
            "metric": "mean learned support recall > mean analytic random-support expectation",
            "mean_learned_support_recall": support_aggregate[
                "mean_learned_support_recall"
            ],
            "mean_analytic_random_support_expectation": support_aggregate[
                "mean_analytic_random_support_expectation"
            ],
            "difference": support_aggregate["difference"],
            "comparison": "strict_greater_than",
            "passed": support_aggregate["difference"] > 0.0,
        },
        "warnings": evaluation_warnings,
    }
    return RecoveryEvaluation(report=report)


def match_factors(
    learned_membership: np.ndarray, planted_membership: np.ndarray
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[str, ...]]:
    """Return all-pairs cosine similarity and globally optimal assignment."""

    learned = np.asarray(learned_membership, dtype=np.float64)
    planted = np.asarray(planted_membership, dtype=np.float64)
    if learned.ndim != 2 or planted.ndim != 2:
        raise RecoveryEvaluationError("membership inputs must be matrices")
    if learned.shape != planted.shape:
        raise RecoveryEvaluationError(
            "learned and planted membership matrices must have equal shape"
        )
    if learned.shape[0] == 0:
        raise RecoveryEvaluationError("at least one factor is required")
    if not np.all(np.isfinite(learned)) or not np.all(np.isfinite(planted)):
        raise RecoveryEvaluationError("membership matrices must be finite")
    if np.any(learned < 0.0) or np.any(planted < 0.0):
        raise RecoveryEvaluationError("membership matrices must be nonnegative")

    similarity = np.zeros((learned.shape[0], planted.shape[0]), dtype=np.float64)
    warnings: list[str] = []
    learned_norms = np.linalg.norm(learned, axis=1)
    planted_norms = np.linalg.norm(planted, axis=1)
    for factor_id, norm in enumerate(learned_norms):
        if norm == 0.0:
            warnings.append(
                f"learned factor {factor_id} has zero membership norm; pairwise "
                "cosine similarities are 0.0"
            )
    for factor_id, norm in enumerate(planted_norms):
        if norm == 0.0:
            raise RecoveryEvaluationError(
                f"planted factor {factor_id} has zero membership norm"
            )
    for learned_factor in range(learned.shape[0]):
        if learned_norms[learned_factor] == 0.0:
            continue
        for planted_factor in range(planted.shape[0]):
            raw_similarity = float(
                np.dot(learned[learned_factor], planted[planted_factor])
                / (learned_norms[learned_factor] * planted_norms[planted_factor])
            )
            similarity[learned_factor, planted_factor] = min(
                1.0, max(0.0, raw_similarity)
            )

    learned_indices, planted_indices = linear_sum_assignment(
        similarity, maximize=True
    )
    assignment = tuple(
        sorted(
            (
                (int(learned_factor), int(planted_factor))
                for learned_factor, planted_factor in zip(
                    learned_indices, planted_indices, strict=True
                )
            ),
            key=lambda pair: pair[0],
        )
    )
    return similarity, assignment, tuple(warnings)


def _validate_learned_contract(
    demand: DemandMatrix,
    learned: LearnedStructure,
    config: StructureLearnerConfig,
) -> None:
    expected_activation = (demand.X.shape[0], config.n_components)
    expected_membership = (config.n_components, demand.X.shape[1])
    if learned.activation_matrix.shape != expected_activation:
        raise RecoveryEvaluationError(
            f"activation matrix shape must be {expected_activation}"
        )
    if learned.membership_matrix.shape != expected_membership:
        raise RecoveryEvaluationError(
            f"membership matrix shape must be {expected_membership}"
        )
    if not np.array_equal(learned.window_ids, demand.window_ids):
        raise RecoveryEvaluationError("learned window IDs do not match demand")
    if not np.array_equal(learned.record_ids, demand.record_ids):
        raise RecoveryEvaluationError("learned record IDs do not match demand")
    if not np.array_equal(
        learned.factor_ids, np.arange(config.n_components, dtype=np.int64)
    ):
        raise RecoveryEvaluationError("learned factor IDs must be contiguous from zero")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RecoveryEvaluationError(f"missing evaluation artifact: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise RecoveryEvaluationError(
            f"malformed JSON in {path.name}: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise RecoveryEvaluationError(f"{path.name} root must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryEvaluationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_evaluation_truth(
    hidden: Mapping[str, Any],
    record_ids: np.ndarray,
    window_ids: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    if hidden.get("schema_version") != 1:
        raise RecoveryEvaluationError("hidden schema_version must equal 1")
    raw_memberships = hidden.get("working_set_memberships")
    if not isinstance(raw_memberships, list) or len(raw_memberships) != n_components:
        raise RecoveryEvaluationError(
            "hidden working_set_memberships must match n_components"
        )
    record_index = {int(record_id): index for index, record_id in enumerate(record_ids)}
    membership = np.zeros((n_components, len(record_ids)), dtype=np.float64)
    for expected_factor, raw_factor in enumerate(raw_memberships):
        if not isinstance(raw_factor, dict):
            raise RecoveryEvaluationError("hidden membership entries must be objects")
        working_set_id = _integer(
            "working_set_id", raw_factor.get("working_set_id"), minimum=0
        )
        if working_set_id != expected_factor:
            raise RecoveryEvaluationError("hidden working_set_ids must be ordered")
        members = raw_factor.get("members")
        if not isinstance(members, list) or not members:
            raise RecoveryEvaluationError(
                f"planted factor {expected_factor} must have nonempty support"
            )
        seen: set[int] = set()
        for raw_member in members:
            if not isinstance(raw_member, dict):
                raise RecoveryEvaluationError("hidden membership members must be objects")
            record_id = _integer(
                "membership record_id", raw_member.get("record_id"), minimum=0
            )
            if record_id not in record_index:
                raise RecoveryEvaluationError(
                    f"hidden membership references unknown record {record_id}"
                )
            if record_id in seen:
                raise RecoveryEvaluationError(
                    f"planted factor {expected_factor} repeats record {record_id}"
                )
            seen.add(record_id)
            weight = _positive_number("membership weight", raw_member.get("weight"))
            membership[expected_factor, record_index[record_id]] = weight
        total = float(membership[expected_factor].sum())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RecoveryEvaluationError(
                f"planted factor {expected_factor} membership must sum to 1; got {total}"
            )

    raw_counts = hidden.get("access_source_counts_by_window")
    if not isinstance(raw_counts, list) or len(raw_counts) != len(window_ids):
        raise RecoveryEvaluationError(
            "hidden access_source_counts_by_window must match window count"
        )
    activation = np.zeros((len(window_ids), n_components), dtype=np.float64)
    for row_index, (expected_window, raw_row) in enumerate(
        zip(window_ids, raw_counts, strict=True)
    ):
        if not isinstance(raw_row, dict):
            raise RecoveryEvaluationError("hidden source-count rows must be objects")
        window_id = _integer(
            "source-count window_id", raw_row.get("window_id"), minimum=0
        )
        if window_id != int(expected_window):
            raise RecoveryEvaluationError("hidden source-count window IDs are invalid")
        raw_factor_counts = raw_row.get("working_set_access_counts")
        if not isinstance(raw_factor_counts, list) or len(raw_factor_counts) != n_components:
            raise RecoveryEvaluationError(
                "working_set_access_counts must match n_components"
            )
        for expected_factor, raw_count in enumerate(raw_factor_counts):
            if not isinstance(raw_count, dict):
                raise RecoveryEvaluationError(
                    "working-set source-count entries must be objects"
                )
            working_set_id = _integer(
                "source-count working_set_id",
                raw_count.get("working_set_id"),
                minimum=0,
            )
            if working_set_id != expected_factor:
                raise RecoveryEvaluationError(
                    "working-set source-count IDs must be ordered"
                )
            activation[row_index, expected_factor] = _integer(
                "source access_count", raw_count.get("access_count"), minimum=0
            )
    return membership, activation


def _support_metrics(
    learned_factor_id: int,
    planted_factor_id: int,
    learned_weights: np.ndarray,
    planted_weights: np.ndarray,
    record_ids: np.ndarray,
) -> dict[str, Any]:
    true_indices = [
        index for index, weight in enumerate(planted_weights) if weight > 0.0
    ]
    support_size = len(true_indices)
    if support_size == 0:
        raise RecoveryEvaluationError("planted support must not be empty")
    ranked_indices = sorted(
        range(len(record_ids)),
        key=lambda index: (-float(learned_weights[index]), int(record_ids[index])),
    )
    predicted = set(ranked_indices[:support_size])
    true = set(true_indices)
    overlap = len(predicted & true)
    precision = overlap / support_size
    recall = overlap / support_size
    f1 = precision
    union_size = len(predicted | true)
    jaccard = overlap / union_size
    chance = support_size / len(record_ids)
    return {
        "learned_factor_id": learned_factor_id,
        "planted_factor_id": planted_factor_id,
        "true_support_size": support_size,
        "recovered_overlap_count": overlap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard_similarity": jaccard,
        "analytic_random_support_expectation": chance,
    }


def _aggregate_support_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recalls = [float(row["recall"]) for row in rows]
    chances = [float(row["analytic_random_support_expectation"]) for row in rows]
    mean_recall = math.fsum(recalls) / len(recalls)
    mean_chance = math.fsum(chances) / len(chances)
    comparisons = [
        (recall > chance) - (recall < chance)
        for recall, chance in zip(recalls, chances, strict=True)
    ]
    return {
        "factor_count": len(rows),
        "mean_learned_support_recall": mean_recall,
        "mean_analytic_random_support_expectation": mean_chance,
        "difference": mean_recall - mean_chance,
        "factors_above_chance": sum(value > 0 for value in comparisons),
        "factors_equal_to_chance": sum(value == 0 for value in comparisons),
        "factors_below_chance": sum(value < 0 for value in comparisons),
    }


def _descriptive_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": math.fsum(values) / len(values),
        "median": statistics.median(values),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return min(1.0, max(0.0, similarity))


def _hash_sources(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in SOURCE_ARTIFACT_FILENAMES:
        path = run_dir / filename
        if not path.is_file():
            raise RecoveryEvaluationError(f"missing source artifact: {filename}")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _integer(path: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryEvaluationError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise RecoveryEvaluationError(f"{path} must be at least {minimum}")
    return value


def _positive_number(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryEvaluationError(f"{path} must be a positive finite number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise RecoveryEvaluationError(f"{path} must be a positive finite number")
    return resolved


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)
