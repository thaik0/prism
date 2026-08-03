"""Simulator-only matching, labels, and diagnostic targets for Milestone 3."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from prism.predictor.features import PredictorDataset
from prism.structure.evaluate import match_factors


class PredictorTargetError(ValueError):
    """Raised when controlled simulator targets are invalid or incompatible."""


@dataclass(frozen=True)
class PredictorTargets:
    """Hidden controlled labels aligned to model-visible examples."""

    activation: np.ndarray
    intensity: np.ndarray
    eligible: np.ndarray
    realized_next_window_accesses: np.ndarray
    learned_to_planted: np.ndarray
    matching_report: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "activation",
            "intensity",
            "eligible",
            "realized_next_window_accesses",
            "learned_to_planted",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def build_predictor_targets(
    run_dir: str | Path,
    learned_membership: np.ndarray,
    record_ids: np.ndarray,
    dataset: PredictorDataset,
) -> PredictorTargets:
    """Freeze factor matching, then align next-window hidden targets."""

    hidden = _load_json(Path(run_dir) / "hidden_ground_truth.json")
    learned = np.asarray(learned_membership, dtype=np.float64)
    planted = _planted_memberships(hidden, record_ids, learned.shape[0])
    similarity, assignment, matching_warnings = match_factors(learned, planted)
    mapping = np.empty(learned.shape[0], dtype=np.int64)
    for learned_factor, planted_factor in assignment:
        mapping[learned_factor] = planted_factor

    trials: dict[tuple[int, int], dict[str, Any]] = {}
    for raw_trial in hidden.get("activation_trials", []):
        if not isinstance(raw_trial, dict):
            raise PredictorTargetError("activation trials must be objects")
        window_id = _integer("trial.window_id", raw_trial.get("window_id"), 0)
        working_set_id = _integer(
            "trial.working_set_id", raw_trial.get("working_set_id"), 0
        )
        key = (window_id, working_set_id)
        if key in trials:
            raise PredictorTargetError(f"duplicate activation trial {key}")
        if not isinstance(raw_trial.get("activated"), bool):
            raise PredictorTargetError("trial.activated must be boolean")
        trials[key] = raw_trial

    bursts = hidden.get("bursts")
    if not isinstance(bursts, list):
        raise PredictorTargetError("hidden bursts must be a list")
    source_counts = _source_access_matrix(hidden, learned.shape[0])
    activation = np.zeros(dataset.example_count, dtype=np.int8)
    intensity = np.full(dataset.example_count, np.nan, dtype=np.float64)
    eligible = np.zeros(dataset.example_count, dtype=np.bool_)
    realized = np.zeros(dataset.example_count, dtype=np.float64)

    for index, (target_window, learned_factor) in enumerate(
        zip(dataset.target_window_ids, dataset.factor_ids, strict=True)
    ):
        planted_factor = int(mapping[learned_factor])
        key = (int(target_window), planted_factor)
        trial = trials.get(key)
        eligible[index] = trial is not None
        if trial is not None and trial["activated"]:
            activation[index] = 1
            burst_id = _integer(
                "trial.created_burst_id", trial.get("created_burst_id"), 0
            )
            if burst_id >= len(bursts) or not isinstance(bursts[burst_id], dict):
                raise PredictorTargetError("successful trial references an invalid burst")
            burst = bursts[burst_id]
            if (
                burst.get("working_set_id") != planted_factor
                or burst.get("start_window") != int(target_window)
            ):
                raise PredictorTargetError("successful trial and burst are inconsistent")
            intensity[index] = _positive_number("burst.intensity", burst.get("intensity"))
        realized[index] = source_counts[int(target_window), planted_factor]

    matching_rows = []
    support_rows = []
    for learned_factor, planted_factor in assignment:
        cosine = float(similarity[learned_factor, planted_factor])
        matching_rows.append(
            {
                "learned_factor_id": learned_factor,
                "planted_factor_id": planted_factor,
                "cosine_similarity": cosine,
            }
        )
        support_size = int(np.count_nonzero(planted[planted_factor] > 0.0))
        ranked = sorted(
            range(len(record_ids)),
            key=lambda index: (
                -float(learned[learned_factor, index]), int(record_ids[index])
            ),
        )
        predicted = set(ranked[:support_size])
        expected = set(np.flatnonzero(planted[planted_factor] > 0.0).tolist())
        recall = len(predicted & expected) / support_size
        chance = support_size / len(record_ids)
        support_rows.append(
            {
                "learned_factor_id": learned_factor,
                "planted_factor_id": planted_factor,
                "support_size": support_size,
                "recall": recall,
                "analytic_random_support_expectation": chance,
            }
        )
    matching_report = {
        "pairwise_membership_cosine_similarity": similarity.tolist(),
        "optimal_assignment": matching_rows,
        "fuzzy_cosine_mean": math.fsum(
            row["cosine_similarity"] for row in matching_rows
        )
        / len(matching_rows),
        "support_recovery": {
            "per_factor": support_rows,
            "mean_recall": math.fsum(row["recall"] for row in support_rows)
            / len(support_rows),
            "mean_analytic_random_support_expectation": math.fsum(
                row["analytic_random_support_expectation"] for row in support_rows
            )
            / len(support_rows),
        },
        "warnings": list(matching_warnings),
    }
    return PredictorTargets(
        activation=activation,
        intensity=intensity,
        eligible=eligible,
        realized_next_window_accesses=realized,
        learned_to_planted=mapping,
        matching_report=matching_report,
    )


def _planted_memberships(
    hidden: dict[str, Any], record_ids: np.ndarray, factor_count: int
) -> np.ndarray:
    raw = hidden.get("working_set_memberships")
    if not isinstance(raw, list) or len(raw) != factor_count:
        raise PredictorTargetError("planted membership count must match learned factors")
    record_index = {int(record_id): index for index, record_id in enumerate(record_ids)}
    matrix = np.zeros((factor_count, len(record_ids)), dtype=np.float64)
    for expected_factor, row in enumerate(raw):
        if not isinstance(row, dict) or row.get("working_set_id") != expected_factor:
            raise PredictorTargetError("planted membership IDs must be ordered")
        members = row.get("members")
        if not isinstance(members, list) or not members:
            raise PredictorTargetError("planted memberships must have nonempty support")
        for member in members:
            if not isinstance(member, dict):
                raise PredictorTargetError("planted membership members must be objects")
            record_id = _integer("membership.record_id", member.get("record_id"), 0)
            if record_id not in record_index:
                raise PredictorTargetError("planted membership references unknown record")
            matrix[expected_factor, record_index[record_id]] = _positive_number(
                "membership.weight", member.get("weight")
            )
        if not math.isclose(
            float(matrix[expected_factor].sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise PredictorTargetError("planted membership must sum to one")
    return matrix


def _source_access_matrix(hidden: dict[str, Any], factor_count: int) -> np.ndarray:
    rows = hidden.get("access_source_counts_by_window")
    if not isinstance(rows, list) or not rows:
        raise PredictorTargetError("hidden source counts must be a nonempty list")
    matrix = np.zeros((len(rows), factor_count), dtype=np.float64)
    for expected_window, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("window_id") != expected_window:
            raise PredictorTargetError("hidden source-count windows must be ordered")
        counts = row.get("working_set_access_counts")
        if not isinstance(counts, list) or len(counts) != factor_count:
            raise PredictorTargetError("working-set source counts have invalid shape")
        for expected_factor, item in enumerate(counts):
            if not isinstance(item, dict) or item.get("working_set_id") != expected_factor:
                raise PredictorTargetError("working-set source-count IDs must be ordered")
            matrix[expected_window, expected_factor] = _integer(
                "source_count.access_count", item.get("access_count"), 0
            )
    return matrix


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PredictorTargetError("missing hidden_ground_truth.json")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise PredictorTargetError(f"malformed hidden ground truth: {error.msg}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PredictorTargetError("hidden ground truth must use schema version 1")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PredictorTargetError(f"duplicate hidden field: {key}")
        result[key] = value
    return result


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PredictorTargetError(f"{name} must be an integer at least {minimum}")
    return value


def _positive_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictorTargetError(f"{name} must be a positive finite number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise PredictorTargetError(f"{name} must be a positive finite number")
    return resolved
