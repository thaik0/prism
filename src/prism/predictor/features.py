"""Observable-only request reconstruction and fixed predictor feature schemas."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from prism.predictor.config import SplitBoundaries
from prism.structure.demand import DemandMatrix
from prism.workload.config import WorkloadConfig
from prism.workload.models import OBSERVABLE_EVENT_FIELDS


class PredictorFeatureError(ValueError):
    """Raised when observable inputs cannot define predictor features."""


@dataclass(frozen=True)
class WindowContext:
    """Observable request and volume aggregates for one source window."""

    session_count: int
    request_count: int
    access_count: int
    user_fractions: np.ndarray
    request_type_fractions: np.ndarray

    def __post_init__(self) -> None:
        for name in ("user_fractions", "request_type_fractions"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PredictorDataset:
    """Model-visible feature matrices in deterministic example order."""

    projected_factor_demand: np.ndarray
    recent_features: np.ndarray
    context_features: np.ndarray
    recent_feature_names: tuple[str, ...]
    context_feature_names: tuple[str, ...]
    recent_continuous_indices: np.ndarray
    context_continuous_indices: np.ndarray
    feature_window_ids: np.ndarray
    target_window_ids: np.ndarray
    factor_ids: np.ndarray
    split_codes: np.ndarray
    user_ids: np.ndarray
    request_types: tuple[str, ...]
    window_contexts: tuple[WindowContext, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        array_fields = (
            "projected_factor_demand",
            "recent_features",
            "context_features",
            "recent_continuous_indices",
            "context_continuous_indices",
            "feature_window_ids",
            "target_window_ids",
            "factor_ids",
            "split_codes",
            "user_ids",
        )
        for name in array_fields:
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def example_count(self) -> int:
        return len(self.feature_window_ids)

    def split_mask(self, split_code: int) -> np.ndarray:
        return self.split_codes == split_code


def project_factor_demand(
    demand: DemandMatrix, membership_matrix: np.ndarray
) -> np.ndarray:
    """Project observable record counts through frozen normalized memberships."""

    membership = np.asarray(membership_matrix, dtype=np.float64)
    if membership.ndim != 2:
        raise PredictorFeatureError("membership_matrix must be two-dimensional")
    if membership.shape[1] != demand.X.shape[1]:
        raise PredictorFeatureError("membership record dimension does not match demand")
    if not np.all(np.isfinite(membership)) or np.any(membership < 0.0):
        raise PredictorFeatureError("membership_matrix must be finite and nonnegative")
    projected = np.asarray(demand.X, dtype=np.float64) @ membership.T
    if not np.all(np.isfinite(projected)):
        raise PredictorFeatureError("projected factor demand must be finite")
    return projected


def reconstruct_window_context(
    run_dir: str | Path,
    config: WorkloadConfig,
) -> tuple[tuple[WindowContext, ...], np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Count each unique observable request once and aggregate current context."""

    events_path = Path(run_dir) / "observable_events.jsonl"
    if not events_path.is_file():
        raise PredictorFeatureError("missing observable_events.jsonl")

    user_ids = np.arange(config.num_users, dtype=np.int64)
    request_types = tuple(sorted(config.request_types))
    request_type_index = {name: index for index, name in enumerate(request_types)}
    requests: dict[int, tuple[int, int, int, str]] = {}
    sessions: dict[int, tuple[int, int]] = {}
    access_counts = np.zeros(config.num_windows, dtype=np.int64)

    try:
        with events_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise PredictorFeatureError(
                        f"observable_events.jsonl line {line_number} is invalid"
                    )
                try:
                    event = json.loads(line, object_pairs_hook=_unique_object)
                except json.JSONDecodeError as error:
                    raise PredictorFeatureError(
                        f"malformed observable event on line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(event, dict) or set(event) != set(OBSERVABLE_EVENT_FIELDS):
                    raise PredictorFeatureError(
                        f"observable event line {line_number} has invalid fields"
                    )
                window_id = _integer(
                    "window_id", event["window_id"], 0, config.num_windows - 1
                )
                user_id = _integer("user_id", event["user_id"], 0, config.num_users - 1)
                session_id = _integer("session_id", event["session_id"], 0, None)
                request_id = _integer("request_id", event["request_id"], 0, None)
                request_type = event["request_type"]
                if request_type not in request_type_index:
                    raise PredictorFeatureError(
                        f"request_type {request_type!r} is not configured"
                    )
                session_context = (window_id, user_id)
                if session_id in sessions and sessions[session_id] != session_context:
                    raise PredictorFeatureError(
                        f"session {session_id} has inconsistent window or user"
                    )
                sessions[session_id] = session_context
                request_context = (window_id, session_id, user_id, request_type)
                if request_id in requests and requests[request_id] != request_context:
                    raise PredictorFeatureError(
                        f"request {request_id} has inconsistent observable metadata"
                    )
                requests[request_id] = request_context
                access_counts[window_id] += 1
    except UnicodeDecodeError as error:
        raise PredictorFeatureError(f"malformed UTF-8 in observable events: {error}") from error

    sessions_by_window: list[set[int]] = [set() for _ in range(config.num_windows)]
    requests_by_window = np.zeros(config.num_windows, dtype=np.int64)
    users_by_window = np.zeros((config.num_windows, config.num_users), dtype=np.int64)
    types_by_window = np.zeros(
        (config.num_windows, len(request_types)), dtype=np.int64
    )
    for request_id in sorted(requests):
        window_id, session_id, user_id, request_type = requests[request_id]
        sessions_by_window[window_id].add(session_id)
        requests_by_window[window_id] += 1
        users_by_window[window_id, user_id] += 1
        types_by_window[window_id, request_type_index[request_type]] += 1

    warnings: list[str] = []
    contexts: list[WindowContext] = []
    for window_id in range(config.num_windows):
        request_count = int(requests_by_window[window_id])
        if request_count == 0:
            user_fractions = np.zeros(config.num_users, dtype=np.float64)
            type_fractions = np.zeros(len(request_types), dtype=np.float64)
            warnings.append(
                f"window {window_id} has zero observable requests; context fractions are zero"
            )
        else:
            user_fractions = users_by_window[window_id] / request_count
            type_fractions = types_by_window[window_id] / request_count
        contexts.append(
            WindowContext(
                session_count=len(sessions_by_window[window_id]),
                request_count=request_count,
                access_count=int(access_counts[window_id]),
                user_fractions=user_fractions,
                request_type_fractions=type_fractions,
            )
        )
    return tuple(contexts), user_ids, request_types, tuple(warnings)


def build_predictor_dataset(
    run_dir: str | Path,
    demand: DemandMatrix,
    membership_matrix: np.ndarray,
    boundaries: SplitBoundaries,
    source_config: WorkloadConfig,
) -> PredictorDataset:
    """Build both fixed observable feature schemas for every usable example."""

    if demand.X.shape[0] != boundaries.num_windows:
        raise PredictorFeatureError("split boundaries do not match demand windows")
    factor_demand = project_factor_demand(demand, membership_matrix)
    num_factors = factor_demand.shape[1]
    contexts, user_ids, request_types, warnings = reconstruct_window_context(
        run_dir, source_config
    )

    recent_names = (
        "factor_demand_t",
        "factor_demand_t_minus_1",
        "factor_demand_t_minus_2",
        "factor_demand_delta_1",
        "factor_demand_mean_3",
        "session_count_t",
        "request_count_t",
        "access_count_t",
        *(f"factor_id_{factor_id}" for factor_id in range(num_factors)),
    )
    context_names = (
        *recent_names,
        *(
            f"factor_user_fraction_factor_{factor_id}_user_{user_id}"
            for factor_id in range(num_factors)
            for user_id in user_ids
        ),
        *(
            "factor_request_type_fraction_"
            f"factor_{factor_id}_request_type_{request_type}"
            for factor_id in range(num_factors)
            for request_type in request_types
        ),
    )
    recent_continuous = np.arange(8, dtype=np.int64)
    context_continuous = np.concatenate(
        [
            recent_continuous,
            np.arange(
                len(recent_names), len(context_names), dtype=np.int64
            ),
        ]
    )

    example_count = (boundaries.num_windows - 3) * num_factors
    recent = np.zeros((example_count, len(recent_names)), dtype=np.float64)
    context = np.zeros((example_count, len(context_names)), dtype=np.float64)
    feature_windows = np.empty(example_count, dtype=np.int64)
    target_windows = np.empty(example_count, dtype=np.int64)
    factor_ids = np.empty(example_count, dtype=np.int64)
    split_codes = np.empty(example_count, dtype=np.int8)
    user_block_start = len(recent_names)
    type_block_start = user_block_start + num_factors * len(user_ids)

    row = 0
    for feature_window in range(2, boundaries.num_windows - 1):
        current_context = contexts[feature_window]
        for factor_id in range(num_factors):
            d0 = factor_demand[feature_window, factor_id]
            d1 = factor_demand[feature_window - 1, factor_id]
            d2 = factor_demand[feature_window - 2, factor_id]
            values = (
                d0,
                d1,
                d2,
                d0 - d1,
                (d0 + d1 + d2) / 3.0,
                current_context.session_count,
                current_context.request_count,
                current_context.access_count,
            )
            recent[row, :8] = values
            recent[row, 8 + factor_id] = 1.0
            context[row, : len(recent_names)] = recent[row]
            user_start = user_block_start + factor_id * len(user_ids)
            context[
                row, user_start : user_start + len(user_ids)
            ] = current_context.user_fractions
            type_start = type_block_start + factor_id * len(request_types)
            context[
                row, type_start : type_start + len(request_types)
            ] = current_context.request_type_fractions
            target_window = feature_window + 1
            feature_windows[row] = feature_window
            target_windows[row] = target_window
            factor_ids[row] = factor_id
            split_codes[row] = boundaries.split_code_for_target(target_window)
            row += 1

    return PredictorDataset(
        projected_factor_demand=factor_demand,
        recent_features=recent,
        context_features=context,
        recent_feature_names=recent_names,
        context_feature_names=context_names,
        recent_continuous_indices=recent_continuous,
        context_continuous_indices=context_continuous,
        feature_window_ids=feature_windows,
        target_window_ids=target_windows,
        factor_ids=factor_ids,
        split_codes=split_codes,
        user_ids=user_ids,
        request_types=request_types,
        window_contexts=contexts,
        warnings=warnings,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PredictorFeatureError(f"duplicate observable event field: {key}")
        result[key] = value
    return result


def _integer(
    name: str,
    value: Any,
    minimum: int,
    maximum: int | None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredictorFeatureError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise PredictorFeatureError(f"{name} is outside the configured range")
    return value
