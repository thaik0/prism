"""Deterministic controlled workload generation and persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

from prism.workload.config import WorkloadConfig
from prism.workload.models import (
    ActivationTrial,
    Burst,
    HiddenGroundTruth,
    IndexedVector,
    MembershipMember,
    NamedVector,
    ObservableEvent,
    PrecursorScores,
    WindowAccessSourceCounts,
    WorkingSetMembership,
    WorkloadResult,
    WorkloadSummary,
)


SCHEMA_VERSION = 1
ARTIFACT_FILENAMES = (
    "config.json",
    "observable_events.jsonl",
    "hidden_ground_truth.json",
    "summary.json",
)


class OutputDirectoryError(ValueError):
    """Raised when workload artifacts cannot safely be written."""


def calculate_activation_probabilities(
    spontaneous_probability: float,
    precursor_probability_scale: float,
    previous_window_precursor_score: float,
) -> tuple[float, float]:
    """Return contextual and independent-opportunity combined probabilities."""

    for name, value in (
        ("spontaneous_probability", spontaneous_probability),
        ("precursor_probability_scale", precursor_probability_scale),
        ("previous_window_precursor_score", previous_window_precursor_score),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite probability")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    contextual_probability = (
        float(precursor_probability_scale)
        * float(previous_window_precursor_score)
    )
    activation_probability = 1.0 - (
        (1.0 - float(spontaneous_probability))
        * (1.0 - contextual_probability)
    )
    return contextual_probability, activation_probability


def _sample_burst_intensity(
    rng: random.Random,
    intensity_min: float,
    intensity_max: float,
    context_weight: float,
    previous_window_precursor_score: float,
) -> float:
    """Blend one random draw with intensity implied by prior-window context."""

    random_intensity = rng.uniform(intensity_min, intensity_max)
    context_implied_intensity = intensity_min + previous_window_precursor_score * (
        intensity_max - intensity_min
    )
    return (
        context_weight * context_implied_intensity
        + (1.0 - context_weight) * random_intensity
    )


def generate_workload(config: WorkloadConfig) -> WorkloadResult:
    """Generate one complete deterministic workload from an explicit seed."""

    if not isinstance(config, WorkloadConfig):
        raise TypeError("config must be a WorkloadConfig")

    rng = random.Random(config.seed)
    record_sizes = tuple(
        rng.randint(config.record_size_min_bytes, config.record_size_max_bytes)
        for _ in range(config.num_records)
    )
    memberships = _generate_memberships(config, rng)
    user_affinities = tuple(
        IndexedVector(user_id, _normalized_positive_vector(config.num_working_sets, rng))
        for user_id in range(config.num_users)
    )
    request_type_affinities = tuple(
        NamedVector(
            request_type,
            _normalized_positive_vector(config.num_working_sets, rng),
        )
        for request_type in config.request_types
    )
    user_request_type_preferences = tuple(
        IndexedVector(user_id, _normalized_positive_vector(len(config.request_types), rng))
        for user_id in range(config.num_users)
    )
    baseline_popularity = _normalized_positive_vector(config.num_records, rng)

    observable_events: list[ObservableEvent] = []
    precursor_scores: list[PrecursorScores] = []
    activation_trials: list[ActivationTrial] = []
    bursts: list[Burst] = []
    source_counts: list[WindowAccessSourceCounts] = []
    active_bursts: dict[int, Burst] = {}
    next_eligible_start = [0] * config.num_working_sets

    previous_precursor_scores = tuple(0.0 for _ in range(config.num_working_sets))
    next_event_id = 0
    next_session_id = 0
    next_request_id = 0
    total_sessions = 0
    total_requests = 0

    request_type_index = {
        request_type: index
        for index, request_type in enumerate(config.request_types)
    }
    operation_types = tuple(config.operation_type_probabilities)
    operation_weights = tuple(
        config.operation_type_probabilities[name] for name in operation_types
    )

    for window_id in range(config.num_windows):
        active_bursts = {
            working_set_id: burst
            for working_set_id, burst in active_bursts.items()
            if burst.end_window_exclusive > window_id
        }

        for working_set_id in range(config.num_working_sets):
            if (
                working_set_id in active_bursts
                or window_id < next_eligible_start[working_set_id]
            ):
                continue
            contextual_probability, activation_probability = (
                calculate_activation_probabilities(
                    config.spontaneous_activation_probability,
                    config.precursor_probability_scale,
                    previous_precursor_scores[working_set_id],
                )
            )
            activation_draw = rng.random()
            activated = activation_draw < activation_probability
            created_burst_id: int | None = None
            if activated:
                duration = rng.randint(
                    config.burst_duration_min_windows,
                    config.burst_duration_max_windows,
                )
                burst = Burst(
                    burst_id=len(bursts),
                    working_set_id=working_set_id,
                    start_window=window_id,
                    sampled_duration_windows=duration,
                    end_window_exclusive=window_id + duration,
                    intensity=_sample_burst_intensity(
                        rng,
                        config.burst_intensity_min,
                        config.burst_intensity_max,
                        config.burst_intensity_context_weight,
                        previous_precursor_scores[working_set_id],
                    ),
                )
                bursts.append(burst)
                active_bursts[working_set_id] = burst
                next_eligible_start[working_set_id] = (
                    burst.end_window_exclusive
                    + config.post_burst_cooldown_windows
                )
                created_burst_id = burst.burst_id
            activation_trials.append(
                ActivationTrial(
                    window_id=window_id,
                    working_set_id=working_set_id,
                    previous_window_precursor_score=(
                        previous_precursor_scores[working_set_id]
                    ),
                    contextual_probability=contextual_probability,
                    spontaneous_probability=(
                        config.spontaneous_activation_probability
                    ),
                    activation_probability=activation_probability,
                    activated=activated,
                    created_burst_id=created_burst_id,
                    spontaneous_component_succeeded=(
                        activated
                        and activation_draw
                        < config.spontaneous_activation_probability
                    ),
                )
            )

        window_baseline_count = 0
        window_noise_count = 0
        window_working_set_counts = [0] * config.num_working_sets
        contribution_sums = [0.0] * config.num_working_sets
        window_request_count = 0

        session_count = rng.randint(
            config.min_sessions_per_window,
            config.max_sessions_per_window,
        )
        total_sessions += session_count
        for _ in range(session_count):
            session_id = next_session_id
            next_session_id += 1
            user_id = rng.randrange(config.num_users)
            request_count = rng.randint(
                config.min_requests_per_session,
                config.max_requests_per_session,
            )
            total_requests += request_count

            for _ in range(request_count):
                request_id = next_request_id
                next_request_id += 1
                window_request_count += 1
                preference_weights = user_request_type_preferences[user_id].weights
                chosen_request_type_index = _weighted_index(preference_weights, rng)
                request_type = config.request_types[chosen_request_type_index]
                affinity = request_type_affinities[
                    request_type_index[request_type]
                ].weights
                for working_set_id in range(config.num_working_sets):
                    contribution_sums[working_set_id] += 0.5 * (
                        user_affinities[user_id].weights[working_set_id]
                        + affinity[working_set_id]
                    )

                access_count = rng.randint(
                    config.min_accesses_per_request,
                    config.max_accesses_per_request,
                )
                for _ in range(access_count):
                    source, record_id = _sample_access_source_and_record(
                        config,
                        rng,
                        active_bursts,
                        memberships,
                        baseline_popularity,
                    )
                    if source == "baseline":
                        window_baseline_count += 1
                    elif source == "noise":
                        window_noise_count += 1
                    else:
                        window_working_set_counts[source] += 1

                    operation_type = operation_types[
                        _weighted_index(operation_weights, rng)
                    ]
                    observable_events.append(
                        ObservableEvent(
                            event_index=next_event_id,
                            window_id=window_id,
                            record_id=record_id,
                            record_size_bytes=record_sizes[record_id],
                            user_id=user_id,
                            session_id=session_id,
                            request_id=request_id,
                            request_type=request_type,
                            operation_type=operation_type,
                        )
                    )
                    next_event_id += 1

        current_precursor_scores = tuple(
            min(
                1.0,
                config.num_working_sets
                * contribution_sums[working_set_id]
                / window_request_count,
            )
            for working_set_id in range(config.num_working_sets)
        )
        precursor_scores.append(
            PrecursorScores(window_id=window_id, scores=current_precursor_scores)
        )
        previous_precursor_scores = current_precursor_scores
        source_counts.append(
            WindowAccessSourceCounts(
                window_id=window_id,
                baseline_access_count=window_baseline_count,
                noise_access_count=window_noise_count,
                working_set_access_counts=tuple(window_working_set_counts),
            )
        )

    records_in_multiple_working_sets = _count_overlapping_records(
        config.num_records, memberships
    )
    total_baseline = sum(count.baseline_access_count for count in source_counts)
    total_noise = sum(count.noise_access_count for count in source_counts)
    total_working_set = sum(count.working_set_access_count for count in source_counts)

    hidden_ground_truth = HiddenGroundTruth(
        schema_version=SCHEMA_VERSION,
        seed=config.seed,
        record_sizes_bytes=record_sizes,
        working_set_memberships=memberships,
        user_working_set_affinities=user_affinities,
        request_type_working_set_affinities=request_type_affinities,
        user_request_type_preferences=user_request_type_preferences,
        baseline_record_popularity=baseline_popularity,
        precursor_scores_by_window=tuple(precursor_scores),
        activation_trials=tuple(activation_trials),
        bursts=tuple(bursts),
        access_source_counts_by_window=tuple(source_counts),
    )
    summary = WorkloadSummary(
        schema_version=SCHEMA_VERSION,
        seed=config.seed,
        num_windows=config.num_windows,
        num_records=config.num_records,
        total_sessions=total_sessions,
        total_requests=total_requests,
        total_events=len(observable_events),
        total_bursts=len(bursts),
        baseline_access_count=total_baseline,
        noise_access_count=total_noise,
        working_set_access_count=total_working_set,
        records_in_multiple_working_sets=records_in_multiple_working_sets,
    )
    return WorkloadResult(
        config=config,
        observable_events=tuple(observable_events),
        hidden_ground_truth=hidden_ground_truth,
        summary=summary,
    )


def persist_workload(
    result: WorkloadResult,
    output_dir: str | Path,
    *,
    include_cooldown_metadata: bool = False,
) -> None:
    """Write exactly four deterministic artifacts to an empty destination."""

    if not isinstance(result, WorkloadResult):
        raise TypeError("result must be a WorkloadResult")

    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise OutputDirectoryError(
                f"output path exists and is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise OutputDirectoryError(
                f"output directory must be empty: {destination}"
            )
    else:
        destination.mkdir(parents=True)

    _write_json(
        destination / "config.json",
        result.config.to_resolved_dict()
        if include_cooldown_metadata
        else result.config.to_dict(),
    )
    events_text = "".join(
        json.dumps(
            event.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for event in result.observable_events
    )
    (destination / "observable_events.jsonl").write_text(
        events_text, encoding="utf-8"
    )
    _write_json(
        destination / "hidden_ground_truth.json",
        result.hidden_ground_truth.to_dict(),
    )
    summary = result.summary.to_dict()
    if include_cooldown_metadata:
        summary["post_burst_cooldown_windows"] = (
            result.config.post_burst_cooldown_windows
        )
    _write_json(destination / "summary.json", summary)


def _generate_memberships(
    config: WorkloadConfig, rng: random.Random
) -> tuple[WorkingSetMembership, ...]:
    memberships: list[WorkingSetMembership] = []
    for working_set_id in range(config.num_working_sets):
        support_size = rng.randint(
            config.working_set_support_min,
            config.working_set_support_max,
        )
        support = sorted(rng.sample(range(config.num_records), support_size))
        weights = _normalized_positive_vector(support_size, rng)
        memberships.append(
            WorkingSetMembership(
                working_set_id=working_set_id,
                members=tuple(
                    MembershipMember(record_id=record_id, weight=weight)
                    for record_id, weight in zip(support, weights, strict=True)
                ),
            )
        )
    return tuple(memberships)


def _normalized_positive_vector(
    length: int, rng: random.Random
) -> tuple[float, ...]:
    raw_weights = tuple(
        -math.log1p(-rng.random()) or sys.float_info.min for _ in range(length)
    )
    total = math.fsum(raw_weights)
    return tuple(weight / total for weight in raw_weights)


def _weighted_index(weights: Sequence[float], rng: random.Random) -> int:
    total = math.fsum(weights)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("weighted sample requires a positive finite total")
    draw = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    return len(weights) - 1


def _sample_access_source_and_record(
    config: WorkloadConfig,
    rng: random.Random,
    active_bursts: dict[int, Burst],
    memberships: tuple[WorkingSetMembership, ...],
    baseline_popularity: tuple[float, ...],
) -> tuple[str | int, int]:
    active_working_set_ids = tuple(sorted(active_bursts))
    sources: list[str | int] = list(active_working_set_ids)
    weights = [
        config.burst_access_weight * active_bursts[working_set_id].intensity
        for working_set_id in active_working_set_ids
    ]
    sources.extend(("baseline", "noise"))
    weights.extend((config.baseline_access_weight, config.noise_access_weight))
    source = sources[_weighted_index(weights, rng)]

    if source == "baseline":
        return source, _weighted_index(baseline_popularity, rng)
    if source == "noise":
        return source, rng.randrange(config.num_records)

    membership = memberships[source]
    member_index = _weighted_index(
        tuple(member.weight for member in membership.members), rng
    )
    return source, membership.members[member_index].record_id


def _count_overlapping_records(
    num_records: int, memberships: tuple[WorkingSetMembership, ...]
) -> int:
    membership_counts = [0] * num_records
    for membership in memberships:
        for member in membership.members:
            membership_counts[member.record_id] += 1
    return sum(count > 1 for count in membership_counts)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
