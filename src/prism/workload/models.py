"""Observable and simulator-only workload result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism.workload.config import WorkloadConfig


OBSERVABLE_EVENT_FIELDS = frozenset(
    {
        "event_index",
        "window_id",
        "record_id",
        "record_size_bytes",
        "user_id",
        "session_id",
        "request_id",
        "request_type",
        "operation_type",
    }
)


@dataclass(frozen=True)
class ObservableEvent:
    """One model-plausible record access with no hidden simulator state."""

    event_index: int
    window_id: int
    record_id: int
    record_size_bytes: int
    user_id: int
    session_id: int
    request_id: int
    request_type: str
    operation_type: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "event_index": self.event_index,
            "window_id": self.window_id,
            "record_id": self.record_id,
            "record_size_bytes": self.record_size_bytes,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "request_type": self.request_type,
            "operation_type": self.operation_type,
        }


@dataclass(frozen=True)
class MembershipMember:
    record_id: int
    weight: float

    def to_dict(self) -> dict[str, int | float]:
        return {"record_id": self.record_id, "weight": self.weight}


@dataclass(frozen=True)
class WorkingSetMembership:
    working_set_id: int
    members: tuple[MembershipMember, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "working_set_id": self.working_set_id,
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class IndexedVector:
    entity_id: int
    weights: tuple[float, ...]

    def to_dict(self, id_field: str) -> dict[str, Any]:
        return {id_field: self.entity_id, "weights": list(self.weights)}


@dataclass(frozen=True)
class NamedVector:
    name: str
    weights: tuple[float, ...]

    def to_dict(self, name_field: str) -> dict[str, Any]:
        return {name_field: self.name, "weights": list(self.weights)}


@dataclass(frozen=True)
class ActivationTrial:
    window_id: int
    working_set_id: int
    previous_window_precursor_score: float
    contextual_probability: float
    spontaneous_probability: float
    activation_probability: float
    activated: bool
    created_burst_id: int | None
    spontaneous_component_succeeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "working_set_id": self.working_set_id,
            "previous_window_precursor_score": (
                self.previous_window_precursor_score
            ),
            "contextual_probability": self.contextual_probability,
            "spontaneous_probability": self.spontaneous_probability,
            "activation_probability": self.activation_probability,
            "activated": self.activated,
            "created_burst_id": self.created_burst_id,
        }


@dataclass(frozen=True)
class Burst:
    burst_id: int
    working_set_id: int
    start_window: int
    sampled_duration_windows: int
    end_window_exclusive: int
    intensity: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "burst_id": self.burst_id,
            "working_set_id": self.working_set_id,
            "start_window": self.start_window,
            "sampled_duration_windows": self.sampled_duration_windows,
            "end_window_exclusive": self.end_window_exclusive,
            "intensity": self.intensity,
        }


@dataclass(frozen=True)
class PrecursorScores:
    window_id: int
    scores: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "scores": list(self.scores)}


@dataclass(frozen=True)
class WindowAccessSourceCounts:
    window_id: int
    baseline_access_count: int
    noise_access_count: int
    working_set_access_counts: tuple[int, ...]

    @property
    def working_set_access_count(self) -> int:
        return sum(self.working_set_access_counts)

    @property
    def total_access_count(self) -> int:
        return (
            self.baseline_access_count
            + self.noise_access_count
            + self.working_set_access_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "baseline_access_count": self.baseline_access_count,
            "noise_access_count": self.noise_access_count,
            "working_set_access_count": self.working_set_access_count,
            "working_set_access_counts": [
                {"working_set_id": working_set_id, "access_count": count}
                for working_set_id, count in enumerate(
                    self.working_set_access_counts
                )
            ],
        }


@dataclass(frozen=True)
class HiddenGroundTruth:
    """Simulator-only state that must never become model input."""

    schema_version: int
    seed: int
    record_sizes_bytes: tuple[int, ...]
    working_set_memberships: tuple[WorkingSetMembership, ...]
    user_working_set_affinities: tuple[IndexedVector, ...]
    request_type_working_set_affinities: tuple[NamedVector, ...]
    user_request_type_preferences: tuple[IndexedVector, ...]
    baseline_record_popularity: tuple[float, ...]
    precursor_scores_by_window: tuple[PrecursorScores, ...]
    activation_trials: tuple[ActivationTrial, ...]
    bursts: tuple[Burst, ...]
    access_source_counts_by_window: tuple[WindowAccessSourceCounts, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "record_sizes_bytes": list(self.record_sizes_bytes),
            "working_set_memberships": [
                membership.to_dict() for membership in self.working_set_memberships
            ],
            "user_working_set_affinities": [
                vector.to_dict("user_id")
                for vector in self.user_working_set_affinities
            ],
            "request_type_working_set_affinities": [
                vector.to_dict("request_type")
                for vector in self.request_type_working_set_affinities
            ],
            "user_request_type_preferences": [
                vector.to_dict("user_id")
                for vector in self.user_request_type_preferences
            ],
            "baseline_record_popularity": list(self.baseline_record_popularity),
            "precursor_scores_by_window": [
                score.to_dict() for score in self.precursor_scores_by_window
            ],
            "activation_trials": [
                trial.to_dict() for trial in self.activation_trials
            ],
            "bursts": [burst.to_dict() for burst in self.bursts],
            "access_source_counts_by_window": [
                counts.to_dict() for counts in self.access_source_counts_by_window
            ],
        }


@dataclass(frozen=True)
class WorkloadSummary:
    schema_version: int
    seed: int
    num_windows: int
    num_records: int
    total_sessions: int
    total_requests: int
    total_events: int
    total_bursts: int
    baseline_access_count: int
    noise_access_count: int
    working_set_access_count: int
    records_in_multiple_working_sets: int

    def to_dict(self) -> dict[str, int]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "num_windows": self.num_windows,
            "num_records": self.num_records,
            "total_sessions": self.total_sessions,
            "total_requests": self.total_requests,
            "total_events": self.total_events,
            "total_bursts": self.total_bursts,
            "baseline_access_count": self.baseline_access_count,
            "noise_access_count": self.noise_access_count,
            "working_set_access_count": self.working_set_access_count,
            "records_in_multiple_working_sets": (
                self.records_in_multiple_working_sets
            ),
        }


@dataclass(frozen=True)
class WorkloadResult:
    """One run with intentionally separate observable and hidden structures."""

    config: WorkloadConfig
    observable_events: tuple[ObservableEvent, ...]
    hidden_ground_truth: HiddenGroundTruth
    summary: WorkloadSummary
