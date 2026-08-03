"""Byte-constrained simulated residency and promotion episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class StorageStateError(ValueError):
    """Raised when a residency transition violates storage invariants."""


@dataclass
class PromotionEpisode:
    record_id: int
    size_bytes: int
    began_in_test: bool
    resident_accesses: int = 0


@dataclass(frozen=True)
class Migration:
    record_id: int
    size_bytes: int
    kind: str
    episode: PromotionEpisode | None = None


class StorageState:
    """One policy's independent fast-tier state; slow storage is implicit."""

    def __init__(self, record_sizes: Mapping[int, int], capacity_bytes: int) -> None:
        if isinstance(capacity_bytes, bool) or capacity_bytes <= 0:
            raise StorageStateError("capacity_bytes must be positive")
        sizes = dict(record_sizes)
        if not sizes or any(
            isinstance(record_id, bool)
            or not isinstance(record_id, int)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            for record_id, size in sizes.items()
        ):
            raise StorageStateError("record sizes must map integer IDs to positive bytes")
        self.record_sizes = sizes
        self.capacity_bytes = capacity_bytes
        self.resident: set[int] = set()
        self._episodes: dict[int, PromotionEpisode] = {}

    @property
    def resident_bytes(self) -> int:
        return sum(self.record_sizes[record_id] for record_id in self.resident)

    def fits(self, record_id: int) -> bool:
        self._require_record(record_id)
        return self.record_sizes[record_id] <= self.capacity_bytes

    def can_admit(self, record_id: int) -> bool:
        self._require_record(record_id)
        return (
            record_id in self.resident
            or self.resident_bytes + self.record_sizes[record_id]
            <= self.capacity_bytes
        )

    def promote(self, record_id: int, *, began_in_test: bool) -> Migration:
        self._require_record(record_id)
        if record_id in self.resident:
            raise StorageStateError(f"record {record_id} is already resident")
        if not self.fits(record_id):
            raise StorageStateError(f"record {record_id} is oversized")
        if not self.can_admit(record_id):
            raise StorageStateError(f"record {record_id} does not fit")
        episode = PromotionEpisode(
            record_id, self.record_sizes[record_id], began_in_test
        )
        self.resident.add(record_id)
        self._episodes[record_id] = episode
        self._require_capacity()
        return Migration(record_id, self.record_sizes[record_id], "promotion")

    def evict(self, record_id: int) -> Migration:
        if record_id not in self.resident:
            raise StorageStateError(f"record {record_id} is not resident")
        self.resident.remove(record_id)
        episode = self._episodes.pop(record_id)
        self._require_capacity()
        return Migration(
            record_id, self.record_sizes[record_id], "eviction", episode
        )

    def note_access(self, record_id: int) -> bool:
        self._require_record(record_id)
        hit = record_id in self.resident
        if hit:
            self._episodes[record_id].resident_accesses += 1
        return hit

    def active_test_episodes(self) -> tuple[PromotionEpisode, ...]:
        return tuple(
            self._episodes[record_id]
            for record_id in sorted(self._episodes)
            if self._episodes[record_id].began_in_test
        )

    def _require_record(self, record_id: int) -> None:
        if record_id not in self.record_sizes:
            raise StorageStateError(f"unknown record {record_id}")

    def _require_capacity(self) -> None:
        if self.resident_bytes > self.capacity_bytes:
            raise StorageStateError("fast-tier capacity exceeded")
        if set(self._episodes) != self.resident:
            raise StorageStateError("promotion episodes and residency disagree")
