"""Independent Python reference ledger for native storage semantics."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence
import zlib

from prism.native.store import (
    EvictionResult,
    PromotionResult,
    ReadResult,
    RecordMetadata,
    ResidencySnapshot,
    StoreStats,
    TargetSetResult,
)


NATIVE_ERROR_CODES = (
    "unknown_record",
    "duplicate_record",
    "invalid_target_set",
    "insufficient_capacity",
    "oversized_record",
    "index_corrupt",
    "unsupported_format_version",
    "data_file_mismatch",
    "truncated_read",
    "checksum_mismatch",
    "io_error",
    "allocation_failure",
    "arithmetic_overflow",
    "invalid_configuration",
    "destination_exists",
    "malformed_manifest",
    "malformed_trace",
)


@dataclass(frozen=True, slots=True)
class ReferenceStoreError(Exception):
    code: str
    message: str
    record_id: int | None = None
    offset: int | None = None
    path: str | None = None

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class _InjectedReadFailure:
    code: str
    offset: int | None


class ReferenceLedger:
    """Transactional expected state that never consults native results."""

    def __init__(self, payloads: Mapping[int, bytes], capacity_bytes: int) -> None:
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int):
            raise TypeError("capacity_bytes must be a non-boolean integer")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if not payloads:
            raise ValueError("payloads must be nonempty")
        self._payloads = {
            int(record_id): bytes(payload)
            for record_id, payload in sorted(payloads.items())
        }
        if any(not payload for payload in self._payloads.values()):
            raise ValueError("payloads must be nonempty bytes")
        if not any(len(payload) <= capacity_bytes for payload in self._payloads.values()):
            raise ValueError("capacity cannot hold any record")
        self.capacity_bytes = capacity_bytes
        self._resident: set[int] = set()
        self._resident_bytes = 0
        self._high_water_mark = 0
        self._counters = {
            "successful_fast_reads": 0,
            "successful_fast_read_bytes": 0,
            "successful_slow_reads": 0,
            "successful_slow_read_bytes": 0,
            "promotion_source_reads": 0,
            "promotion_source_read_bytes": 0,
            "committed_promotions": 0,
            "committed_promotion_bytes": 0,
            "committed_evictions": 0,
            "committed_eviction_bytes": 0,
            "target_set_calls": 0,
            "successful_target_set_calls": 0,
            "failed_target_set_calls": 0,
            "aborted_staged_bytes": 0,
        }
        self._failures = {code: 0 for code in NATIVE_ERROR_CODES}
        self._read_failures: dict[int, _InjectedReadFailure] = {}
        self._metadata = self._build_metadata()

    def clone(self) -> ReferenceLedger:
        return copy.deepcopy(self)

    @property
    def resident_record_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._resident))

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def fits(self, record_id: int) -> bool:
        return self._size(record_id) <= self.capacity_bytes

    def can_admit(self, record_id: int) -> bool:
        if record_id in self._resident:
            return True
        return self._resident_bytes + self._size(record_id) <= self.capacity_bytes

    def mark_read_failure(
        self, record_id: int, code: str, *, offset: int | None = None
    ) -> None:
        if record_id not in self._payloads:
            raise ValueError("cannot corrupt an unknown reference record")
        if code not in {"checksum_mismatch", "truncated_read", "io_error"}:
            raise ValueError("unsupported injected read failure code")
        self._read_failures[record_id] = _InjectedReadFailure(code, offset)

    def clear_read_failure(self, record_id: int) -> None:
        self._read_failures.pop(record_id, None)

    def record_metadata(self, record_id: int) -> RecordMetadata:
        try:
            return self._metadata[record_id]
        except KeyError as error:
            raise ReferenceStoreError(
                "unknown_record",
                "record ID is not present in the store",
                record_id,
            ) from error

    def read(self, record_id: int) -> ReadResult:
        self._require_known(record_id)
        payload = self._payloads[record_id]
        if record_id in self._resident:
            self._counters["successful_fast_reads"] += 1
            self._counters["successful_fast_read_bytes"] += len(payload)
            return ReadResult(payload, "fast", len(payload))
        try:
            self._read_slow(record_id)
        except ReferenceStoreError as error:
            self._raise_failure(error)
        self._counters["successful_slow_reads"] += 1
        self._counters["successful_slow_read_bytes"] += len(payload)
        return ReadResult(payload, "slow", len(payload))

    def promote(self, record_id: int) -> PromotionResult:
        self._require_known(record_id)
        if record_id in self._resident:
            return PromotionResult(False, record_id, 0)
        size = self._size(record_id)
        if size > self.capacity_bytes:
            self._raise_failure(
                ReferenceStoreError(
                    "oversized_record",
                    "record is larger than total fast-tier capacity",
                    record_id,
                )
            )
        if size > self.capacity_bytes - self._resident_bytes:
            self._raise_failure(
                ReferenceStoreError(
                    "insufficient_capacity",
                    "remaining fast-tier capacity is insufficient",
                    record_id,
                )
            )
        try:
            self._read_slow(record_id)
        except ReferenceStoreError as error:
            self._raise_failure(error)
        self._counters["promotion_source_reads"] += 1
        self._counters["promotion_source_read_bytes"] += size
        self._resident.add(record_id)
        self._resident_bytes += size
        self._counters["committed_promotions"] += 1
        self._counters["committed_promotion_bytes"] += size
        self._update_high_water_mark()
        return PromotionResult(True, record_id, size)

    def evict(self, record_id: int) -> EvictionResult:
        self._require_known(record_id)
        if record_id not in self._resident:
            return EvictionResult(False, record_id, 0)
        size = self._size(record_id)
        self._resident.remove(record_id)
        self._resident_bytes -= size
        self._counters["committed_evictions"] += 1
        self._counters["committed_eviction_bytes"] += size
        return EvictionResult(True, record_id, size)

    def apply_target_set(self, record_ids: Sequence[int]) -> TargetSetResult:
        self._counters["target_set_calls"] += 1
        target = sorted(record_ids)
        if len(set(target)) != len(target):
            self._target_failure(
                ReferenceStoreError(
                    "invalid_target_set",
                    "target set contains a duplicate record ID",
                ),
                staged_bytes=0,
            )
        total_bytes = 0
        for record_id in target:
            if record_id not in self._payloads:
                self._target_failure(
                    ReferenceStoreError(
                        "unknown_record",
                        "target set contains an unknown record ID",
                        record_id,
                    ),
                    staged_bytes=0,
                )
            size = self._size(record_id)
            if size > self.capacity_bytes:
                self._target_failure(
                    ReferenceStoreError(
                        "oversized_record",
                        "target set contains an oversized record",
                        record_id,
                    ),
                    staged_bytes=0,
                )
            total_bytes += size
        if total_bytes > self.capacity_bytes:
            self._target_failure(
                ReferenceStoreError(
                    "insufficient_capacity",
                    "target set exceeds fast-tier capacity",
                ),
                staged_bytes=0,
            )

        current = sorted(self._resident)
        if current == target:
            self._counters["successful_target_set_calls"] += 1
            return TargetSetResult(
                (), (), 0, 0, 0, 0, False, self.snapshot()
            )
        incoming = sorted(set(target) - self._resident)
        outgoing = sorted(self._resident - set(target))
        staged_bytes = 0
        for record_id in incoming:
            try:
                self._read_slow(record_id)
            except ReferenceStoreError as error:
                self._target_failure(error, staged_bytes=staged_bytes)
            size = self._size(record_id)
            self._counters["promotion_source_reads"] += 1
            self._counters["promotion_source_read_bytes"] += size
            staged_bytes += size

        incoming_bytes = sum(self._size(record_id) for record_id in incoming)
        outgoing_bytes = sum(self._size(record_id) for record_id in outgoing)
        self._resident = set(target)
        self._resident_bytes = total_bytes
        self._counters["committed_promotions"] += len(incoming)
        self._counters["committed_promotion_bytes"] += incoming_bytes
        self._counters["committed_evictions"] += len(outgoing)
        self._counters["committed_eviction_bytes"] += outgoing_bytes
        self._counters["successful_target_set_calls"] += 1
        self._update_high_water_mark()
        return TargetSetResult(
            tuple(incoming),
            tuple(outgoing),
            len(incoming),
            incoming_bytes,
            len(outgoing),
            outgoing_bytes,
            True,
            self.snapshot(),
        )

    def snapshot(self) -> ResidencySnapshot:
        return ResidencySnapshot(
            self.resident_record_ids,
            self._resident_bytes,
            self.capacity_bytes,
        )

    def stats(self) -> StoreStats:
        return StoreStats(
            **self._counters,
            current_resident_records=len(self._resident),
            current_resident_bytes=self._resident_bytes,
            resident_byte_high_water_mark=self._high_water_mark,
            failures_by_code=MappingProxyType(dict(self._failures)),
        )

    def _build_metadata(self) -> dict[int, RecordMetadata]:
        result = {}
        offset = 0
        for record_id, payload in self._payloads.items():
            result[record_id] = RecordMetadata(
                record_id,
                offset,
                len(payload),
                zlib.crc32(payload),
            )
            offset += len(payload)
        return result

    def _size(self, record_id: int) -> int:
        self._require_known(record_id)
        return len(self._payloads[record_id])

    def _require_known(self, record_id: int) -> None:
        if record_id not in self._payloads:
            self._raise_failure(
                ReferenceStoreError(
                    "unknown_record",
                    "record ID is not present in the store",
                    record_id,
                )
            )

    def _read_slow(self, record_id: int) -> bytes:
        failure = self._read_failures.get(record_id)
        if failure is not None:
            metadata = self._metadata[record_id]
            messages = {
                "checksum_mismatch": "record checksum does not match metadata",
                "truncated_read": "data file ended before the record was complete",
                "io_error": "offset-based data read failed",
            }
            raise ReferenceStoreError(
                failure.code,
                messages[failure.code],
                record_id,
                metadata.byte_offset if failure.offset is None else failure.offset,
                "store.data",
            )
        return self._payloads[record_id]

    def _raise_failure(self, error: ReferenceStoreError) -> None:
        self._failures[error.code] += 1
        raise error

    def _target_failure(
        self, error: ReferenceStoreError, *, staged_bytes: int
    ) -> None:
        self._counters["failed_target_set_calls"] += 1
        self._counters["aborted_staged_bytes"] += staged_bytes
        self._raise_failure(error)

    def _update_high_water_mark(self) -> None:
        self._high_water_mark = max(
            self._high_water_mark, self._resident_bytes
        )
