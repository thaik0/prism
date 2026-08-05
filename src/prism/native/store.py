"""Thin immutable Python values around the private native storage binding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from prism.native._binding import native
from prism.native.errors import NativeStoreError


@dataclass(frozen=True, slots=True)
class BuildSummary:
    format_version: int
    record_count: int
    data_file_bytes: int
    store_data_sha256: str
    store_index_sha256: str


@dataclass(frozen=True, slots=True)
class ReadResult:
    payload: bytes
    tier: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class PromotionResult:
    moved: bool
    record_id: int
    bytes_moved: int


@dataclass(frozen=True, slots=True)
class EvictionResult:
    moved: bool
    record_id: int
    bytes_moved: int


@dataclass(frozen=True, slots=True)
class ResidencySnapshot:
    resident_record_ids: tuple[int, ...]
    resident_bytes: int
    capacity_bytes: int


@dataclass(frozen=True, slots=True)
class TargetSetResult:
    incoming_record_ids: tuple[int, ...]
    outgoing_record_ids: tuple[int, ...]
    promotion_count: int
    promotion_bytes: int
    eviction_count: int
    eviction_bytes: int
    target_changed: bool
    residency: ResidencySnapshot


@dataclass(frozen=True, slots=True)
class RecordMetadata:
    record_id: int
    byte_offset: int
    byte_length: int
    crc32: int


@dataclass(frozen=True, slots=True)
class StoreStats:
    successful_fast_reads: int
    successful_fast_read_bytes: int
    successful_slow_reads: int
    successful_slow_read_bytes: int
    promotion_source_reads: int
    promotion_source_read_bytes: int
    committed_promotions: int
    committed_promotion_bytes: int
    committed_evictions: int
    committed_eviction_bytes: int
    target_set_calls: int
    successful_target_set_calls: int
    failed_target_set_calls: int
    aborted_staged_bytes: int
    current_resident_records: int
    current_resident_bytes: int
    resident_byte_high_water_mark: int
    failures_by_code: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "successful_fast_reads",
                "successful_fast_read_bytes",
                "successful_slow_reads",
                "successful_slow_read_bytes",
                "promotion_source_reads",
                "promotion_source_read_bytes",
                "committed_promotions",
                "committed_promotion_bytes",
                "committed_evictions",
                "committed_eviction_bytes",
                "target_set_calls",
                "successful_target_set_calls",
                "failed_target_set_calls",
                "aborted_staged_bytes",
                "current_resident_records",
                "current_resident_bytes",
                "resident_byte_high_water_mark",
            )
        } | {"failures_by_code": dict(self.failures_by_code)}


def _translate(error: BaseException, operation: str) -> NativeStoreError:
    return NativeStoreError(
        code=str(getattr(error, "code")),
        message=str(getattr(error, "message")),
        record_id=getattr(error, "record_id"),
        offset=getattr(error, "offset"),
        path=getattr(error, "path"),
        operation=operation,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_store(
    records: Sequence[tuple[int, bytes]], output_dir: str | Path
) -> BuildSummary:
    """Build and fully verify a native store from immutable payload bytes."""

    destination = Path(output_dir)
    try:
        result = native.build_store(records, destination)
    except native.NativeStoreError as error:
        raise _translate(error, "build_store") from error
    data_path = destination / "store.data"
    index_path = destination / "store.index"
    if not data_path.is_file() or not index_path.is_file():
        raise RuntimeError("native builder returned without both store files")
    return BuildSummary(
        format_version=int(result["format_version"]),
        record_count=int(result["record_count"]),
        data_file_bytes=int(result["data_file_bytes"]),
        store_data_sha256=_sha256(data_path),
        store_index_sha256=_sha256(index_path),
    )


def _snapshot(value: Mapping[str, object]) -> ResidencySnapshot:
    return ResidencySnapshot(
        resident_record_ids=tuple(int(item) for item in value["resident_record_ids"]),
        resident_bytes=int(value["resident_bytes"]),
        capacity_bytes=int(value["capacity_bytes"]),
    )


class TieredStore:
    """Public synchronous owner of one native two-tier store."""

    __slots__ = ("_store",)

    def __init__(self, native_store: object) -> None:
        self._store = native_store

    @classmethod
    def open(
        cls, store_dir: str | Path, fast_capacity_bytes: int
    ) -> TieredStore:
        try:
            opened = native.TieredStore.open(store_dir, fast_capacity_bytes)
        except native.NativeStoreError as error:
            raise _translate(error, "open") from error
        return cls(opened)

    def read(self, record_id: int) -> ReadResult:
        try:
            result = self._store.read(record_id)
        except native.NativeStoreError as error:
            raise _translate(error, "read") from error
        return ReadResult(
            payload=bytes(result["payload"]),
            tier=str(result["tier"]),
            byte_count=int(result["byte_count"]),
        )

    def promote(self, record_id: int) -> PromotionResult:
        try:
            result = self._store.promote(record_id)
        except native.NativeStoreError as error:
            raise _translate(error, "promote") from error
        return PromotionResult(
            moved=bool(result["moved"]),
            record_id=int(result["record_id"]),
            bytes_moved=int(result["bytes_moved"]),
        )

    def evict(self, record_id: int) -> EvictionResult:
        try:
            result = self._store.evict(record_id)
        except native.NativeStoreError as error:
            raise _translate(error, "evict") from error
        return EvictionResult(
            moved=bool(result["moved"]),
            record_id=int(result["record_id"]),
            bytes_moved=int(result["bytes_moved"]),
        )

    def apply_target_set(self, record_ids: Sequence[int]) -> TargetSetResult:
        try:
            result = self._store.apply_target_set(record_ids)
        except native.NativeStoreError as error:
            raise _translate(error, "apply_target_set") from error
        return TargetSetResult(
            incoming_record_ids=tuple(int(item) for item in result["incoming_record_ids"]),
            outgoing_record_ids=tuple(int(item) for item in result["outgoing_record_ids"]),
            promotion_count=int(result["promotion_count"]),
            promotion_bytes=int(result["promotion_bytes"]),
            eviction_count=int(result["eviction_count"]),
            eviction_bytes=int(result["eviction_bytes"]),
            target_changed=bool(result["target_changed"]),
            residency=_snapshot(result["residency"]),
        )

    def snapshot(self) -> ResidencySnapshot:
        return _snapshot(self._store.snapshot())

    def stats(self) -> StoreStats:
        result = self._store.stats()
        scalar_names = (
            "successful_fast_reads",
            "successful_fast_read_bytes",
            "successful_slow_reads",
            "successful_slow_read_bytes",
            "promotion_source_reads",
            "promotion_source_read_bytes",
            "committed_promotions",
            "committed_promotion_bytes",
            "committed_evictions",
            "committed_eviction_bytes",
            "target_set_calls",
            "successful_target_set_calls",
            "failed_target_set_calls",
            "aborted_staged_bytes",
            "current_resident_records",
            "current_resident_bytes",
            "resident_byte_high_water_mark",
        )
        values = {name: int(result[name]) for name in scalar_names}
        failures = MappingProxyType(
            {str(code): int(count) for code, count in result["failures_by_code"].items()}
        )
        return StoreStats(**values, failures_by_code=failures)

    def record_metadata(self, record_id: int) -> RecordMetadata:
        try:
            result = self._store.record_metadata(record_id)
        except native.NativeStoreError as error:
            raise _translate(error, "record_metadata") from error
        return RecordMetadata(
            record_id=int(result["record_id"]),
            byte_offset=int(result["byte_offset"]),
            byte_length=int(result["byte_length"]),
            crc32=int(result["crc32"]),
        )
