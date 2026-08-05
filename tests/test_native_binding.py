from __future__ import annotations

from pathlib import Path

import pytest


native = pytest.importorskip(
    "prism._native",
    reason="private native extension is exercised after editable or wheel installation",
)


def _build(tmp_path: Path, records=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store_dir = tmp_path / "store"
    summary = native.build_store(
        records or [(3, b"ccc"), (1, b"a"), (2, b"bb")], store_dir
    )
    return store_dir, summary


def test_private_extension_builder_sorts_verifies_and_rejects_bad_inputs(
    tmp_path: Path,
) -> None:
    store_dir, summary = _build(tmp_path)

    assert summary == {
        "format_version": 1,
        "record_count": 3,
        "data_file_bytes": 6,
    }
    assert (store_dir / "store.data").read_bytes() == b"abbccc"

    with pytest.raises(TypeError, match="tuple"):
        native.build_store([[1, b"a"]], tmp_path / "list-row")
    with pytest.raises(ValueError, match="exactly two"):
        native.build_store([(1, b"a", b"b")], tmp_path / "long-row")
    with pytest.raises(TypeError, match="immutable bytes"):
        native.build_store([(1, bytearray(b"a"))], tmp_path / "mutable")
    with pytest.raises(TypeError, match="non-boolean"):
        native.build_store([(True, b"a")], tmp_path / "boolean")
    with pytest.raises(OverflowError):
        native.build_store([(-1, b"a")], tmp_path / "negative")
    with pytest.raises(OverflowError):
        native.build_store([(2**64, b"a")], tmp_path / "overflow")
    with pytest.raises(TypeError, match="finite sequence"):
        native.build_store(((index, b"a") for index in range(2)), tmp_path / "gen")


def test_private_extension_preserves_native_builder_errors(tmp_path: Path) -> None:
    with pytest.raises(native.NativeStoreError) as duplicate:
        native.build_store([(1, b"a"), (1, b"b")], tmp_path / "duplicate")
    assert duplicate.value.code == "duplicate_record"
    assert duplicate.value.record_id == 1
    assert duplicate.value.offset is None
    assert duplicate.value.path is None

    with pytest.raises(native.NativeStoreError) as empty:
        native.build_store([(1, b"")], tmp_path / "empty")
    assert empty.value.code == "malformed_manifest"

    store_dir, _ = _build(tmp_path / "occupied-source")
    with pytest.raises(native.NativeStoreError) as occupied:
        native.build_store([(4, b"d")], store_dir)
    assert occupied.value.code == "destination_exists"
    assert occupied.value.path == str(store_dir)


def test_private_store_read_movement_metadata_snapshot_and_stats(tmp_path: Path) -> None:
    store_dir, _ = _build(tmp_path)
    store = native.TieredStore.open(store_dir, 3)

    assert store.record_metadata(2) == {
        "record_id": 2,
        "byte_offset": 1,
        "byte_length": 2,
        "crc32": 3048086446,
    }
    assert store.read(2) == {"payload": b"bb", "tier": "slow", "byte_count": 2}
    assert store.promote(2) == {"moved": True, "record_id": 2, "bytes_moved": 2}
    assert store.promote(2) == {"moved": False, "record_id": 2, "bytes_moved": 0}
    assert store.read(2) == {"payload": b"bb", "tier": "fast", "byte_count": 2}
    assert store.evict(2) == {"moved": True, "record_id": 2, "bytes_moved": 2}
    assert store.evict(2) == {"moved": False, "record_id": 2, "bytes_moved": 0}
    assert store.snapshot() == {
        "resident_record_ids": [],
        "resident_bytes": 0,
        "capacity_bytes": 3,
    }
    stats = store.stats()
    assert stats["successful_slow_reads"] == 1
    assert stats["successful_fast_reads"] == 1
    assert stats["promotion_source_reads"] == 1
    assert stats["committed_promotions"] == 1
    assert stats["committed_evictions"] == 1
    assert stats["failures_by_code"]["unknown_record"] == 0
    assert set(stats["failures_by_code"]) == {
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
    }


def test_private_target_set_reports_exact_changes_and_failure_atomicity(
    tmp_path: Path,
) -> None:
    store_dir, _ = _build(tmp_path, [(1, b"aa"), (2, b"bb"), (3, b"cc")])
    store = native.TieredStore.open(store_dir, 4)

    first = store.apply_target_set([2, 1])
    assert first == {
        "incoming_record_ids": [1, 2],
        "outgoing_record_ids": [],
        "promotion_count": 2,
        "promotion_bytes": 4,
        "eviction_count": 0,
        "eviction_bytes": 0,
        "target_changed": True,
        "residency": {
            "resident_record_ids": [1, 2],
            "resident_bytes": 4,
            "capacity_bytes": 4,
        },
    }
    unchanged = store.apply_target_set([1, 2])
    assert not unchanged["target_changed"]
    assert unchanged["promotion_count"] == 0
    assert unchanged["eviction_count"] == 0

    with pytest.raises(native.NativeStoreError) as duplicate:
        store.apply_target_set([1, 1])
    assert duplicate.value.code == "invalid_target_set"
    assert store.snapshot()["resident_record_ids"] == [1, 2]

    data_path = store_dir / "store.data"
    corrupted = bytearray(data_path.read_bytes())
    corrupted[4] ^= 0xFF
    data_path.write_bytes(corrupted)
    before = store.snapshot()
    with pytest.raises(native.NativeStoreError) as checksum:
        store.apply_target_set([3])
    assert checksum.value.code == "checksum_mismatch"
    assert store.snapshot() == before
    assert store.read(1)["tier"] == "fast"
    stats = store.stats()
    assert stats["target_set_calls"] == 4
    assert stats["successful_target_set_calls"] == 2
    assert stats["failed_target_set_calls"] == 2
    assert stats["failures_by_code"]["invalid_target_set"] == 1
    assert stats["failures_by_code"]["checksum_mismatch"] == 1


def test_private_store_factory_and_operation_conversions(tmp_path: Path) -> None:
    store_dir, _ = _build(tmp_path)
    with pytest.raises(TypeError):
        native.TieredStore()
    with pytest.raises(TypeError, match="non-boolean"):
        native.TieredStore.open(store_dir, True)
    with pytest.raises(OverflowError):
        native.TieredStore.open(store_dir, -1)

    store = native.TieredStore.open(store_dir, 3)
    with pytest.raises(TypeError, match="non-boolean"):
        store.read(False)
    with pytest.raises(TypeError, match="finite sequence"):
        store.apply_target_set(index for index in [1])
    with pytest.raises(native.NativeStoreError) as unknown:
        store.read(99)
    assert unknown.value.code == "unknown_record"
    assert unknown.value.message == "record ID is not present in the store"
    assert unknown.value.record_id == 99
    assert unknown.value.offset is None
    assert unknown.value.path is None
