from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

import pytest

from prism.native import (
    NativeStoreError,
    TieredStore,
    build_store,
    build_verified_native_store,
    generate_payloads,
    generate_record_payload,
)
from prism.workload import generate_workload, persist_workload


def test_payload_schema_is_exact_deterministic_and_validated() -> None:
    payload = generate_record_payload(workload_seed=1729, record_id=7, byte_size=40)

    assert len(payload) == 40
    assert payload.hex() == (
        "73470e5305e968034d0b00ced81b83691d74ca1c4881bbcc11abec31fe343340"
        "65c5c4923aff305e"
    )
    assert payload == generate_record_payload(
        workload_seed=1729, record_id=7, byte_size=40
    )
    assert payload != generate_record_payload(
        workload_seed=1729, record_id=8, byte_size=40
    )
    with pytest.raises(TypeError, match="non-boolean"):
        generate_record_payload(workload_seed=True, record_id=0, byte_size=1)
    with pytest.raises(ValueError, match="uint64"):
        generate_record_payload(workload_seed=-1, record_id=0, byte_size=1)
    with pytest.raises(ValueError, match="uint64"):
        generate_record_payload(workload_seed=0, record_id=2**64, byte_size=1)
    with pytest.raises(ValueError, match="positive"):
        generate_record_payload(workload_seed=0, record_id=0, byte_size=0)


def test_payload_collection_uses_ascending_ids_and_exact_sizes() -> None:
    records = generate_payloads(workload_seed=5, record_sizes=[1, 32, 33])

    assert [record.record_id for record in records] == [0, 1, 2]
    assert [record.byte_size for record in records] == [1, 32, 33]
    assert [len(record.payload) for record in records] == [1, 32, 33]
    assert all(len(record.sha256) == 64 for record in records)
    with pytest.raises(FrozenInstanceError):
        records[0].record_id = 99


def test_public_wrapper_returns_immutable_values_and_operation_errors(
    tmp_path: Path,
) -> None:
    summary = build_store([(2, b"bb"), (1, b"a")], tmp_path / "store")
    store = TieredStore.open(tmp_path / "store", 2)

    assert summary.record_count == 2
    assert summary.data_file_bytes == 3
    assert len(summary.store_data_sha256) == 64
    assert len(summary.store_index_sha256) == 64
    slow = store.read(1)
    assert (slow.payload, slow.tier, slow.byte_count) == (b"a", "slow", 1)
    assert store.promote(1).moved
    assert store.read(1).tier == "fast"
    assert store.apply_target_set([2]).incoming_record_ids == (2,)
    assert store.snapshot().resident_record_ids == (2,)
    assert store.record_metadata(2).byte_length == 2
    stats = store.stats()
    assert stats.committed_promotions == 2
    assert stats.committed_evictions == 1
    with pytest.raises(TypeError):
        stats.failures_by_code["unknown_record"] = 1
    with pytest.raises(FrozenInstanceError):
        slow.tier = "changed"

    before = store.snapshot()
    with pytest.raises(NativeStoreError) as caught:
        store.promote(1)
    error = caught.value
    assert error.code == "insufficient_capacity"
    assert error.operation == "promote"
    assert error.record_id == 1
    assert error.offset is None
    assert error.path is None
    assert str(error) == (
        "[insufficient_capacity] promote: remaining fast-tier capacity is "
        "insufficient (record_id=1)"
    )
    assert store.snapshot() == before
    assert store.read(2).tier == "fast"


def test_verified_native_store_manifest_and_files_are_byte_deterministic(
    tmp_path: Path, make_config
) -> None:
    source = tmp_path / "source"
    persist_workload(generate_workload(make_config(seed=73)), source)
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    output_a.mkdir()
    output_b.mkdir()

    first = build_verified_native_store(
        source,
        output_a / "native_store",
        400,
        manifest_path=output_a / "native_store_manifest.json",
    )
    second = build_verified_native_store(
        source,
        output_b / "native_store",
        400,
        manifest_path=output_b / "native_store_manifest.json",
    )

    assert (output_a / "native_store/store.data").read_bytes() == (
        output_b / "native_store/store.data"
    ).read_bytes()
    assert (output_a / "native_store/store.index").read_bytes() == (
        output_b / "native_store/store.index"
    ).read_bytes()
    assert (output_a / "native_store_manifest.json").read_bytes() == (
        output_b / "native_store_manifest.json"
    ).read_bytes()
    assert first == second
    manifest = json.loads((output_a / "native_store_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["payload_schema_version"] == 1
    assert manifest["workload_seed"] == 73
    assert manifest["record_count"] == 8
    assert manifest["total_payload_bytes"] == sum(
        record.byte_size for record in first.payloads
    )
    assert [record["record_id"] for record in manifest["records"]] == list(range(8))
    assert manifest["payload_verification_passed"]
    assert set(manifest["source_workload_hashes"]) == {
        "config.json",
        "hidden_ground_truth.json",
        "observable_events.jsonl",
        "summary.json",
    }
    assert str(tmp_path) not in json.dumps(manifest)
    assert all("payload" not in record for record in manifest["records"])
    fresh = TieredStore.open(output_a / "native_store", 400)
    assert fresh.stats().successful_slow_reads == 0
    for expected in first.payloads:
        assert fresh.record_metadata(expected.record_id).byte_length == expected.byte_size


def test_source_only_imports_survive_missing_extension_with_actionable_native_error() -> None:
    pure = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['prism._native']=None; "
            "import prism; import prism.simulation; print('pure imports ok')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    missing = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['prism._native']=None; import prism.native",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert pure.returncode == 0, pure.stderr
    assert pure.stdout.strip() == "pure imports ok"
    assert missing.returncode != 0
    assert "Prism's native extension is not installed" in missing.stderr
    assert "python3 -m pip install -e ." in missing.stderr
