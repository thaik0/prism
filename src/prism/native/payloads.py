"""Deterministic payload fixtures and verified native-store manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import struct
from typing import Any, Sequence
import zlib

from prism.native.store import BuildSummary, TieredStore, build_store
from prism.workload import WorkloadConfig
from prism.workload.validate import SOURCE_ARTIFACT_FILENAMES


PAYLOAD_SCHEMA_VERSION = 1
PAYLOAD_DOMAIN_SEPARATOR = b"PRISM_RECORD_PAYLOAD"
NATIVE_STORE_MANIFEST_SCHEMA_VERSION = 1
UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class GeneratedPayload:
    record_id: int
    byte_size: int
    payload: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class NativeManifestRecord:
    record_id: int
    byte_size: int
    byte_offset: int
    crc32: int
    sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "record_id": self.record_id,
            "byte_size": self.byte_size,
            "byte_offset": self.byte_offset,
            "crc32": self.crc32,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class NativeStoreManifest:
    payload_schema_version: int
    source_workload_hashes: tuple[tuple[str, str], ...]
    workload_seed: int
    record_count: int
    total_payload_bytes: int
    records: tuple[NativeManifestRecord, ...]
    native_format_version: int
    capacity_bytes: int
    store_data_sha256: str
    store_index_sha256: str
    dependency_versions: tuple[tuple[str, str], ...]
    payload_verification_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_STORE_MANIFEST_SCHEMA_VERSION,
            "payload_schema_version": self.payload_schema_version,
            "source_workload_hashes": dict(self.source_workload_hashes),
            "workload_seed": self.workload_seed,
            "record_count": self.record_count,
            "total_payload_bytes": self.total_payload_bytes,
            "records": [record.to_dict() for record in self.records],
            "native_store": {
                "format_version": self.native_format_version,
                "capacity_bytes": self.capacity_bytes,
                "store_data_sha256": self.store_data_sha256,
                "store_index_sha256": self.store_index_sha256,
            },
            "dependency_versions": dict(self.dependency_versions),
            "payload_verification_passed": self.payload_verification_passed,
        }


@dataclass(frozen=True, slots=True)
class NativeStoreArtifacts:
    build_summary: BuildSummary
    payloads: tuple[GeneratedPayload, ...]
    manifest: NativeStoreManifest


def _uint64(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be in uint64 range")
    return value


def generate_record_payload(
    *, workload_seed: int, record_id: int, byte_size: int
) -> bytes:
    """Generate exact bytes from a fixed domain and little-endian integers."""

    seed = _uint64("workload_seed", workload_seed)
    identifier = _uint64("record_id", record_id)
    if isinstance(byte_size, bool) or not isinstance(byte_size, int):
        raise TypeError("byte_size must be a non-boolean integer")
    if not 0 < byte_size <= UINT64_MAX:
        raise ValueError("byte_size must be positive and in uint64 range")
    output = bytearray()
    counter = 0
    while len(output) < byte_size:
        encoded = PAYLOAD_DOMAIN_SEPARATOR + struct.pack(
            "<IQQQ", PAYLOAD_SCHEMA_VERSION, seed, identifier, counter
        )
        output.extend(hashlib.sha256(encoded).digest())
        counter += 1
    return bytes(output[:byte_size])


def generate_payloads(
    *, workload_seed: int, record_sizes: Sequence[int]
) -> tuple[GeneratedPayload, ...]:
    payloads = []
    for record_id, byte_size in enumerate(record_sizes):
        payload = generate_record_payload(
            workload_seed=workload_seed,
            record_id=record_id,
            byte_size=byte_size,
        )
        payloads.append(
            GeneratedPayload(
                record_id=record_id,
                byte_size=byte_size,
                payload=payload,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return tuple(payloads)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_workload_payloads(
    run_dir: str | Path,
) -> tuple[WorkloadConfig, tuple[GeneratedPayload, ...], tuple[tuple[str, str], ...]]:
    source = Path(run_dir)
    for filename in SOURCE_ARTIFACT_FILENAMES:
        if not (source / filename).is_file():
            raise ValueError(f"missing required workload artifact: {filename}")
    config = WorkloadConfig.from_json(source / "config.json")
    hidden = json.loads((source / "hidden_ground_truth.json").read_text(encoding="utf-8"))
    sizes = hidden.get("record_sizes_bytes") if isinstance(hidden, dict) else None
    if (
        not isinstance(sizes, list)
        or len(sizes) != config.num_records
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes)
    ):
        raise ValueError("workload record_sizes_bytes are missing or invalid")
    payloads = generate_payloads(workload_seed=config.seed, record_sizes=sizes)
    hashes = tuple(
        (filename, _sha256(source / filename))
        for filename in sorted(SOURCE_ARTIFACT_FILENAMES)
    )
    return config, payloads, hashes


def build_verified_native_store(
    run_dir: str | Path,
    store_dir: str | Path,
    fast_capacity_bytes: int,
    *,
    manifest_path: str | Path | None = None,
) -> NativeStoreArtifacts:
    """Create through C++, audit metadata, and verify every exact payload."""

    config, payloads, source_hashes = load_workload_payloads(run_dir)
    return _build_verified_payload_store(
        payloads,
        source_hashes,
        config.seed,
        store_dir,
        fast_capacity_bytes,
        manifest_path=manifest_path,
    )


def _build_verified_payload_store(
    payloads: Sequence[GeneratedPayload],
    source_hashes: Sequence[tuple[str, str]],
    workload_seed: int,
    store_dir: str | Path,
    fast_capacity_bytes: int,
    *,
    manifest_path: str | Path | None = None,
) -> NativeStoreArtifacts:
    """Shared verified construction for workload and hand-checkable fixtures."""

    resolved_payloads = tuple(payloads)
    summary = build_store(
        [(record.record_id, record.payload) for record in resolved_payloads], store_dir
    )
    store = TieredStore.open(store_dir, fast_capacity_bytes)
    records = []
    for expected in resolved_payloads:
        metadata = store.record_metadata(expected.record_id)
        if metadata.byte_length != expected.byte_size:
            raise ValueError(
                f"native metadata length mismatch for record {expected.record_id}"
            )
        read = store.read(expected.record_id)
        if read.tier != "slow" or read.byte_count != expected.byte_size:
            raise ValueError(f"native setup read mismatch for record {expected.record_id}")
        if hashlib.sha256(read.payload).hexdigest() != expected.sha256:
            raise ValueError(f"native payload mismatch for record {expected.record_id}")
        records.append(
            NativeManifestRecord(
                record_id=expected.record_id,
                byte_size=expected.byte_size,
                byte_offset=metadata.byte_offset,
                crc32=metadata.crc32,
                sha256=expected.sha256,
            )
        )
    manifest = NativeStoreManifest(
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
        source_workload_hashes=tuple(sorted(source_hashes)),
        workload_seed=workload_seed,
        record_count=len(resolved_payloads),
        total_payload_bytes=sum(record.byte_size for record in resolved_payloads),
        records=tuple(records),
        native_format_version=summary.format_version,
        capacity_bytes=fast_capacity_bytes,
        store_data_sha256=summary.store_data_sha256,
        store_index_sha256=summary.store_index_sha256,
        dependency_versions=(
            ("flatbuffers", "24.3.25"),
            ("pybind11", "3.0.4"),
            ("python", platform.python_version()),
            ("scikit-build-core", "0.11.6"),
            ("zlib", zlib.ZLIB_VERSION),
        ),
        payload_verification_passed=True,
    )
    if manifest_path is not None:
        write_native_store_manifest(manifest, manifest_path)
    return NativeStoreArtifacts(summary, resolved_payloads, manifest)


def write_native_store_manifest(
    manifest: NativeStoreManifest, path: str | Path
) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(manifest.to_dict(), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
