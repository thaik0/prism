"""Deterministic minimal input bundle for the frozen Phase 1 runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from prism.cloud.contract import (
    CLOUD_SCHEMA_VERSION,
    INPUT_MANIFEST_NAME,
    PHASE1_SPEC_MEMBER,
    REQUIRED_INPUT_MEMBERS,
    CloudContractError,
    canonical_json_bytes,
    input_manifest,
    safe_relative_path,
    sha256_bytes,
)
from prism.container.runner import SPEC_KIND, _load_spec
from prism.experiments.config import ExperimentManifest, load_manifest


@dataclass(frozen=True, slots=True)
class InputBundle:
    members: Mapping[str, bytes]
    manifest_bytes: bytes
    manifest_sha256: str

    @property
    def all_objects(self) -> Mapping[str, bytes]:
        return {**self.members, INPUT_MANIFEST_NAME: self.manifest_bytes}


def build_input_bundle(spec_path: str | Path) -> InputBundle:
    """Build exactly the five files needed by the accepted Phase 1 runner."""

    spec = Path(spec_path).resolve()
    spec_value, _ = _load_spec(spec)
    if spec_value["kind"] != SPEC_KIND or spec_value["experiment_id"] != "baseline__seed_1729":
        raise CloudContractError("only the accepted Phase 1 representative spec is allowed")
    repository_root = spec.parent.parent
    if spec.relative_to(repository_root).as_posix() != PHASE1_SPEC_MEMBER:
        raise CloudContractError(
            f"Phase 1 spec must be located at {PHASE1_SPEC_MEMBER}"
        )
    manifest_path = (repository_root / spec_value["experiment_manifest"]).resolve()
    loaded = load_manifest(manifest_path)
    if not isinstance(loaded, ExperimentManifest):
        raise CloudContractError("Phase 1 cloud input requires the Milestone 5 manifest")
    sources = (
        spec,
        loaded.structure_config,
        loaded.predictor_config,
        loaded.base_workload_config,
        loaded.path,
    )
    members = {
        path.resolve().relative_to(repository_root).as_posix(): path.read_bytes()
        for path in sources
    }
    if tuple(sorted(members)) != REQUIRED_INPUT_MEMBERS:
        raise CloudContractError("Phase 1 inputs no longer match prism-cloud-v1")
    records = [
        {"path": path, "size": len(data), "sha256": sha256_bytes(data)}
        for path, data in sorted(members.items())
    ]
    manifest_bytes = canonical_json_bytes(
        {
            "schema_version": CLOUD_SCHEMA_VERSION,
            "phase1_spec": PHASE1_SPEC_MEMBER,
            "experiment_id": "baseline__seed_1729",
            "members": records,
        }
    )
    input_manifest(manifest_bytes)
    return InputBundle(members, manifest_bytes, sha256_bytes(manifest_bytes))


def verify_input_bundle(manifest_bytes: bytes, members: Mapping[str, bytes]) -> None:
    """Verify an input manifest and every exact required member."""

    manifest = input_manifest(manifest_bytes)
    expected = {item["path"]: item for item in manifest["members"]}
    if set(members) != set(expected):
        raise CloudContractError("downloaded input members do not match the manifest")
    for path, data in members.items():
        safe_relative_path(path, "input member path")
        record = expected[path]
        if len(data) != record["size"] or sha256_bytes(data) != record["sha256"]:
            raise CloudContractError(f"input member hash or size mismatch: {path}")


def materialize_input_bundle(bundle: InputBundle, destination: str | Path) -> Path:
    """Write a verified bundle below a new empty local directory."""

    root = Path(destination).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise CloudContractError("input destination must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    verify_input_bundle(bundle.manifest_bytes, bundle.members)
    for relative, data in bundle.all_objects.items():
        target = root / safe_relative_path(relative, "input member path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def stable_run_id(
    *,
    input_manifest_sha256: str,
    image_uri: str,
    region: str,
    bucket: str,
    job_queue: str,
    job_definition: str,
) -> str:
    """Derive a stable run identifier from deterministic submission inputs."""

    identity = canonical_json_bytes(
        {
            "schema_version": CLOUD_SCHEMA_VERSION,
            "input_manifest_sha256": input_manifest_sha256,
            "image_uri": image_uri,
            "region": region,
            "bucket": bucket,
            "job_queue": job_queue,
            "job_definition": job_definition,
        }
    )
    return "pcv1-" + sha256_bytes(identity)[:24]
