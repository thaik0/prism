"""Versioned, deterministic Prism Cloud input and completion contracts."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


CLOUD_SCHEMA_VERSION = "prism-cloud-v1"
RUNTIME_ARCHITECTURE = "ARM64"
INPUT_MANIFEST_NAME = "input_manifest.json"
COMPLETION_MANIFEST_NAME = "completion_manifest.json"
FAILURE_MANIFEST_NAME = "failure_manifest.json"
SUBMISSION_MANIFEST_NAME = "submission.json"
PHASE1_SPEC_MEMBER = "container/phase1-experiment.json"
REQUIRED_INPUT_MEMBERS = (
    "configs/milestone2_representative.json",
    "configs/milestone3_predictor.json",
    "configs/milestone3_predictor_workload.json",
    "configs/milestone5_experiments.json",
    PHASE1_SPEC_MEMBER,
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"pcv1-[0-9a-f]{24}")
PATH_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
ECR_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class CloudContractError(ValueError):
    """Raised when a Prism Cloud value violates the versioned contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a contract value deterministically."""

    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_sha256(value: bytes) -> str:
    """Return the base64 checksum representation accepted by S3."""

    return base64.b64encode(hashlib.sha256(value).digest()).decode("ascii")


def safe_relative_path(value: object, description: str = "path") -> str:
    """Validate one normalized portable path used in a manifest or S3 key."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise CloudContractError(f"{description} must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise CloudContractError(f"{description} must be a normalized relative path")
    if any(
        part in {"", ".", ".."} or not PATH_COMPONENT_PATTERN.fullmatch(part)
        for part in path.parts
    ):
        raise CloudContractError(f"{description} contains an unsafe component")
    return value


def safe_run_id(value: object) -> str:
    if not isinstance(value, str) or not RUN_ID_PATTERN.fullmatch(value):
        raise CloudContractError("run ID does not match the prism-cloud-v1 format")
    return value


def safe_bucket(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) < 3
        or len(value) > 63
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*[a-z0-9]", value)
        or ".." in value
    ):
        raise CloudContractError("S3 bucket name is invalid")
    return value


def require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise CloudContractError(f"{description} must be a lowercase SHA-256")
    return value


def require_ecr_digest(value: object) -> str:
    if not isinstance(value, str) or not ECR_DIGEST_PATTERN.fullmatch(value):
        raise CloudContractError("ECR image digest must be an immutable SHA-256")
    return value


def input_prefix(run_id: str) -> str:
    return f"prism-cloud/v1/runs/{safe_run_id(run_id)}/input"


def attempt_prefix(run_id: str, batch_job_id: str) -> str:
    safe_run_id(run_id)
    if not isinstance(batch_job_id, str) or not re.fullmatch(
        r"[0-9a-fA-F-]{16,64}", batch_job_id
    ):
        raise CloudContractError("AWS Batch job ID is invalid")
    return f"prism-cloud/v1/runs/{run_id}/attempts/{batch_job_id}"


def submission_key(run_id: str) -> str:
    return f"prism-cloud/v1/runs/{safe_run_id(run_id)}/{SUBMISSION_MANIFEST_NAME}"


def input_manifest(manifest_bytes: bytes) -> Mapping[str, Any]:
    value = _json_object(manifest_bytes, "input manifest")
    if set(value) != {
        "schema_version",
        "phase1_spec",
        "experiment_id",
        "members",
    }:
        raise CloudContractError("input manifest fields do not match prism-cloud-v1")
    if value["schema_version"] != CLOUD_SCHEMA_VERSION:
        raise CloudContractError("input manifest schema version is invalid")
    if value["phase1_spec"] != PHASE1_SPEC_MEMBER:
        raise CloudContractError("input manifest Phase 1 spec is invalid")
    if value["experiment_id"] != "baseline__seed_1729":
        raise CloudContractError("input manifest experiment ID is not accepted")
    members = _artifact_records(value["members"], key_field=False)
    if tuple(item["path"] for item in members) != REQUIRED_INPUT_MEMBERS:
        raise CloudContractError("input manifest does not contain the exact required files")
    return value


def completion_manifest(
    manifest_bytes: bytes,
    *,
    expected_run_id: str | None = None,
    expected_batch_job_id: str | None = None,
) -> Mapping[str, Any]:
    value = _json_object(manifest_bytes, "completion manifest")
    if set(value) != {
        "schema_version",
        "status",
        "scientific_provenance",
        "cloud_execution",
        "artifacts",
    }:
        raise CloudContractError(
            "completion manifest fields do not match prism-cloud-v1"
        )
    if value["schema_version"] != CLOUD_SCHEMA_VERSION or value["status"] != "succeeded":
        raise CloudContractError("completion manifest does not record success")
    scientific = value["scientific_provenance"]
    if not isinstance(scientific, dict) or set(scientific) != {
        "prism_git_revision",
        "prism_package_version",
        "phase1_container_spec_schema_version",
        "phase1_experiment_id",
        "phase1_run_manifest_sha256",
        "input_manifest_sha256",
        "experiment_spec_sha256",
        "source_sha256",
    }:
        raise CloudContractError("scientific provenance is malformed")
    if scientific["phase1_container_spec_schema_version"] != 1:
        raise CloudContractError("Phase 1 container spec version is invalid")
    if scientific["phase1_experiment_id"] != "baseline__seed_1729":
        raise CloudContractError("Phase 1 experiment ID is invalid")
    for field in (
        "phase1_run_manifest_sha256",
        "input_manifest_sha256",
        "experiment_spec_sha256",
    ):
        require_sha256(scientific[field], field)
    if not isinstance(scientific["source_sha256"], dict) or not scientific[
        "source_sha256"
    ]:
        raise CloudContractError("scientific source hashes are missing")
    for name, digest in scientific["source_sha256"].items():
        safe_relative_path(name, "scientific source name")
        require_sha256(digest, f"scientific source hash for {name}")

    execution = value["cloud_execution"]
    if not isinstance(execution, dict) or set(execution) != {
        "run_id",
        "batch_job_id",
        "aws_region",
        "ecr_image_uri",
        "ecr_image_digest",
        "runtime_architecture",
        "attempt_prefix",
    }:
        raise CloudContractError("cloud execution metadata is malformed")
    run_id = safe_run_id(execution["run_id"])
    batch_job_id = execution["batch_job_id"]
    expected_prefix = attempt_prefix(run_id, batch_job_id)
    if execution["attempt_prefix"] != expected_prefix:
        raise CloudContractError("completion attempt prefix is inconsistent")
    if expected_run_id is not None and run_id != expected_run_id:
        raise CloudContractError("completion manifest run ID does not match submission")
    if expected_batch_job_id is not None and batch_job_id != expected_batch_job_id:
        raise CloudContractError("completion manifest Batch job ID does not match")
    if execution["runtime_architecture"] != RUNTIME_ARCHITECTURE:
        raise CloudContractError("completion runtime architecture is not ARM64")
    digest = require_ecr_digest(execution["ecr_image_digest"])
    if not isinstance(execution["ecr_image_uri"], str) or not execution[
        "ecr_image_uri"
    ].endswith("@" + digest):
        raise CloudContractError("completion image URI is not digest-pinned")
    if not isinstance(execution["aws_region"], str) or not re.fullmatch(
        r"[a-z]{2}(?:-gov)?-[a-z]+-\d", execution["aws_region"]
    ):
        raise CloudContractError("completion AWS Region is invalid")

    artifacts = _artifact_records(value["artifacts"], key_field=True)
    if not artifacts or "run_manifest.json" not in {
        item["path"] for item in artifacts
    }:
        raise CloudContractError("completion canonical artifact list is incomplete")
    for item in artifacts:
        expected_key = f"{expected_prefix}/artifacts/{item['path']}"
        if item["key"] != expected_key:
            raise CloudContractError("completion artifact key is inconsistent")
    return value


def failure_manifest(manifest_bytes: bytes) -> Mapping[str, Any]:
    """Validate the non-authoritative diagnostic record for a failed attempt."""

    value = _json_object(manifest_bytes, "failure manifest")
    if set(value) != {
        "schema_version",
        "status",
        "run_id",
        "batch_job_id",
        "stage",
        "error_type",
    }:
        raise CloudContractError("failure manifest fields do not match prism-cloud-v1")
    if value["schema_version"] != CLOUD_SCHEMA_VERSION or value["status"] != "failed":
        raise CloudContractError("failure manifest status is invalid")
    safe_run_id(value["run_id"])
    attempt_prefix(value["run_id"], value["batch_job_id"])
    if value["stage"] not in {"bootstrap", "phase1"}:
        raise CloudContractError("failure manifest stage is invalid")
    if not isinstance(value["error_type"], str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]{0,127}", value["error_type"]
    ):
        raise CloudContractError("failure manifest error type is invalid")
    return value


def _artifact_records(value: object, *, key_field: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise CloudContractError("manifest artifact records must be a list")
    expected = {"path", "size", "sha256"} | ({"key"} if key_field else set())
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != expected:
            raise CloudContractError("manifest artifact record is malformed")
        path = safe_relative_path(item["path"], "artifact path")
        if path in seen:
            raise CloudContractError("manifest artifact paths must be unique")
        seen.add(path)
        if not isinstance(item["size"], int) or isinstance(item["size"], bool) or item[
            "size"
        ] < 0:
            raise CloudContractError("artifact size must be a nonnegative integer")
        require_sha256(item["sha256"], f"artifact hash for {path}")
        if key_field:
            if not isinstance(item["key"], str) or "//" in item["key"]:
                raise CloudContractError("artifact S3 key is malformed")
        records.append(item)
    if [item["path"] for item in records] != sorted(seen):
        raise CloudContractError("manifest artifact records must be path-sorted")
    return records


def _json_object(value: bytes, description: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CloudContractError(f"invalid {description} JSON") from error
    if not isinstance(parsed, dict):
        raise CloudContractError(f"{description} must be a JSON object")
    return parsed


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CloudContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value
