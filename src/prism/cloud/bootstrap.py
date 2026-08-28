"""AWS Batch bootstrap for the accepted Phase 1 Prism runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping, Sequence

from prism.cloud.aws import AwsCloud
from prism.cloud.bundle import verify_input_bundle
from prism.cloud.contract import (
    CLOUD_SCHEMA_VERSION,
    COMPLETION_MANIFEST_NAME,
    FAILURE_MANIFEST_NAME,
    INPUT_MANIFEST_NAME,
    PHASE1_SPEC_MEMBER,
    RUNTIME_ARCHITECTURE,
    CloudContractError,
    attempt_prefix,
    canonical_json_bytes,
    completion_manifest,
    input_manifest,
    input_prefix,
    require_ecr_digest,
    safe_bucket,
    safe_relative_path,
    safe_run_id,
    sha256_bytes,
    sha256_file,
)
from prism.container.runner import run_container_experiment
from prism.container.verify import validate_output_root


class PrismExperimentFailure(RuntimeError):
    """Raised when Phase 1 execution or its local output contract fails."""


def run_bootstrap(
    *,
    cloud: AwsCloud,
    bucket: str,
    run_id: str,
    batch_job_id: str,
    image_uri: str,
    workspace: str | Path | None = None,
) -> Mapping[str, Any]:
    """Download, execute, validate, and publish one immutable Batch attempt."""

    safe_bucket(bucket)
    safe_run_id(run_id)
    prefix = input_prefix(run_id)
    output_prefix = attempt_prefix(run_id, batch_job_id)
    image_base, separator, image_digest = image_uri.rpartition("@")
    if not separator or not image_base:
        raise CloudContractError("bootstrap image URI must be digest-pinned")
    require_ecr_digest(image_digest)

    manifest_bytes = cloud._get_bytes(bucket, f"{prefix}/{INPUT_MANIFEST_NAME}")
    manifest = input_manifest(manifest_bytes)
    members = {
        item["path"]: cloud._get_bytes(bucket, f"{prefix}/{item['path']}")
        for item in manifest["members"]
    }
    verify_input_bundle(manifest_bytes, members)

    if workspace is None:
        with tempfile.TemporaryDirectory(prefix="prism-cloud-") as temporary:
            return _execute_and_publish(
                cloud=cloud,
                bucket=bucket,
                run_id=run_id,
                batch_job_id=batch_job_id,
                image_uri=image_uri,
                image_digest=image_digest,
                output_prefix=output_prefix,
                manifest_bytes=manifest_bytes,
                members=members,
                workspace=Path(temporary),
            )
    return _execute_and_publish(
        cloud=cloud,
        bucket=bucket,
        run_id=run_id,
        batch_job_id=batch_job_id,
        image_uri=image_uri,
        image_digest=image_digest,
        output_prefix=output_prefix,
        manifest_bytes=manifest_bytes,
        members=members,
        workspace=Path(workspace).resolve(),
    )


def _execute_and_publish(
    *,
    cloud: AwsCloud,
    bucket: str,
    run_id: str,
    batch_job_id: str,
    image_uri: str,
    image_digest: str,
    output_prefix: str,
    manifest_bytes: bytes,
    members: Mapping[str, bytes],
    workspace: Path,
) -> Mapping[str, Any]:
    if workspace.exists() and any(workspace.iterdir()):
        raise CloudContractError("bootstrap workspace must be empty")
    workspace.mkdir(parents=True, exist_ok=True)
    input_root = workspace / "input"
    output_root = workspace / "output"
    input_root.mkdir()
    output_root.mkdir()
    for relative, body in sorted(members.items()):
        target = input_root / safe_relative_path(relative, "input member path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    (input_root / INPUT_MANIFEST_NAME).write_bytes(manifest_bytes)

    try:
        run_container_experiment(
            input_root / PHASE1_SPEC_MEMBER,
            output_root,
            input_root=input_root,
            output_root=workspace,
        )
        phase1 = validate_output_root(output_root)
    except Exception as error:
        raise PrismExperimentFailure("accepted Phase 1 experiment failed") from error
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        safe_relative_path(relative, "canonical artifact path")
        body = path.read_bytes()
        key = f"{output_prefix}/artifacts/{relative}"
        cloud._put_bytes(bucket, key, body)
        artifacts.append(
            {
                "path": relative,
                "key": key,
                "size": len(body),
                "sha256": sha256_bytes(body),
            }
        )

    phase1_manifest_path = output_root / "run_manifest.json"
    completion = {
        "schema_version": CLOUD_SCHEMA_VERSION,
        "status": "succeeded",
        "scientific_provenance": {
            "prism_git_revision": phase1["prism"]["git_revision"],
            "prism_package_version": phase1["prism"]["package_version"],
            "phase1_container_spec_schema_version": 1,
            "phase1_experiment_id": phase1["experiment"]["experiment_id"],
            "phase1_run_manifest_sha256": sha256_file(phase1_manifest_path),
            "input_manifest_sha256": sha256_bytes(manifest_bytes),
            "experiment_spec_sha256": phase1["experiment"]["spec_sha256"],
            "source_sha256": phase1["experiment"]["source_sha256"],
        },
        "cloud_execution": {
            "run_id": run_id,
            "batch_job_id": batch_job_id,
            "aws_region": cloud.region,
            "ecr_image_uri": image_uri,
            "ecr_image_digest": image_digest,
            "runtime_architecture": RUNTIME_ARCHITECTURE,
            "attempt_prefix": output_prefix,
        },
        "artifacts": artifacts,
    }
    completion_bytes = canonical_json_bytes(completion)
    completion_manifest(
        completion_bytes,
        expected_run_id=run_id,
        expected_batch_job_id=batch_job_id,
    )
    # This is deliberately the final S3 write. It is the commit marker, not an
    # object rename or transaction abstraction.
    cloud._put_bytes(
        bucket,
        f"{output_prefix}/{COMPLETION_MANIFEST_NAME}",
        completion_bytes,
    )
    return completion


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise CloudContractError(f"required bootstrap environment is missing: {name}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        print("prism-cloud-bootstrap: command arguments are not allowed", file=sys.stderr)
        return 2
    cloud: AwsCloud | None = None
    bucket = run_id = batch_job_id = ""
    try:
        region = _required_environment("PRISM_CLOUD_REGION")
        bucket = _required_environment("PRISM_CLOUD_BUCKET")
        run_id = _required_environment("PRISM_CLOUD_RUN_ID")
        batch_job_id = _required_environment("AWS_BATCH_JOB_ID")
        image_uri = _required_environment("PRISM_CLOUD_IMAGE_URI")
        machine = platform.machine().lower()
        if machine not in {"arm64", "aarch64"}:
            raise CloudContractError("bootstrap runtime is not Linux/ARM64")
        cloud = AwsCloud.from_boto3(region)
        completion = run_bootstrap(
            cloud=cloud,
            bucket=bucket,
            run_id=run_id,
            batch_job_id=batch_job_id,
            image_uri=image_uri,
        )
    except Exception as error:
        if cloud is not None and bucket and run_id and batch_job_id:
            try:
                prefix = attempt_prefix(run_id, batch_job_id)
                failure = canonical_json_bytes(
                    {
                        "schema_version": CLOUD_SCHEMA_VERSION,
                        "status": "failed",
                        "run_id": run_id,
                        "batch_job_id": batch_job_id,
                        "stage": (
                            "phase1"
                            if isinstance(error, PrismExperimentFailure)
                            else "bootstrap"
                        ),
                        "error_type": type(error).__name__,
                    }
                )
                cloud._put_bytes(
                    bucket, f"{prefix}/{FAILURE_MANIFEST_NAME}", failure
                )
            except Exception as publication_error:
                print(
                    f"prism-cloud-bootstrap: failure record upload failed: {publication_error}",
                    file=sys.stderr,
                )
        print(f"prism-cloud-bootstrap: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_count": len(completion["artifacts"]),
                "batch_job_id": completion["cloud_execution"]["batch_job_id"],
                "runtime_architecture": platform.machine(),
                "status": "completed",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
