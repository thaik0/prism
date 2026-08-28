"""Small boto3 adapter for Prism-owned AWS interactions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

from prism.cloud.bundle import InputBundle
from prism.cloud.contract import (
    CLOUD_SCHEMA_VERSION,
    COMPLETION_MANIFEST_NAME,
    FAILURE_MANIFEST_NAME,
    INPUT_MANIFEST_NAME,
    RUNTIME_ARCHITECTURE,
    CloudContractError,
    canonical_json_bytes,
    checksum_sha256,
    completion_manifest,
    failure_manifest,
    input_prefix,
    require_ecr_digest,
    safe_bucket,
    safe_relative_path,
    safe_run_id,
    sha256_bytes,
    submission_key,
)


BATCH_STATES = {
    "SUBMITTED",
    "PENDING",
    "RUNNABLE",
    "STARTING",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
}
ECR_URI_PATTERN = re.compile(
    r"(?P<registry>[0-9]{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com/"
    r"(?P<repository>[a-z0-9]+(?:[._/-][a-z0-9]+)*)(?:(?P<tag>:[A-Za-z0-9._-]+)|@(?P<digest>sha256:[0-9a-f]{64}))"
)
SUPPORTED_FARGATE_MEMORY = {
    "0.25": {"512", "1024", "2048"},
    "0.5": {"1024", "2048", "3072", "4096"},
    "1": {str(value) for value in range(2048, 8193, 1024)},
    "2": {str(value) for value in range(4096, 16385, 1024)},
    "4": {str(value) for value in range(8192, 30721, 1024)},
    "8": {str(value) for value in range(16384, 61441, 4096)},
    "16": {str(value) for value in range(32768, 122881, 8192)},
}


@dataclass(frozen=True, slots=True)
class Submission:
    run_id: str
    batch_job_id: str
    region: str
    bucket: str
    job_queue: str
    job_definition: str
    image_uri: str
    image_digest: str
    input_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ResultInspection:
    state: str
    manifest: Mapping[str, Any] | None


class AwsCloud:
    """AWS operations used by the local CLI and cloud bootstrap."""

    def __init__(
        self,
        *,
        region: str,
        s3: Any,
        batch: Any | None = None,
        ecr: Any | None = None,
        logs: Any | None = None,
    ) -> None:
        if not isinstance(region, str) or not region:
            raise CloudContractError("an explicit AWS Region is required")
        self.region = region
        self.s3 = s3
        self.batch = batch
        self.ecr = ecr
        self.logs = logs

    @classmethod
    def from_boto3(cls, region: str) -> AwsCloud:
        """Create clients through the standard boto3 provider chain."""

        try:
            import boto3
        except ImportError as error:  # pragma: no cover - installation boundary
            raise CloudContractError("boto3 is required for Prism Cloud") from error
        session = boto3.Session(region_name=region)
        resolved_region = session.region_name
        if not resolved_region:
            raise CloudContractError("an AWS Region is required")
        return cls(
            region=resolved_region,
            s3=session.client("s3"),
            batch=session.client("batch"),
            ecr=session.client("ecr"),
            logs=session.client("logs"),
        )

    def resolve_ecr_image(self, image: str) -> tuple[str, str]:
        """Resolve a tag or verify a digest and return an immutable ECR URI."""

        if self.ecr is None:
            raise CloudContractError("ECR client is unavailable")
        match = ECR_URI_PATTERN.fullmatch(image)
        if match is None or match.group("region") != self.region:
            raise CloudContractError("image must be a private ECR URI in the selected Region")
        registry = match.group("registry")
        repository = match.group("repository")
        tag = match.group("tag")
        digest = match.group("digest")
        if tag == ":latest":
            raise CloudContractError("latest is forbidden as experiment provenance")
        image_id = {"imageDigest": digest} if digest else {"imageTag": tag[1:]}
        response = self.ecr.describe_images(
            registryId=registry,
            repositoryName=repository,
            imageIds=[image_id],
        )
        details = response.get("imageDetails", [])
        if len(details) != 1:
            raise CloudContractError("ECR image reference did not resolve uniquely")
        resolved_digest = require_ecr_digest(details[0].get("imageDigest"))
        if digest is not None and resolved_digest != digest:
            raise CloudContractError("ECR returned a different image digest")
        repository_uri = image.split("@", 1)[0].rsplit(":", 1)[0]
        return f"{repository_uri}@{resolved_digest}", resolved_digest

    def validate_job_definition(self, name: str, image_uri: str) -> str:
        """Validate and return the exact active ARM64/Fargate job definition ARN."""

        if self.batch is None:
            raise CloudContractError("Batch client is unavailable")
        response = self.batch.describe_job_definitions(
            jobDefinitionName=name, status="ACTIVE"
        )
        definitions = response.get("jobDefinitions", [])
        if not definitions:
            raise CloudContractError("active AWS Batch job definition was not found")
        definition = max(definitions, key=lambda item: int(item.get("revision", 0)))
        _validate_job_definition_value(definition, image_uri)
        arn = definition.get("jobDefinitionArn")
        if not isinstance(arn, str) or not arn:
            raise CloudContractError("Batch job definition ARN is missing")
        return arn

    def upload_input_bundle(
        self, *, bucket: str, run_id: str, bundle: InputBundle
    ) -> None:
        safe_bucket(bucket)
        prefix = input_prefix(run_id)
        for relative, body in sorted(bundle.all_objects.items()):
            self._put_bytes(bucket, f"{prefix}/{relative}", body)

    def submit(
        self,
        *,
        run_id: str,
        bucket: str,
        job_queue: str,
        job_definition: str,
        image_uri: str,
        image_digest: str,
        input_manifest_sha256: str,
    ) -> Submission:
        if self.batch is None:
            raise CloudContractError("Batch client is unavailable")
        request = build_submit_request(
            run_id=run_id,
            region=self.region,
            bucket=bucket,
            job_queue=job_queue,
            job_definition=job_definition,
            image_uri=image_uri,
        )
        response = self.batch.submit_job(**request)
        job_id = response.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise CloudContractError("AWS Batch did not return a job ID")
        submission = Submission(
            run_id,
            job_id,
            self.region,
            bucket,
            job_queue,
            job_definition,
            image_uri,
            image_digest,
            input_manifest_sha256,
        )
        self._put_bytes(bucket, submission_key(run_id), _submission_bytes(submission))
        return submission

    def read_submission(self, bucket: str, run_id: str) -> Submission:
        value = _submission_value(self._get_bytes(safe_bucket(bucket), submission_key(run_id)))
        if value.run_id != safe_run_id(run_id) or value.bucket != bucket:
            raise CloudContractError("submission metadata does not match the requested run")
        return value

    def describe_job(self, job_id: str) -> Mapping[str, Any]:
        if self.batch is None:
            raise CloudContractError("Batch client is unavailable")
        response = self.batch.describe_jobs(jobs=[job_id])
        jobs = response.get("jobs", [])
        if len(jobs) != 1:
            raise CloudContractError("AWS Batch job was not found")
        status = jobs[0].get("status")
        if status not in BATCH_STATES:
            raise CloudContractError("AWS Batch returned an unknown state")
        return jobs[0]

    def inspect_result(self, submission: Submission) -> ResultInspection:
        """Validate completion and every remote canonical artifact hash."""

        job = self.describe_job(submission.batch_job_id)
        if job["status"] == "FAILED":
            prefix = f"prism-cloud/v1/runs/{submission.run_id}/attempts/{submission.batch_job_id}"
            try:
                failure = failure_manifest(
                    self._get_bytes(
                        submission.bucket, f"{prefix}/{FAILURE_MANIFEST_NAME}"
                    )
                )
            except Exception as error:
                if not _not_found(error):
                    return ResultInspection("Batch job failed", None)
            else:
                if (
                    failure["run_id"] == submission.run_id
                    and failure["batch_job_id"] == submission.batch_job_id
                    and failure["stage"] == "phase1"
                ):
                    return ResultInspection("Prism experiment failed", None)
            return ResultInspection("Batch job failed", None)
        if job["status"] != "SUCCEEDED":
            return ResultInspection("not complete", None)
        prefix = f"prism-cloud/v1/runs/{submission.run_id}/attempts/{submission.batch_job_id}"
        try:
            raw = self._get_bytes(
                submission.bucket, f"{prefix}/{COMPLETION_MANIFEST_NAME}"
            )
        except Exception as error:
            if _not_found(error):
                return ResultInspection(
                    "incomplete — completion manifest missing", None
                )
            raise
        try:
            manifest = completion_manifest(
                raw,
                expected_run_id=submission.run_id,
                expected_batch_job_id=submission.batch_job_id,
            )
        except CloudContractError:
            return ResultInspection("completion manifest invalid", None)
        bodies: dict[str, bytes] = {}
        for artifact in manifest["artifacts"]:
            try:
                body = self._get_bytes(submission.bucket, artifact["key"])
            except Exception as error:
                if _not_found(error):
                    return ResultInspection("artifact hash mismatch", manifest)
                raise
            if len(body) != artifact["size"] or sha256_bytes(body) != artifact["sha256"]:
                return ResultInspection("artifact hash mismatch", manifest)
            bodies[artifact["path"]] = body
        if not _phase1_artifact_contract_matches(manifest, bodies):
            return ResultInspection("completion manifest invalid", None)
        return ResultInspection("verified", manifest)

    def wait_for_result(
        self,
        submission: Submission,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> tuple[Mapping[str, Any], ResultInspection]:
        """Wait for native Batch completion and then inspect Prism validity."""

        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise CloudContractError("wait timeout and poll interval must be positive")
        deadline = clock() + timeout_seconds
        while True:
            job = self.describe_job(submission.batch_job_id)
            if job["status"] in {"SUCCEEDED", "FAILED"}:
                return job, self.inspect_result(submission)
            remaining = deadline - clock()
            if remaining <= 0:
                raise CloudContractError(
                    f"timed out waiting for AWS Batch job {submission.batch_job_id}"
                )
            sleeper(min(poll_interval_seconds, remaining))

    def recent_logs(self, job_id: str, *, limit: int = 200) -> list[str]:
        if self.logs is None:
            raise CloudContractError("CloudWatch Logs client is unavailable")
        job = self.describe_job(job_id)
        stream = job.get("container", {}).get("logStreamName")
        if not isinstance(stream, str) or not stream:
            raise CloudContractError("Batch job has no CloudWatch log stream yet")
        response = self.logs.get_log_events(
            logGroupName="/aws/batch/job",
            logStreamName=stream,
            startFromHead=False,
            limit=limit,
        )
        return [
            str(event.get("message", "")) for event in response.get("events", [])
        ]

    def download_verified(self, submission: Submission, destination: str | Path) -> Path:
        inspection = self.inspect_result(submission)
        if inspection.state != "verified" or inspection.manifest is None:
            raise CloudContractError(
                f"Prism result is not downloadable: {inspection.state}"
            )
        root = Path(destination).resolve()
        if root.exists() and not root.is_dir():
            raise CloudContractError("download destination exists and is not a directory")
        root.mkdir(parents=True, exist_ok=True)
        expected_paths = {item["path"] for item in inspection.manifest["artifacts"]}
        existing = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if existing - expected_paths:
            raise CloudContractError("download destination contains unexpected files")
        for item in inspection.manifest["artifacts"]:
            relative = safe_relative_path(item["path"], "artifact path")
            target = root / relative
            if target.is_symlink():
                raise CloudContractError("download destination contains a symlink")
            body = self._get_bytes(submission.bucket, item["key"])
            if len(body) != item["size"] or sha256_bytes(body) != item["sha256"]:
                raise CloudContractError(f"artifact hash mismatch during download: {relative}")
            if target.exists():
                if not target.is_file() or target.read_bytes() != body:
                    raise CloudContractError(
                        f"refusing to overwrite non-identical artifact: {relative}"
                    )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".prism-download")
            if temporary.exists():
                raise CloudContractError("stale temporary download file exists")
            temporary.write_bytes(body)
            temporary.replace(target)
        return root

    def _put_bytes(self, bucket: str, key: str, body: bytes) -> None:
        self.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ChecksumSHA256=checksum_sha256(body),
            Metadata={"sha256": sha256_bytes(body)},
        )

    def _get_bytes(self, bucket: str, key: str) -> bytes:
        response = self.s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()


def build_submit_request(
    *,
    run_id: str,
    region: str,
    bucket: str,
    job_queue: str,
    job_definition: str,
    image_uri: str,
) -> dict[str, Any]:
    """Construct the only allowlisted per-submission Batch override."""

    safe_run_id(run_id)
    safe_bucket(bucket)
    if not all(
        isinstance(value, str) and value
        for value in (region, job_queue, job_definition, image_uri)
    ):
        raise CloudContractError("Batch submission configuration is incomplete")
    if "@sha256:" not in image_uri:
        raise CloudContractError("Batch submission image must be digest-pinned")
    environment = [
        {"name": "PRISM_CLOUD_BUCKET", "value": bucket},
        {"name": "PRISM_CLOUD_IMAGE_URI", "value": image_uri},
        {"name": "PRISM_CLOUD_REGION", "value": region},
        {"name": "PRISM_CLOUD_RUN_ID", "value": run_id},
    ]
    return {
        "jobName": run_id,
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "containerOverrides": {"environment": environment},
        "retryStrategy": {"attempts": 1},
        "tags": {
            "PrismContract": CLOUD_SCHEMA_VERSION,
            "PrismRunId": run_id,
        },
    }


def _validate_job_definition_value(value: Mapping[str, Any], image_uri: str) -> None:
    if value.get("platformCapabilities") != ["FARGATE"]:
        raise CloudContractError("Batch job definition must be FARGATE-only")
    if value.get("retryStrategy", {}).get("attempts") != 1:
        raise CloudContractError("Batch job definition retry attempts must equal 1")
    container = value.get("containerProperties")
    if not isinstance(container, dict):
        raise CloudContractError("Batch container properties are missing")
    if container.get("image") != image_uri:
        raise CloudContractError("Batch job definition image is not the resolved digest")
    runtime = container.get("runtimePlatform", {})
    if runtime != {
        "operatingSystemFamily": "LINUX",
        "cpuArchitecture": RUNTIME_ARCHITECTURE,
    }:
        raise CloudContractError("Batch runtime must be Linux/ARM64")
    if container.get("command") != ["prism-cloud-bootstrap"]:
        raise CloudContractError("Batch command must be the fixed Prism bootstrap")
    if container.get("networkConfiguration", {}).get("assignPublicIp") != "ENABLED":
        raise CloudContractError("Batch Fargate task must assign a public IP")
    log_configuration = container.get("logConfiguration", {})
    if log_configuration.get("logDriver") != "awslogs":
        raise CloudContractError("Batch job definition must use awslogs")
    execution_role = container.get("executionRoleArn")
    job_role = container.get("jobRoleArn")
    if not execution_role or not job_role or execution_role == job_role:
        raise CloudContractError("Batch execution and Prism job roles must be separate")
    resources = {
        item.get("type"): str(item.get("value"))
        for item in container.get("resourceRequirements", [])
        if isinstance(item, dict)
    }
    vcpu = resources.get("VCPU")
    memory = resources.get("MEMORY")
    if vcpu not in SUPPORTED_FARGATE_MEMORY or memory not in SUPPORTED_FARGATE_MEMORY[vcpu]:
        raise CloudContractError("Batch vCPU/memory values are not a supported Fargate pair")


def _submission_bytes(value: Submission) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": CLOUD_SCHEMA_VERSION,
            "run_id": value.run_id,
            "batch_job_id": value.batch_job_id,
            "aws_region": value.region,
            "bucket": value.bucket,
            "job_queue": value.job_queue,
            "job_definition": value.job_definition,
            "ecr_image_uri": value.image_uri,
            "ecr_image_digest": value.image_digest,
            "input_manifest_sha256": value.input_manifest_sha256,
        }
    )


def _submission_value(raw: bytes) -> Submission:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CloudContractError("submission metadata is invalid JSON") from error
    expected = {
        "schema_version",
        "run_id",
        "batch_job_id",
        "aws_region",
        "bucket",
        "job_queue",
        "job_definition",
        "ecr_image_uri",
        "ecr_image_digest",
        "input_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise CloudContractError("submission metadata fields are invalid")
    if value["schema_version"] != CLOUD_SCHEMA_VERSION:
        raise CloudContractError("submission schema version is invalid")
    safe_run_id(value["run_id"])
    safe_bucket(value["bucket"])
    require_ecr_digest(value["ecr_image_digest"])
    if value["ecr_image_uri"].split("@")[-1] != value["ecr_image_digest"]:
        raise CloudContractError("submission ECR provenance is inconsistent")
    if not re.fullmatch(r"[0-9a-f]{64}", value["input_manifest_sha256"]):
        raise CloudContractError("submission input manifest hash is invalid")
    return Submission(
        value["run_id"],
        value["batch_job_id"],
        value["aws_region"],
        value["bucket"],
        value["job_queue"],
        value["job_definition"],
        value["ecr_image_uri"],
        value["ecr_image_digest"],
        value["input_manifest_sha256"],
    )


def _not_found(error: Exception) -> bool:
    response = getattr(error, "response", {})
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    return code in {"NoSuchKey", "404", "NotFound"}


def _phase1_artifact_contract_matches(
    completion: Mapping[str, Any], bodies: Mapping[str, bytes]
) -> bool:
    """Tie the cloud artifact list back to the canonical Phase 1 manifest."""

    raw = bodies.get("run_manifest.json")
    if raw is None:
        return False
    scientific = completion["scientific_provenance"]
    if sha256_bytes(raw) != scientific["phase1_run_manifest_sha256"]:
        return False
    try:
        phase1 = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(phase1, dict)
        or phase1.get("schema_version") != 1
        or not isinstance(phase1.get("artifacts"), dict)
        or phase1.get("prism", {}).get("git_revision")
        != scientific["prism_git_revision"]
        or phase1.get("prism", {}).get("package_version")
        != scientific["prism_package_version"]
        or phase1.get("experiment", {}).get("experiment_id")
        != scientific["phase1_experiment_id"]
        or phase1.get("experiment", {}).get("spec_sha256")
        != scientific["experiment_spec_sha256"]
        or phase1.get("experiment", {}).get("source_sha256")
        != scientific["source_sha256"]
    ):
        return False
    phase1_artifacts = phase1["artifacts"]
    expected_paths = set(phase1_artifacts) | {"run_manifest.json"}
    if set(bodies) != expected_paths:
        return False
    completion_hashes = {
        item["path"]: item["sha256"] for item in completion["artifacts"]
    }
    return all(
        isinstance(digest, str) and completion_hashes.get(path) == digest
        for path, digest in phase1_artifacts.items()
    )
