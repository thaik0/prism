from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import pytest

from prism.cloud.aws import AwsCloud, Submission
from prism.cloud.bootstrap import main, run_bootstrap
from prism.cloud.bundle import build_input_bundle
from prism.cloud.contract import CloudContractError


ROOT = Path(__file__).resolve().parents[1]
BUCKET = "prism-example"
RUN_ID = "pcv1-" + "a" * 24
JOB_ID = "11111111-1111-1111-1111-111111111111"
DIGEST = "sha256:" + "b" * 64
IMAGE = f"123456789012.dkr.ecr.us-east-1.amazonaws.com/prism@{DIGEST}"


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_order: list[str] = []

    def put_object(self, **kwargs):
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]
        self.put_order.append(kwargs["Key"])
        return {}

    def get_object(self, **kwargs):
        try:
            body = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        except KeyError as error:
            raise MissingObject() from error
        return {"Body": BytesIO(body)}


class FakeBatch:
    def __init__(self, status: str) -> None:
        self.status = status

    def describe_jobs(self, **kwargs):
        return {"jobs": [{"jobId": JOB_ID, "status": self.status, "container": {}}]}


def _fake_phase1(spec, output, **kwargs):
    root = Path(output)
    artifact = root / "experiment/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"passed":true}\n', encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "prism": {"package_version": "1.0.0", "git_revision": "abc123"},
        "experiment": {
            "kind": "accepted_milestone5_native_parity",
            "experiment_id": "baseline__seed_1729",
            "spec_sha256": "c" * 64,
            "source_sha256": {"experiment_spec": "d" * 64},
        },
        "artifacts": {"experiment/result.json": digest},
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _submission(bundle_hash: str) -> Submission:
    return Submission(
        RUN_ID,
        JOB_ID,
        "us-east-1",
        BUCKET,
        "queue",
        "definition:1",
        IMAGE,
        DIGEST,
        bundle_hash,
    )


def test_bootstrap_publishes_completion_last_and_download_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s3 = FakeS3()
    cloud = AwsCloud(region="us-east-1", s3=s3)
    bundle = build_input_bundle(ROOT / "container/phase1-experiment.json")
    cloud.upload_input_bundle(bucket=BUCKET, run_id=RUN_ID, bundle=bundle)
    monkeypatch.setattr("prism.cloud.bootstrap.run_container_experiment", _fake_phase1)

    completion = run_bootstrap(
        cloud=cloud,
        bucket=BUCKET,
        run_id=RUN_ID,
        batch_job_id=JOB_ID,
        image_uri=IMAGE,
        workspace=tmp_path / "work",
    )
    assert completion["cloud_execution"]["runtime_architecture"] == "ARM64"
    assert s3.put_order[-1].endswith("completion_manifest.json")
    assert [item["path"] for item in completion["artifacts"]] == [
        "experiment/result.json",
        "run_manifest.json",
    ]

    verifier = AwsCloud(region="us-east-1", s3=s3, batch=FakeBatch("SUCCEEDED"))
    submission = _submission(bundle.manifest_sha256)
    assert verifier.inspect_result(submission).state == "verified"
    output = verifier.download_verified(submission, tmp_path / "download")
    assert (output / "run_manifest.json").is_file()
    assert verifier.download_verified(submission, output) == output

    (output / "unexpected.txt").write_text("caller data", encoding="utf-8")
    with pytest.raises(CloudContractError, match="unexpected"):
        verifier.download_verified(submission, output)


def test_bootstrap_rejects_input_hash_failure_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s3 = FakeS3()
    cloud = AwsCloud(region="us-east-1", s3=s3)
    bundle = build_input_bundle(ROOT / "container/phase1-experiment.json")
    cloud.upload_input_bundle(bucket=BUCKET, run_id=RUN_ID, bundle=bundle)
    first_member = next(iter(bundle.members))
    key = (BUCKET, f"prism-cloud/v1/runs/{RUN_ID}/input/{first_member}")
    s3.objects[key] += b"corrupt"
    monkeypatch.setattr("prism.cloud.bootstrap.run_container_experiment", _fake_phase1)

    with pytest.raises(CloudContractError, match="hash or size mismatch"):
        run_bootstrap(
            cloud=cloud,
            bucket=BUCKET,
            run_id=RUN_ID,
            batch_job_id=JOB_ID,
            image_uri=IMAGE,
            workspace=tmp_path / "work",
        )
    assert not any("/attempts/" in key for _, key in s3.objects)


def test_succeeded_batch_without_completion_is_incomplete() -> None:
    s3 = FakeS3()
    cloud = AwsCloud(region="us-east-1", s3=s3, batch=FakeBatch("SUCCEEDED"))
    assert cloud.inspect_result(_submission("e" * 64)).state == (
        "incomplete — completion manifest missing"
    )


def test_bootstrap_cli_fails_nonzero_without_required_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "PRISM_CLOUD_REGION",
        "PRISM_CLOUD_BUCKET",
        "PRISM_CLOUD_RUN_ID",
        "AWS_BATCH_JOB_ID",
        "PRISM_CLOUD_IMAGE_URI",
    ):
        monkeypatch.delenv(name, raising=False)
    assert main([]) == 2
