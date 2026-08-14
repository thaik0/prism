from __future__ import annotations

from io import BytesIO

import pytest

from prism.cloud.aws import AwsCloud, build_submit_request
from prism.cloud.contract import CloudContractError


DIGEST = "sha256:" + "b" * 64
IMAGE = f"123456789012.dkr.ecr.us-east-1.amazonaws.com/prism@{DIGEST}"
RUN_ID = "pcv1-" + "a" * 24


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, **kwargs):
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]
        return {}

    def get_object(self, **kwargs):
        return {"Body": BytesIO(self.objects[(kwargs["Bucket"], kwargs["Key"])])}


class FakeEcr:
    def describe_images(self, **kwargs):
        assert kwargs["repositoryName"] == "prism"
        return {"imageDetails": [{"imageDigest": DIGEST}]}


class FakeBatch:
    def __init__(self, definition):
        self.definition = definition
        self.describe_requests = []
        self.submitted = None

    def describe_job_definitions(self, **kwargs):
        self.describe_requests.append(kwargs)
        return {"jobDefinitions": [self.definition]}

    def submit_job(self, **kwargs):
        self.submitted = kwargs
        return {"jobId": "11111111-1111-1111-1111-111111111111"}


def _definition() -> dict:
    return {
        "revision": 3,
        "jobDefinitionArn": "arn:aws:batch:us-east-1:123456789012:job-definition/prism:3",
        "platformCapabilities": ["FARGATE"],
        "retryStrategy": {"attempts": 1},
        "containerProperties": {
            "image": IMAGE,
            "command": ["prism-cloud-bootstrap"],
            "executionRoleArn": "arn:execution",
            "jobRoleArn": "arn:job",
            "runtimePlatform": {
                "operatingSystemFamily": "LINUX",
                "cpuArchitecture": "ARM64",
            },
            "networkConfiguration": {"assignPublicIp": "ENABLED"},
            "logConfiguration": {"logDriver": "awslogs"},
            "resourceRequirements": [
                {"type": "VCPU", "value": "1"},
                {"type": "MEMORY", "value": "2048"},
            ],
        },
    }


def test_ecr_resolution_and_arm64_fargate_job_definition_validation() -> None:
    batch = FakeBatch(_definition())
    cloud = AwsCloud(
        region="us-east-1", s3=FakeS3(), ecr=FakeEcr(), batch=batch
    )
    image, digest = cloud.resolve_ecr_image(
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/prism:phase1"
    )
    assert (image, digest) == (IMAGE, DIGEST)
    assert cloud.validate_job_definition("prism", IMAGE).endswith("prism:3")
    assert batch.describe_requests == [
        {"jobDefinitionName": "prism", "status": "ACTIVE"}
    ]

    invalid = _definition()
    invalid["containerProperties"]["runtimePlatform"]["cpuArchitecture"] = "X86_64"
    with pytest.raises(CloudContractError, match="Linux/ARM64"):
        AwsCloud(
            region="us-east-1", s3=FakeS3(), batch=FakeBatch(invalid)
        ).validate_job_definition("prism", IMAGE)


def test_job_definition_lookup_uses_boto3_name_filter_contract() -> None:
    boto3 = pytest.importorskip("boto3")
    stubber_module = pytest.importorskip("botocore.stub")
    batch = boto3.client(
        "batch",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )
    definition = _definition()
    definition.update(
        {
            "jobDefinitionName": "prism",
            "status": "ACTIVE",
            "type": "container",
        }
    )
    with stubber_module.Stubber(batch) as stubber:
        stubber.add_response(
            "describe_job_definitions",
            {"jobDefinitions": [definition]},
            {"jobDefinitionName": "prism", "status": "ACTIVE"},
        )
        cloud = AwsCloud(region="us-east-1", s3=FakeS3(), batch=batch)
        assert cloud.validate_job_definition("prism", IMAGE).endswith("prism:3")


def test_batch_submission_request_has_only_allowlisted_overrides() -> None:
    request = build_submit_request(
        run_id=RUN_ID,
        region="us-east-1",
        bucket="prism-example",
        job_queue="queue",
        job_definition="definition:3",
        image_uri=IMAGE,
    )
    assert set(request["containerOverrides"]) == {"environment"}
    assert [item["name"] for item in request["containerOverrides"]["environment"]] == [
        "PRISM_CLOUD_BUCKET",
        "PRISM_CLOUD_IMAGE_URI",
        "PRISM_CLOUD_REGION",
        "PRISM_CLOUD_RUN_ID",
    ]
    assert request["retryStrategy"] == {"attempts": 1}
    assert "command" not in request["containerOverrides"]
