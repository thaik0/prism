"""Local CLI for one accepted Prism experiment on AWS Batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from prism.cloud.aws import AwsCloud
from prism.cloud.bundle import build_input_bundle, stable_run_id
from prism.cloud.contract import CloudContractError


def _setting(value: str | None, environment: str, description: str) -> str:
    resolved = value or os.environ.get(environment)
    if not resolved:
        raise CloudContractError(
            f"{description} is required (option or {environment})"
        )
    return resolved


def _cloud(args: argparse.Namespace) -> tuple[AwsCloud, str]:
    region = _setting(args.region, "AWS_REGION", "AWS Region")
    bucket = _setting(args.bucket, "PRISM_CLOUD_BUCKET", "Prism Cloud bucket")
    return AwsCloud.from_boto3(region), bucket


def _submit(args: argparse.Namespace) -> dict[str, object]:
    cloud, bucket = _cloud(args)
    bundle = build_input_bundle(args.spec)
    image_uri, digest = cloud.resolve_ecr_image(
        _setting(args.image, "PRISM_CLOUD_IMAGE", "ECR image")
    )
    queue = _setting(args.job_queue, "PRISM_CLOUD_JOB_QUEUE", "Batch job queue")
    requested_definition = _setting(
        args.job_definition,
        "PRISM_CLOUD_JOB_DEFINITION",
        "Batch job definition",
    )
    definition = cloud.validate_job_definition(requested_definition, image_uri)
    run_id = stable_run_id(
        input_manifest_sha256=bundle.manifest_sha256,
        image_uri=image_uri,
        region=cloud.region,
        bucket=bucket,
        job_queue=queue,
        job_definition=definition,
    )
    cloud.upload_input_bundle(bucket=bucket, run_id=run_id, bundle=bundle)
    submission = cloud.submit(
        run_id=run_id,
        bucket=bucket,
        job_queue=queue,
        job_definition=definition,
        image_uri=image_uri,
        image_digest=digest,
        input_manifest_sha256=bundle.manifest_sha256,
    )
    return {
        "batch_job_id": submission.batch_job_id,
        "run_id": submission.run_id,
        "status": "SUBMITTED",
    }


def _status(args: argparse.Namespace) -> dict[str, object]:
    cloud, bucket = _cloud(args)
    submission = cloud.read_submission(bucket, args.run_id)
    job = cloud.describe_job(submission.batch_job_id)
    inspection = cloud.inspect_result(submission)
    return {
        "batch": job["status"],
        "batch_job_id": submission.batch_job_id,
        "prism_result": inspection.state,
        "run_id": submission.run_id,
    }


def _logs(args: argparse.Namespace) -> dict[str, object]:
    cloud, bucket = _cloud(args)
    submission = cloud.read_submission(bucket, args.run_id)
    return {
        "batch_job_id": submission.batch_job_id,
        "messages": cloud.recent_logs(submission.batch_job_id, limit=args.limit),
        "run_id": submission.run_id,
    }


def _wait(args: argparse.Namespace) -> dict[str, object]:
    cloud, bucket = _cloud(args)
    submission = cloud.read_submission(bucket, args.run_id)
    job, inspection = cloud.wait_for_result(
        submission,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    return {
        "batch": job["status"],
        "batch_job_id": submission.batch_job_id,
        "prism_result": inspection.state,
        "run_id": submission.run_id,
    }


def _download(args: argparse.Namespace) -> dict[str, object]:
    cloud, bucket = _cloud(args)
    submission = cloud.read_submission(bucket, args.run_id)
    output = cloud.download_verified(submission, args.output_dir)
    return {
        "batch_job_id": submission.batch_job_id,
        "output_dir": str(output),
        "run_id": submission.run_id,
        "status": "verified",
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region")
    parser.add_argument("--bucket")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit and verify the accepted Prism experiment on AWS Batch."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit")
    _common(submit)
    submit.add_argument("--spec", required=True, type=Path)
    submit.add_argument("--image")
    submit.add_argument("--job-queue")
    submit.add_argument("--job-definition")
    submit.set_defaults(handler=_submit)

    status = commands.add_parser("status")
    _common(status)
    status.add_argument("run_id")
    status.set_defaults(handler=_status)

    logs = commands.add_parser("logs")
    _common(logs)
    logs.add_argument("run_id")
    logs.add_argument("--limit", type=int, default=200)
    logs.set_defaults(handler=_logs)

    wait = commands.add_parser("wait")
    _common(wait)
    wait.add_argument("run_id")
    wait.add_argument("--poll-interval-seconds", type=float, default=20.0)
    wait.add_argument("--timeout-seconds", type=float, default=3600.0)
    wait.set_defaults(handler=_wait)

    download = commands.add_parser("download")
    _common(download)
    download.add_argument("run_id")
    download.add_argument("--output-dir", required=True, type=Path)
    download.set_defaults(handler=_download)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = args.handler(args)
    except Exception as error:
        print(f"prism-cloud: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
