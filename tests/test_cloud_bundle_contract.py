from __future__ import annotations

from pathlib import Path

import pytest

from prism.cloud.bundle import (
    build_input_bundle,
    stable_run_id,
    verify_input_bundle,
)
from prism.cloud.contract import (
    CloudContractError,
    REQUIRED_INPUT_MEMBERS,
    input_manifest,
    safe_relative_path,
)


ROOT = Path(__file__).resolve().parents[1]


def test_input_bundle_is_minimal_deterministic_and_hash_validated() -> None:
    first = build_input_bundle(ROOT / "container/phase1-experiment.json")
    second = build_input_bundle(ROOT / "container/phase1-experiment.json")
    assert first.manifest_bytes == second.manifest_bytes
    assert first.manifest_sha256 == second.manifest_sha256
    assert tuple(sorted(first.members)) == REQUIRED_INPUT_MEMBERS
    assert input_manifest(first.manifest_bytes)["experiment_id"] == "baseline__seed_1729"
    verify_input_bundle(first.manifest_bytes, first.members)

    corrupted = dict(first.members)
    corrupted[REQUIRED_INPUT_MEMBERS[0]] += b"corrupt"
    with pytest.raises(CloudContractError, match="hash or size mismatch"):
        verify_input_bundle(first.manifest_bytes, corrupted)


def test_run_identity_is_stable_and_submission_sensitive() -> None:
    values = {
        "input_manifest_sha256": "a" * 64,
        "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/prism@sha256:"
        + "b" * 64,
        "region": "us-east-1",
        "bucket": "prism-example",
        "job_queue": "prism-queue",
        "job_definition": "arn:aws:batch:us-east-1:123456789012:job-definition/prism:1",
    }
    first = stable_run_id(**values)
    assert first == stable_run_id(**values)
    assert first.startswith("pcv1-") and len(first) == 29
    assert first != stable_run_id(**{**values, "job_queue": "different"})


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a//b", "a\\b", "a/./b"])
def test_cloud_paths_reject_unsafe_or_non_normalized_values(path: str) -> None:
    with pytest.raises(CloudContractError):
        safe_relative_path(path)
