from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prism.container.verify import VerificationError, verify_outputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(
    path: Path,
    *,
    numeric_delta: float = 0.0,
    report_count: int = 4,
    runtime_numpy: str = "2.3.2",
    exact_note: str | None = None,
    discrete_id: int = 2,
) -> Path:
    exact = path / "experiment/runs/example/resolved_workload_config.json"
    numerical = path / "experiment/runs/example/structure/learned_structure.npz"
    report = path / "experiment/runs/example/structure/recovery_report.json"
    exact.parent.mkdir(parents=True)
    numerical.parent.mkdir(parents=True)
    exact.write_text('{"seed":1729}\n', encoding="utf-8")
    if exact_note is not None:
        (path / "experiment/notes.txt").write_text(exact_note, encoding="utf-8")
    np.savez(
        numerical,
        membership=np.asarray([0.25 + numeric_delta, 0.75 - numeric_delta]),
        ids=np.asarray([1, discrete_id], dtype=np.int64),
    )
    report.write_text(
        json.dumps(
            {
                "count": report_count,
                "passed": True,
                "metric": 0.5 + numeric_delta,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {
        item.relative_to(path).as_posix(): _sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    manifest = {
        "schema_version": 1,
        "prism": {"package_version": "1.0.0", "git_revision": "abc"},
        "experiment": {
            "kind": "accepted_milestone5_native_parity",
            "experiment_id": "baseline__seed_1729",
            "spec_sha256": "a" * 64,
            "source_sha256": {"spec": "b" * 64},
        },
        "runtime": {
            "python_version": "3.12.10",
            "python_implementation": "CPython",
            "os": {"id": "debian", "version_id": "12"},
            "machine": "aarch64",
            "dependency_versions": {
                "numpy": runtime_numpy,
                "scipy": "1.16.1",
                "scikit-learn": "1.9.0",
            },
        },
        "native_build": {
            "build_type": "Release",
            "compiler_id": "GNU",
            "compiler_version": "12.2.0",
            "extension_imported": True,
            "parity_passed": True,
        },
        "artifacts": artifacts,
    }
    (path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_repeat_requires_complete_byte_identity(tmp_path: Path) -> None:
    left = _root(tmp_path / "left")
    right = _root(tmp_path / "right")
    report = verify_outputs(left, right, mode="repeat")
    assert report["passed"] is True
    assert report["byte_identical_file_count"] == 4

    (right / "experiment/runs/example/resolved_workload_config.json").write_text(
        '{"seed":2718}\n', encoding="utf-8"
    )
    with pytest.raises(VerificationError, match="artifact hashes"):
        verify_outputs(left, right, mode="repeat")


def test_cross_platform_uses_exact_discrete_and_tight_numerical_contract(
    tmp_path: Path,
) -> None:
    left = _root(tmp_path / "left")
    right = _root(tmp_path / "right", numeric_delta=5e-10)
    report = verify_outputs(left, right, mode="cross-platform")
    assert report == {
        "mode": "cross-platform",
        "passed": True,
        "byte_identical_file_count": 1,
        "numerical_file_count": 1,
        "semantic_report_count": 1,
        "verified_artifact_count": 3,
        "absolute_tolerance": 1e-9,
        "maximum_absolute_difference": pytest.approx(5e-10),
    }

    far = _root(tmp_path / "far", numeric_delta=2e-9)
    with pytest.raises(VerificationError, match="numerical array"):
        verify_outputs(left, far, mode="cross-platform")


def test_cross_host_requires_same_declared_runtime_and_allows_numerical_drift(
    tmp_path: Path,
) -> None:
    left = _root(tmp_path / "left")
    right = _root(tmp_path / "right", numeric_delta=5e-10)
    report = verify_outputs(left, right, mode="cross-host")
    assert report["passed"] is True
    assert report["mode"] == "cross-host"
    assert report["verified_artifact_count"] == 3

    incompatible = _root(tmp_path / "incompatible", runtime_numpy="2.4.0")
    with pytest.raises(VerificationError, match="declared runtime"):
        verify_outputs(left, incompatible, mode="cross-host")


def test_semantic_parity_is_exact_for_unclassified_artifacts_and_integers(
    tmp_path: Path,
) -> None:
    left = _root(tmp_path / "left", exact_note="same host-independent text\n")
    changed_note = _root(tmp_path / "changed-note", exact_note="changed text\n")
    with pytest.raises(VerificationError, match="byte-identical artifact"):
        verify_outputs(left, changed_note, mode="cross-host")

    changed_integer = _root(
        tmp_path / "changed-integer",
        report_count=5,
        exact_note="same host-independent text\n",
    )
    with pytest.raises(VerificationError, match="JSON integer differs"):
        verify_outputs(left, changed_integer, mode="cross-host")

    changed_discrete = _root(
        tmp_path / "changed-discrete",
        exact_note="same host-independent text\n",
        discrete_id=3,
    )
    with pytest.raises(VerificationError, match="discrete array differs"):
        verify_outputs(left, changed_discrete, mode="cross-host")
