from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prism.container.verify import VerificationError, verify_outputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(path: Path, *, numeric_delta: float = 0.0) -> Path:
    exact = path / "experiment/runs/example/resolved_workload_config.json"
    numerical = path / "experiment/runs/example/structure/learned_structure.npz"
    report = path / "experiment/runs/example/structure/recovery_report.json"
    exact.parent.mkdir(parents=True)
    numerical.parent.mkdir(parents=True)
    exact.write_text('{"seed":1729}\n', encoding="utf-8")
    np.savez(
        numerical,
        membership=np.asarray([0.25 + numeric_delta, 0.75 - numeric_delta]),
        ids=np.asarray([1, 2], dtype=np.int64),
    )
    report.write_text(
        json.dumps(
            {"passed": True, "metric": 0.5 + numeric_delta},
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
        "absolute_tolerance": 1e-9,
    }

    far = _root(tmp_path / "far", numeric_delta=2e-9)
    with pytest.raises(VerificationError, match="numerical array"):
        verify_outputs(left, far, mode="cross-platform")
