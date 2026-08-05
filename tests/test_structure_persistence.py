from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from prism.structure import (
    StructureOutputDirectoryError,
    build_demand_matrix,
    run_structure_recovery,
)
from prism.structure.persistence import STRUCTURE_ARTIFACT_FILENAMES
from prism.workload import generate_workload, persist_workload
from prism.workload.validate import SOURCE_ARTIFACT_FILENAMES
from tests.conftest import REPOSITORY_ROOT


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_learner_config(path: Path, *, max_iter: int = 500) -> None:
    _write_json(
        path,
        {
            "n_components": 2,
            "fit_seed": 23,
            "max_iter": max_iter,
            "tolerance": 1e-5,
        },
    )


def _artifact_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_persistence_contract_hashes_source_preservation_and_determinism(
    tmp_path, make_config
) -> None:
    source = tmp_path / "source"
    persist_workload(generate_workload(make_config(num_windows=8)), source)
    source_before = {
        filename: (source / filename).read_bytes()
        for filename in SOURCE_ARTIFACT_FILENAMES
    }
    learner_config = tmp_path / "learner.json"
    _write_learner_config(learner_config)
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"

    first = run_structure_recovery(source, learner_config, output_a)
    second = run_structure_recovery(source, learner_config, output_b)

    assert set(_artifact_bytes(output_a)) == set(STRUCTURE_ARTIFACT_FILENAMES)
    assert _artifact_bytes(output_a) == _artifact_bytes(output_b)
    assert all(
        (output_a / name).read_bytes().endswith(b"\n")
        for name in ("learner_config.json", "recovery_report.json")
    )
    with np.load(output_a / "demand_matrix.npz", allow_pickle=False) as artifact:
        assert set(artifact.files) == {"X", "window_ids", "record_ids"}
        assert np.issubdtype(artifact["X"].dtype, np.integer)
    with np.load(output_a / "learned_structure.npz", allow_pickle=False) as artifact:
        assert set(artifact.files) == {
            "activation_matrix",
            "membership_matrix",
            "factor_ids",
            "window_ids",
            "record_ids",
        }
        assert artifact["membership_matrix"].sum(axis=1) == pytest.approx(
            [1.0, 1.0]
        )
        assert not any("planted" in name or "hidden" in name for name in artifact.files)
    learner_artifact = json.loads(
        (output_a / "learner_config.json").read_text(encoding="utf-8")
    )
    assert learner_artifact["resolved_learner_configuration"]["algorithm"] == (
        "sklearn.decomposition.NMF"
    )
    assert set(learner_artifact["dependency_versions"]) == {
        "python",
        "numpy",
        "scipy",
        "scikit-learn",
    }
    for filename, content in source_before.items():
        assert learner_artifact["source_artifact_sha256"][filename] == (
            hashlib.sha256(content).hexdigest()
        )
    assert {
        filename: (source / filename).read_bytes()
        for filename in SOURCE_ARTIFACT_FILENAMES
    } == source_before
    assert np.array_equal(
        first.learned_structure.membership_matrix,
        second.learned_structure.membership_matrix,
    )


def test_nonempty_output_directory_is_rejected(tmp_path, make_config) -> None:
    source = tmp_path / "source"
    persist_workload(generate_workload(make_config()), source)
    learner_config = tmp_path / "learner.json"
    _write_learner_config(learner_config)
    destination = tmp_path / "occupied"
    destination.mkdir()
    existing = destination / "keep.txt"
    existing.write_text("keep\n", encoding="utf-8")

    with pytest.raises(StructureOutputDirectoryError, match="must be empty"):
        run_structure_recovery(source, learner_config, destination)

    assert existing.read_text(encoding="utf-8") == "keep\n"


def test_hidden_membership_changes_affect_only_evaluation_and_provenance(
    tmp_path, make_config
) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    persist_workload(generate_workload(make_config(num_windows=8)), source_a)
    shutil.copytree(source_a, source_b)
    hidden_path = source_b / "hidden_ground_truth.json"
    hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
    left = hidden["working_set_memberships"][0]["members"]
    right = hidden["working_set_memberships"][1]["members"]
    hidden["working_set_memberships"][0]["members"] = right
    hidden["working_set_memberships"][1]["members"] = left
    _write_json(hidden_path, hidden)
    learner_config = tmp_path / "learner.json"
    _write_learner_config(learner_config)

    result_a = run_structure_recovery(source_a, learner_config, tmp_path / "out-a")
    result_b = run_structure_recovery(source_b, learner_config, tmp_path / "out-b")

    assert np.array_equal(
        build_demand_matrix(source_a).X, build_demand_matrix(source_b).X
    )
    assert np.array_equal(
        result_a.learned_structure.activation_matrix,
        result_b.learned_structure.activation_matrix,
    )
    assert np.array_equal(
        result_a.learned_structure.membership_matrix,
        result_b.learned_structure.membership_matrix,
    )
    assert result_a.recovery_evaluation.report != result_b.recovery_evaluation.report
    assert (tmp_path / "out-a" / "demand_matrix.npz").read_bytes() == (
        tmp_path / "out-b" / "demand_matrix.npz"
    ).read_bytes()
    assert (tmp_path / "out-a" / "learned_structure.npz").read_bytes() == (
        tmp_path / "out-b" / "learned_structure.npz"
    ).read_bytes()


def test_cli_writes_report_then_exits_nonzero_on_nonconvergence(
    tmp_path, make_config
) -> None:
    source = tmp_path / "source"
    persist_workload(generate_workload(make_config(num_windows=8)), source)
    learner_config = tmp_path / "learner.json"
    _write_learner_config(learner_config, max_iter=1)
    output = tmp_path / "output"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prism.structure.cli",
            "--run-dir",
            str(source),
            "--config",
            str(learner_config),
            "--output-dir",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "did not converge" in completed.stderr
    assert set(_artifact_bytes(output)) == set(STRUCTURE_ARTIFACT_FILENAMES)
    report = json.loads((output / "recovery_report.json").read_text())
    assert not report["convergence"]["converged"]
    assert report["convergence"]["iteration_count"] == 1
