from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from prism.workload import OutputDirectoryError, generate_workload, persist_workload
from prism.workload.generator import ARTIFACT_FILENAMES
from prism.workload.models import OBSERVABLE_EVENT_FIELDS
from tests.conftest import REPRESENTATIVE_CONFIG_PATH, REPOSITORY_ROOT


def _artifact_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    return environment


def test_persistence_is_byte_deterministic_and_parseable(tmp_path, make_config) -> None:
    result_a = generate_workload(make_config(seed=73))
    result_b = generate_workload(make_config(seed=73))
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"

    persist_workload(result_a, output_a)
    persist_workload(result_b, output_b)

    artifacts_a = _artifact_bytes(output_a)
    artifacts_b = _artifact_bytes(output_b)
    assert set(artifacts_a) == set(ARTIFACT_FILENAMES)
    assert artifacts_a == artifacts_b
    assert all(content.endswith(b"\n") for content in artifacts_a.values())

    assert json.loads(artifacts_a["config.json"]) == result_a.config.to_dict()
    assert json.loads(artifacts_a["summary.json"]) == result_a.summary.to_dict()
    assert json.loads(artifacts_a["hidden_ground_truth.json"]) == (
        result_a.hidden_ground_truth.to_dict()
    )
    events = [
        json.loads(line)
        for line in artifacts_a["observable_events.jsonl"].splitlines()
    ]
    assert [event["event_index"] for event in events] == list(range(len(events)))
    assert all(set(event) == OBSERVABLE_EVENT_FIELDS for event in events)


def test_nonempty_output_directory_is_rejected(tmp_path, make_config) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    existing = destination / "keep.txt"
    existing.write_text("user data\n", encoding="utf-8")

    with pytest.raises(OutputDirectoryError, match="must be empty"):
        persist_workload(generate_workload(make_config()), destination)

    assert existing.read_text(encoding="utf-8") == "user data\n"


def test_cli_creates_only_required_artifacts_and_prints_summary(tmp_path) -> None:
    destination = tmp_path / "representative"
    command = [
        sys.executable,
        "-m",
        "prism.workload.cli",
        "--config",
        str(REPRESENTATIVE_CONFIG_PATH),
        "--output-dir",
        str(destination),
    ]

    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Generated Prism Milestone 1 workload" in completed.stdout
    assert "Events: 2604" in completed.stdout
    assert {path.name for path in destination.iterdir()} == set(ARTIFACT_FILENAMES)


def test_cli_reports_invalid_configuration(tmp_path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"seed": 1}\n', encoding="utf-8")
    destination = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "prism.workload.cli",
        "--config",
        str(invalid_path),
        "--output-dir",
        str(destination),
    ]

    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "missing required configuration fields" in completed.stderr
    assert not destination.exists()
