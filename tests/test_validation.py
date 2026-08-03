from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import pytest

from prism.workload import WorkloadConfig, generate_workload, persist_workload
from prism.workload.validate import (
    SOURCE_ARTIFACT_FILENAMES,
    WorkloadValidationError,
    _classify_precursors,
    _demonstration_checks,
    validate_workload_run,
    write_validation_report,
)
from tests.conftest import REPRESENTATIVE_CONFIG_PATH, REPOSITORY_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )


@pytest.fixture
def run_factory(tmp_path, make_config) -> Callable[..., Path]:
    counter = 0

    def factory(**overrides: Any) -> Path:
        nonlocal counter
        counter += 1
        run_dir = tmp_path / f"run-{counter}"
        defaults = {
            "num_windows": 6,
            "spontaneous_activation_probability": 1.0,
            "burst_duration_min_windows": 1,
            "burst_duration_max_windows": 2,
        }
        defaults.update(overrides)
        persist_workload(generate_workload(make_config(**defaults)), run_dir)
        return run_dir

    return factory


@pytest.fixture
def representative_run(tmp_path) -> Path:
    run_dir = tmp_path / "representative"
    config = WorkloadConfig.from_json(REPRESENTATIVE_CONFIG_PATH)
    persist_workload(generate_workload(config), run_dir)
    return run_dir


def test_missing_artifact_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    (run_dir / "summary.json").unlink()

    with pytest.raises(WorkloadValidationError, match="missing required artifact: summary.json"):
        validate_workload_run(run_dir)


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("hidden_ground_truth.json", "{bad json\n", "malformed JSON"),
        ("observable_events.jsonl", "{bad json\n", "malformed JSON"),
    ],
)
def test_malformed_json_and_jsonl_are_rejected(
    run_factory, filename: str, content: str, message: str
) -> None:
    run_dir = run_factory()
    (run_dir / filename).write_text(content, encoding="utf-8")

    with pytest.raises(WorkloadValidationError, match=message):
        validate_workload_run(run_dir)


def test_noncontiguous_event_indices_are_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "observable_events.jsonl"
    events = _read_events(path)
    events[1]["event_index"] = 8
    _write_events(path, events)

    with pytest.raises(WorkloadValidationError, match="indices must be contiguous"):
        validate_workload_run(run_dir)


def test_observable_hidden_field_leakage_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "observable_events.jsonl"
    events = _read_events(path)
    events[0]["working_set_id"] = 0
    _write_events(path, events)

    with pytest.raises(WorkloadValidationError, match="unexpected.*working_set_id"):
        validate_workload_run(run_dir)


def test_invalid_record_reference_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "observable_events.jsonl"
    events = _read_events(path)
    events[0]["record_id"] = 99
    _write_events(path, events)

    with pytest.raises(WorkloadValidationError, match="record_id must be at most"):
        validate_workload_run(run_dir)


def test_inconsistent_record_size_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "observable_events.jsonl"
    events = _read_events(path)
    events[0]["record_size_bytes"] += 1
    _write_events(path, events)

    with pytest.raises(WorkloadValidationError, match="record size is inconsistent"):
        validate_workload_run(run_dir)


def test_summary_event_count_mismatch_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "summary.json"
    summary = _read_json(path)
    summary["total_events"] += 1
    _write_json(path, summary)

    with pytest.raises(WorkloadValidationError, match="summary.total_events"):
        validate_workload_run(run_dir)


def test_source_totals_not_matching_events_are_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "hidden_ground_truth.json"
    hidden = _read_json(path)
    hidden["access_source_counts_by_window"][0]["baseline_access_count"] += 1
    _write_json(path, hidden)

    with pytest.raises(WorkloadValidationError, match="source totals do not match"):
        validate_workload_run(run_dir)


def test_invalid_normalized_vector_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "hidden_ground_truth.json"
    hidden = _read_json(path)
    hidden["user_working_set_affinities"][0]["weights"][0] += 0.25
    _write_json(path, hidden)

    with pytest.raises(WorkloadValidationError, match="must sum to 1"):
        validate_workload_run(run_dir)


def test_overlapping_bursts_are_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "hidden_ground_truth.json"
    hidden = _read_json(path)
    duplicate = dict(hidden["bursts"][0])
    duplicate["burst_id"] = len(hidden["bursts"])
    hidden["bursts"].append(duplicate)
    _write_json(path, hidden)

    with pytest.raises(WorkloadValidationError, match="overlapping bursts"):
        validate_workload_run(run_dir)


def test_activation_outcome_inconsistent_with_burst_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "hidden_ground_truth.json"
    hidden = _read_json(path)
    successful_trial = next(
        trial for trial in hidden["activation_trials"] if trial["activated"]
    )
    successful_trial["created_burst_id"] = None
    _write_json(path, hidden)

    with pytest.raises(WorkloadValidationError, match="created_burst_id"):
        validate_workload_run(run_dir)


def test_activation_probability_formula_mismatch_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "hidden_ground_truth.json"
    hidden = _read_json(path)
    trial = hidden["activation_trials"][0]
    trial["activation_probability"] = 0.0
    _write_json(path, hidden)

    with pytest.raises(
        WorkloadValidationError, match="combined activation probability"
    ):
        validate_workload_run(run_dir)


def test_request_relationship_inconsistency_is_rejected(run_factory) -> None:
    run_dir = run_factory()
    path = run_dir / "observable_events.jsonl"
    events = _read_events(path)
    request_id = next(
        event["request_id"]
        for event in events
        if sum(candidate["request_id"] == event["request_id"] for candidate in events)
        >= 2
    )
    request_events = [event for event in events if event["request_id"] == request_id]
    original = request_events[0]["request_type"]
    request_events[1]["request_type"] = (
        "batch" if original == "interactive" else "interactive"
    )
    _write_events(path, events)

    with pytest.raises(WorkloadValidationError, match="request.*inconsistent"):
        validate_workload_run(run_dir)


def _trial(
    window_id: int,
    working_set_id: int,
    score: float,
    activated: bool = False,
) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "working_set_id": working_set_id,
        "previous_window_precursor_score": score,
        "activation_probability": 0.5,
        "activated": activated,
    }


def test_quartile_classification_includes_exact_boundaries() -> None:
    trials = [_trial(index, 0, float(index)) for index in range(5)]
    warnings: list[str] = []

    classification = _classify_precursors(trials, 1, warnings)
    working_set = classification.report["working_sets"][0]

    assert working_set["q1"] == 1.0
    assert working_set["q3"] == 3.0
    assert classification.trial_classes == {
        (0, 0): "no_clear_precursor",
        (1, 0): "no_clear_precursor",
        (2, 0): "intermediate",
        (3, 0): "clear_precursor",
        (4, 0): "clear_precursor",
    }
    assert warnings == []


def test_insufficient_and_degenerate_quartiles_are_excluded_without_double_counting() -> None:
    trials = [
        _trial(0, 0, 0.5),
        _trial(1, 0, 0.5),
        _trial(2, 0, 0.5),
        _trial(3, 0, 0.5),
        _trial(0, 1, 0.2),
    ]
    warnings: list[str] = []

    classification = _classify_precursors(trials, 2, warnings)
    counts = classification.report["trial_counts"]

    assert classification.report["working_sets"][0]["classification_status"] == "degenerate"
    assert classification.report["working_sets"][1]["classification_status"] == "insufficient"
    assert counts == {
        "clear_precursor": 0,
        "no_clear_precursor": 0,
        "intermediate": 0,
        "excluded_insufficient": 1,
        "excluded_degenerate": 4,
    }
    assert set(classification.trial_classes.values()) == {None}
    assert len(warnings) == 2


def test_all_three_demonstration_categories_are_counted() -> None:
    trials = [
        _trial(0, 0, 0.0, True),
        _trial(1, 0, 1.0),
        _trial(2, 0, 2.0),
        _trial(3, 0, 3.0),
        _trial(4, 0, 4.0),
        _trial(5, 0, 5.0),
        _trial(6, 0, 6.0, True),
        _trial(7, 0, 7.0, False),
    ]
    warnings: list[str] = []
    classification = _classify_precursors(trials, 1, warnings)

    checks = _demonstration_checks(trials, classification, warnings)

    assert checks["clear_precursor_followed_by_burst"] == {
        "count": 1,
        "passed": True,
    }
    assert checks["clear_precursor_followed_by_no_burst"] == {
        "count": 1,
        "passed": True,
    }
    assert checks["no_clear_precursor_followed_by_burst"] == {
        "count": 1,
        "passed": True,
    }
    assert checks["all_required_demonstrations_passed"]


def test_missing_demonstrations_are_reported_but_not_structural_failures(
    run_factory,
) -> None:
    run_dir = run_factory(
        spontaneous_activation_probability=0.0,
        precursor_probability_scale=0.0,
    )

    result = validate_workload_run(run_dir)

    assert result.report["structural_validation"]["passed"]
    assert not result.demonstrations_passed
    assert any("demonstration category" in warning for warning in result.report["warnings"])


def test_report_is_deterministic_hashed_and_preserves_sources(run_factory) -> None:
    run_dir = run_factory()
    source_before = {
        filename: (run_dir / filename).read_bytes()
        for filename in SOURCE_ARTIFACT_FILENAMES
    }

    first = validate_workload_run(run_dir)
    second = validate_workload_run(run_dir)
    assert first == second
    for filename, content in source_before.items():
        assert first.report["source_artifact_sha256"][filename] == hashlib.sha256(
            content
        ).hexdigest()

    report_path = run_dir / "workload_validation.json"
    report_path.write_text("stale derived output\n", encoding="utf-8")
    write_validation_report(first, report_path)
    first_bytes = report_path.read_bytes()
    write_validation_report(second, report_path)
    assert report_path.read_bytes() == first_bytes
    assert first_bytes.endswith(b"\n")
    assert str(run_dir).encode() not in first_bytes
    assert {
        filename: (run_dir / filename).read_bytes()
        for filename in SOURCE_ARTIFACT_FILENAMES
    } == source_before


def test_default_cli_succeeds_but_required_demonstrations_gate_fails(
    run_factory,
) -> None:
    run_dir = run_factory(
        spontaneous_activation_probability=0.0,
        precursor_probability_scale=0.0,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    base_command = [
        sys.executable,
        "-m",
        "prism.workload.validate",
        "--run-dir",
        str(run_dir),
    ]

    default = subprocess.run(
        base_command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    required = subprocess.run(
        [*base_command, "--require-demonstrations"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert default.returncode == 0, default.stderr
    assert "Structural validation: passed" in default.stdout
    assert required.returncode == 1
    assert "required representative demonstrations are incomplete" in required.stderr


def test_representative_diagnostic_arithmetic(representative_run) -> None:
    report = validate_workload_run(representative_run).report
    context = report["context_signal"]
    structure = report["working_set_structure"]
    burst = report["burst_diversity"]
    demand = report["demand_decomposition"]

    assert report["demonstration_checks"]["clear_precursor_followed_by_burst"]["count"] == 11
    assert report["demonstration_checks"]["clear_precursor_followed_by_no_burst"]["count"] == 7
    assert report["demonstration_checks"]["no_clear_precursor_followed_by_burst"]["count"] == 3
    assert context["eligible_activation_trials"] == 76
    assert context["unconditional_activation_rate"] == {
        "numerator": 34,
        "denominator": 76,
        "value": 34 / 76,
    }
    assert context["clear_precursor_precision"] == {
        "numerator": 11,
        "denominator": 18,
        "value": 11 / 18,
    }
    assert context["clear_precursor_recall"] == {
        "numerator": 11,
        "denominator": 25,
        "value": 11 / 25,
    }
    assert context["activation_rate_after_no_clear_precursor"] == {
        "numerator": 3,
        "denominator": 15,
        "value": 3 / 15,
    }
    assert context["classified_bursts_without_clear_precursor"] == {
        "numerator": 14,
        "denominator": 25,
        "value": 14 / 25,
    }
    assert context["clear_precursors_not_followed_by_burst"] == {
        "numerator": 7,
        "denominator": 18,
        "value": 7 / 18,
    }
    support = structure["support_structure"]
    assert support["support_size_summary"] == {
        "count": 4,
        "minimum": 14,
        "maximum": 19,
        "mean": 16.0,
        "median": 15.5,
    }
    assert support["records_in_zero_working_sets"]["numerator"] == 20
    assert support["records_in_exactly_one_working_set"]["numerator"] == 27
    assert support["records_in_multiple_working_sets"]["numerator"] == 17
    balance = structure["working_set_demand_balance"]
    assert [row["access_count"] for row in balance["by_working_set"]] == [
        660,
        455,
        457,
        708,
    ]
    assert balance["dominant_working_set_traffic_share"] == 708 / 2280
    assert balance["working_sets_with_zero_generated_accesses"] == []
    assert burst["total_burst_count"] == 34
    assert [row["burst_count"] for row in burst["burst_count_by_working_set"]] == [
        9,
        7,
        9,
        9,
    ]
    assert burst["duration_summary"] == {
        "count": 34,
        "minimum": 2,
        "maximum": 5,
        "mean": 123 / 34,
        "median": 4.0,
    }
    assert burst["intensity_summary"] == {
        "count": 34,
        "minimum": 1.103925156359538,
        "maximum": 3.8330343749333347,
        "mean": 2.440873105014606,
        "median": 2.3788534123452765,
    }
    assert burst["maximum_simultaneous_active_working_sets"] == 4
    assert burst["simultaneous_burst_window_fraction"] == 36 / 40
    assert demand["global"]["baseline"]["numerator"] == 240
    assert demand["global"]["noise"]["numerator"] == 84
    assert demand["global"]["working_set"]["numerator"] == 2280
    assert demand["windows_with_no_working_set_accesses"] == 1
    assert demand["windows_with_working_set_accesses"] == 39
    assert demand["event_count_per_window_summary"] == {
        "count": 40,
        "minimum": 32,
        "maximum": 107,
        "mean": 65.1,
        "median": 69.0,
    }


def test_representative_observable_associations(representative_run) -> None:
    associations = validate_workload_run(representative_run).report[
        "observable_associations"
    ]

    user = associations["user_id"]["supported_rate_summary"]
    request_type = associations["request_type"]["supported_rate_summary"]
    pair = associations["user_id_request_type"]["supported_rate_summary"]
    assert user["maximum_rate_category"] == 1
    assert user["maximum_rate_sample_count"] == 140
    assert user["maximum_activation_rate"] == 76 / 140
    assert request_type["maximum_rate_category"] == "maintenance"
    assert request_type["maximum_rate_sample_count"] == 247
    assert request_type["maximum_activation_rate"] == 126 / 247
    assert pair["maximum_rate_category"] == {
        "user_id": 4,
        "request_type": "maintenance",
    }
    assert pair["maximum_rate_sample_count"] == 17
    assert pair["maximum_activation_rate"] == 12 / 17


def test_representative_cli_passes_required_demonstrations(representative_run) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    command = [
        sys.executable,
        "-m",
        "prism.workload.validate",
        "--run-dir",
        str(representative_run),
        "--require-demonstrations",
    ]

    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "clear->burst=11, clear->no_burst=7, no_clear->burst=3" in completed.stdout
    assert (representative_run / "workload_validation.json").is_file()
