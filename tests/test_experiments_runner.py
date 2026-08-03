from __future__ import annotations

import json

import pytest

from prism.experiments.runner import ExperimentRunError, run_experiments
from tests.conftest import REPOSITORY_ROOT


MANIFEST = REPOSITORY_ROOT / "configs" / "milestone5_experiments.json"


def _stub_aggregate(monkeypatch) -> None:
    monkeypatch.setattr(
        "prism.experiments.aggregate.write_aggregate_outputs",
        lambda output_dir, manifest: {},
    )


def _completed(manifest, variant_id, seed, run_dir, entry):
    marker = run_dir / "marker.txt"
    marker.write_text(f"{variant_id}:{seed}\n", encoding="utf-8")
    from prism.experiments.runner import _hash_run_artifacts

    return {
        "resolved_configuration_sha256": {"fixture": "a" * 64},
        "source_artifact_sha256": {"fixture": "b" * 64},
        "artifact_sha256": _hash_run_artifacts(run_dir),
        "scientific_gate_outcomes": {"fixture_gate": False},
    }


def test_one_experiment_lifecycle_keeps_scientific_failure_completed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("prism.experiments.runner._execute_run", _completed)
    _stub_aggregate(monkeypatch)
    output = tmp_path / "output"

    result = run_experiments(
        MANIFEST, output, experiment_id="context_weak__seed_31415"
    )
    index = json.loads((output / "experiment_index.json").read_text())
    selected = next(
        row
        for row in index["runs"]
        if row["experiment_id"] == "context_weak__seed_31415"
    )

    assert result.completed_count == 1
    assert result.failed_count == 0
    assert selected["status"] == "completed"
    assert selected["scientific_gate_outcomes"] == {"fixture_gate": False}
    assert sum(row["status"] == "pending" for row in index["runs"]) == 35


def test_engineering_failure_is_explicit_and_resume_reruns_it(
    tmp_path, monkeypatch
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("fixture engineering failure")

    output = tmp_path / "output"
    _stub_aggregate(monkeypatch)
    monkeypatch.setattr("prism.experiments.runner._execute_run", fail)
    first = run_experiments(
        MANIFEST, output, experiment_id="baseline__seed_1729"
    )
    assert first.failed_count == 1
    status = json.loads(
        (output / "runs" / "baseline__seed_1729" / "run_status.json").read_text()
    )
    assert status["failure"] == {
        "type": "RuntimeError",
        "message": "fixture engineering failure",
    }

    monkeypatch.setattr("prism.experiments.runner._execute_run", _completed)
    resumed = run_experiments(
        MANIFEST,
        output,
        experiment_id="baseline__seed_1729",
        resume=True,
    )
    assert resumed.completed_count == 1
    assert resumed.failed_count == 0


def test_resume_reuses_only_hash_verified_completed_runs(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output"
    _stub_aggregate(monkeypatch)
    monkeypatch.setattr("prism.experiments.runner._execute_run", _completed)
    run_experiments(MANIFEST, output, experiment_id="baseline__seed_1729")

    def must_not_run(*args, **kwargs):
        raise AssertionError("completed run was recomputed")

    monkeypatch.setattr("prism.experiments.runner._execute_run", must_not_run)
    resumed = run_experiments(
        MANIFEST,
        output,
        experiment_id="baseline__seed_1729",
        resume=True,
    )
    assert resumed.reused_experiment_ids == ("baseline__seed_1729",)

    marker = output / "runs" / "baseline__seed_1729" / "marker.txt"
    marker.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ExperimentRunError, match="artifact hash mismatch"):
        run_experiments(
            MANIFEST,
            output,
            experiment_id="baseline__seed_1729",
            resume=True,
        )


def test_nonempty_output_and_manifest_hash_mismatch_are_rejected(
    tmp_path, monkeypatch
) -> None:
    occupied = tmp_path / "occupied"
    _stub_aggregate(monkeypatch)
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(ExperimentRunError, match="must be empty"):
        run_experiments(MANIFEST, occupied)

    output = tmp_path / "output"
    monkeypatch.setattr("prism.experiments.runner._execute_run", _completed)
    run_experiments(MANIFEST, output, experiment_id="baseline__seed_1729")
    persisted_path = output / "experiment_manifest.json"
    persisted = json.loads(persisted_path.read_text())
    persisted["source_manifest_sha256"] = "0" * 64
    persisted_path.write_text(
        json.dumps(persisted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ExperimentRunError, match="manifest hash"):
        run_experiments(
            MANIFEST,
            output,
            experiment_id="baseline__seed_1729",
            resume=True,
        )
