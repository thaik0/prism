from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from prism.predictor.persistence import run_predictor_experiment
from prism.simulation.persistence import (
    POLICY_TRACE_ARRAYS,
    PROJECTION_ARRAYS,
    SIMULATION_ARTIFACT_FILENAMES,
    SimulationInputError,
    SimulationOutputDirectoryError,
    load_policy_inputs,
    run_simulated_evaluation,
)
from prism.workload import WorkloadConfig, generate_workload, persist_workload
from tests.conftest import (
    MILESTONE3_PREDICTOR_CONFIG_PATH,
    MILESTONE3_WORKLOAD_CONFIG_PATH,
    MILESTONE4_SIMULATION_CONFIG_PATH,
    REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
)


def _bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_representative_simulation_gates_artifacts_and_determinism(tmp_path) -> None:
    source = tmp_path / "source"
    predictor = tmp_path / "predictor"
    persist_workload(
        generate_workload(WorkloadConfig.from_json(MILESTONE3_WORKLOAD_CONFIG_PATH)),
        source,
    )
    run_predictor_experiment(
        source,
        REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
        MILESTONE3_PREDICTOR_CONFIG_PATH,
        predictor,
    )
    source_before = _bytes(source)
    predictor_before = _bytes(predictor)
    output_a = tmp_path / "simulation-a"
    output_b = tmp_path / "simulation-b"

    first = run_simulated_evaluation(
        source, predictor, MILESTONE4_SIMULATION_CONFIG_PATH, output_a
    )
    second = run_simulated_evaluation(
        source, predictor, MILESTONE4_SIMULATION_CONFIG_PATH, output_b
    )
    policy_inputs = load_policy_inputs(
        source, predictor, MILESTONE4_SIMULATION_CONFIG_PATH
    )

    assert set(_bytes(output_a)) == set(SIMULATION_ARTIFACT_FILENAMES)
    assert _bytes(output_a) == _bytes(output_b)
    assert _bytes(source) == source_before
    assert _bytes(predictor) == predictor_before
    assert first.evaluation_report["scientific_gates"]["all_passed"]
    assert second.evaluation_report["scientific_gates"]["all_passed"]
    assert first.replay.capacity_violations == 0
    assert first.evaluation_report["controller_diagnostics"][
        "all_exact_windows_optimal"
    ]
    assert len(first.replay.exact_solver_diagnostics) == 400
    assert first.projection.model.training_target_window_ids[0] == 3
    assert first.projection.model.training_target_window_ids[-1] == 599
    assert len(first.projection.model.training_target_window_ids) == 597
    assert policy_inputs.workload_seed == 1729
    assert policy_inputs.train_end == 600
    assert policy_inputs.validation_end == 800
    assert policy_inputs.evaluation_end == 1000
    assert policy_inputs.record_ids == tuple(range(64))
    assert policy_inputs.observable_demand.flags.writeable is False
    assert policy_inputs.predicted_record_demand.flags.writeable is False
    assert np.array_equal(
        policy_inputs.predicted_record_demand,
        first.projection.predicted_record_demand,
        equal_nan=True,
    )

    with np.load(output_a / "projection_model.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(PROJECTION_ARRAYS)
        assert not any(
            token in name
            for name in archive.files
            for token in ("hidden", "burst", "planted", "oracle", "precursor")
        )
    with np.load(output_a / "policy_traces.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(POLICY_TRACE_ARRAYS)
        assert archive["per_event_tier_cost"].shape[0] == 11
        assert archive["per_window_combined_cost"].shape == (11, 200)
        assert not any(
            token in name
            for name in archive.files
            for token in ("hidden", "burst", "planted", "oracle_demand")
        )
    config = json.loads((output_a / "simulation_config.json").read_text())
    static = config["derived_static_metadata"]
    assert static["total_record_bytes"] == 618914
    assert static["median_record_size_bytes"] == 10013.0
    assert static["resolved_fast_capacity_bytes"] == 154728
    assert static["median_record_promotion_cost"] == pytest.approx(18.0)
    assert str(tmp_path).encode() not in (output_a / "simulation_config.json").read_bytes()
    assert str(tmp_path).encode() not in (output_a / "evaluation_report.json").read_bytes()
    assert (output_a / "simulation_config.json").read_bytes().endswith(b"\n")
    assert (output_a / "evaluation_report.json").read_bytes().endswith(b"\n")


def test_hash_mismatch_and_nonempty_output_are_rejected_safely(tmp_path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    keep = occupied / "keep.txt"
    keep.write_text("keep\n", encoding="utf-8")
    with pytest.raises(SimulationOutputDirectoryError, match="must be empty"):
        run_simulated_evaluation("missing", "missing", "missing", occupied)
    assert keep.read_text(encoding="utf-8") == "keep\n"

    source = tmp_path / "source"
    predictor = tmp_path / "predictor"
    persist_workload(
        generate_workload(WorkloadConfig.from_json(MILESTONE3_WORKLOAD_CONFIG_PATH)),
        source,
    )
    run_predictor_experiment(
        source,
        REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
        MILESTONE3_PREDICTOR_CONFIG_PATH,
        predictor,
    )
    corrupted = tmp_path / "corrupted"
    shutil.copytree(predictor, corrupted)
    raw = json.loads((corrupted / "predictor_config.json").read_text())
    raw["source_artifact_sha256"]["summary.json"] = "0" * 64
    (corrupted / "predictor_config.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(SimulationInputError, match="source hashes"):
        run_simulated_evaluation(
            source,
            corrupted,
            MILESTONE4_SIMULATION_CONFIG_PATH,
            tmp_path / "output",
        )
