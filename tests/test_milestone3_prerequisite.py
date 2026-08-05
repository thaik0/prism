from __future__ import annotations

from pathlib import Path

import numpy as np

from prism.structure import run_structure_recovery
from prism.workload import WorkloadConfig, generate_workload, persist_workload
from prism.workload.generator import ARTIFACT_FILENAMES
from prism.workload.validate import main as validation_main
from prism.workload.validate import validate_workload_run
from tests.conftest import (
    MILESTONE3_WORKLOAD_CONFIG_PATH,
    REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
)


def _artifact_bytes(directory: Path) -> dict[str, bytes]:
    return {
        filename: (directory / filename).read_bytes()
        for filename in ARTIFACT_FILENAMES
    }


def test_milestone3_workload_signal_determinism_and_structure_recovery(
    tmp_path,
) -> None:
    config = WorkloadConfig.from_json(MILESTONE3_WORKLOAD_CONFIG_PATH)
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    persist_workload(generate_workload(config), source_a)
    persist_workload(generate_workload(config), source_b)

    assert _artifact_bytes(source_a) == _artifact_bytes(source_b)
    assert validation_main(
        [
            "--run-dir",
            str(source_a),
            "--require-demonstrations",
            "--require-intensity-signal",
        ]
    ) == 0
    first_validation_bytes = (source_a / "workload_validation.json").read_bytes()
    assert validation_main(
        [
            "--run-dir",
            str(source_a),
            "--require-demonstrations",
            "--require-intensity-signal",
        ]
    ) == 0
    assert (source_a / "workload_validation.json").read_bytes() == (
        first_validation_bytes
    )

    validation = validate_workload_run(source_a)
    assert validation.demonstrations_passed
    assert validation.intensity_signal_passed
    result = run_structure_recovery(
        source_a,
        REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
        tmp_path / "structure",
    )
    report = result.recovery_evaluation.report

    assert result.demand_matrix.X.shape == (1000, 64)
    assert result.learned_structure.converged
    np.testing.assert_allclose(
        result.learned_structure.membership_matrix.sum(axis=1), np.ones(4)
    )
    assert report["support_recovery"]["aggregate"][
        "mean_learned_support_recall"
    ] > report["support_recovery"]["aggregate"][
        "mean_analytic_random_support_expectation"
    ]
    assert report["representative_gate"]["passed"]
