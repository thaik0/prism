from __future__ import annotations

import numpy as np

from prism.structure import run_structure_recovery
from prism.workload import WorkloadConfig, generate_workload, persist_workload
from prism.workload.validate import validate_workload_run
from tests.conftest import (
    REPRESENTATIVE_CONFIG_PATH,
    REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
)


def test_representative_structural_recovery_end_to_end(tmp_path) -> None:
    source = tmp_path / "source"
    source_config = WorkloadConfig.from_json(REPRESENTATIVE_CONFIG_PATH)
    persist_workload(generate_workload(source_config), source)
    validation = validate_workload_run(source)
    assert validation.demonstrations_passed

    result = run_structure_recovery(
        source,
        REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
        tmp_path / "structure",
    )
    report = result.recovery_evaluation.report

    assert result.demand_matrix.X.shape == (40, 64)
    assert result.demand_matrix.event_count == 2604
    assert result.learned_structure.converged
    np.testing.assert_allclose(
        result.learned_structure.membership_matrix.sum(axis=1), np.ones(4)
    )
    assert len(report["optimal_assignment"]) == 4
    assert {
        row["learned_factor_id"] for row in report["optimal_assignment"]
    } == set(range(4))
    assert {
        row["planted_factor_id"] for row in report["optimal_assignment"]
    } == set(range(4))
    assert report["fuzzy_membership_recovery"]["aggregate"]["count"] == 4
    assert report["activation_alignment"]["aggregate"]["count"] == 4
    assert report["representative_gate"]["passed"]
    assert report["support_recovery"]["aggregate"][
        "mean_learned_support_recall"
    ] > report["support_recovery"]["aggregate"][
        "mean_analytic_random_support_expectation"
    ]
