"""Command-line entry point for Milestone 2 structural recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from prism.structure.config import StructureLearnerConfigError
from prism.structure.demand import DemandMatrixError
from prism.structure.evaluate import RecoveryEvaluationError
from prism.structure.learner import StructureFitError
from prism.structure.persistence import (
    StructureOutputDirectoryError,
    run_structure_recovery,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover fuzzy working-set structure from one persisted trace."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_structure_recovery(
            arguments.run_dir, arguments.config, arguments.output_dir
        )
    except (
        OSError,
        DemandMatrixError,
        RecoveryEvaluationError,
        StructureFitError,
        StructureLearnerConfigError,
        StructureOutputDirectoryError,
    ) as error:
        parser.error(str(error))

    learned = result.learned_structure
    report = result.recovery_evaluation.report
    fuzzy = report["fuzzy_membership_recovery"]["aggregate"]
    support = report["support_recovery"]["aggregate"]
    reconstruction = report["reconstruction"]
    print("Completed Prism Milestone 2 structural recovery")
    print(f"Output directory: {arguments.output_dir}")
    print(f"Demand matrix shape: {result.demand_matrix.X.shape}")
    print(f"Factors: {learned.membership_matrix.shape[0]}")
    print(
        f"Converged: {learned.converged} in {learned.iteration_count} iterations"
    )
    print(
        "Normalized reconstruction error: "
        f"{reconstruction['normalized_frobenius_error']}"
    )
    print(f"Mean fuzzy cosine similarity: {fuzzy['mean']}")
    print(
        "Mean support recall versus chance: "
        f"{support['mean_learned_support_recall']} > "
        f"{support['mean_analytic_random_support_expectation']}"
    )
    print(
        "Representative gate: "
        f"{report['representative_gate']['passed']}"
    )
    print(f"Warnings: {len(report['warnings'])}")
    if not learned.converged:
        print(
            "NMF did not converge; diagnostics were written to recovery_report.json",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
