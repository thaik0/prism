"""Command-line entry point for the Milestone 3 fast predictor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from prism.predictor.config import PredictorConfigError
from prism.predictor.features import PredictorFeatureError
from prism.predictor.models import PredictorFitError
from prism.predictor.persistence import (
    PredictorOutputDirectoryError,
    run_predictor_experiment,
)
from prism.predictor.targets import PredictorTargetError
from prism.structure.config import StructureLearnerConfigError
from prism.structure.demand import DemandMatrixError
from prism.structure.learner import StructureFitError
from prism.workload.config import WorkloadConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the Prism Milestone 3 fast predictor."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--structure-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_predictor_experiment(
            arguments.run_dir,
            arguments.structure_config,
            arguments.config,
            arguments.output_dir,
        )
    except (
        OSError,
        DemandMatrixError,
        PredictorConfigError,
        PredictorFeatureError,
        PredictorFitError,
        PredictorOutputDirectoryError,
        PredictorTargetError,
        StructureFitError,
        StructureLearnerConfigError,
        WorkloadConfigError,
        ValueError,
    ) as error:
        parser.error(str(error))

    report = result.evaluation.report
    splits = report["split_counts"]
    gates = report["scientific_gates"]
    print("Completed Prism Milestone 3 fast prediction")
    print(f"Output directory: {arguments.output_dir}")
    print(
        "Examples: "
        f"train={splits['training']['example_count']}, "
        f"validation={splits['validation']['example_count']}, "
        f"test={splits['test']['example_count']}"
    )
    print(
        "Training-only NMF: "
        f"converged={result.training_structure.converged}, "
        f"iterations={result.training_structure.iteration_count}"
    )
    print(
        "Activation models: "
        f"recent_converged={result.predictor.convergence['recent_activation']['converged']}, "
        f"context_converged={result.predictor.convergence['context_activation']['converged']}"
    )
    print(
        "Scientific gates: "
        f"gate1={gates['gate_1']['passed']}, "
        f"gate2={gates['gate_2']['passed']}, "
        f"gate3={gates['gate_3']['passed']}"
    )
    print(f"Warnings: {len(report['warnings'])}")
    if not result.training_structure.converged or not result.predictor.converged:
        print(
            "one or more fitted models did not converge; diagnostics were written",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
