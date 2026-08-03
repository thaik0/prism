"""Command-line entry point for Milestone 4 simulated policy evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from prism.simulation.config import SimulationConfigError
from prism.simulation.controllers import ControllerError
from prism.simulation.persistence import (
    SimulationInputError,
    SimulationOutputDirectoryError,
    run_simulated_evaluation,
)
from prism.simulation.projection import ProjectionError
from prism.simulation.replay import ReplayError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate six deterministic policies on one frozen Prism trace."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--predictor-run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_simulated_evaluation(
            arguments.run_dir,
            arguments.predictor_run_dir,
            arguments.config,
            arguments.output_dir,
        )
    except (
        ControllerError,
        OSError,
        ProjectionError,
        ReplayError,
        SimulationConfigError,
        SimulationInputError,
        SimulationOutputDirectoryError,
        ValueError,
    ) as error:
        parser.error(str(error))

    report = result.evaluation_report
    gates = report["scientific_gates"]
    print("Completed Prism Milestone 4 simulated evaluation")
    print(f"Output directory: {arguments.output_dir}")
    print(
        "Splits: training=[0,600), validation=[600,800), test=[800,1000)"
    )
    for policy_name in result.replay.policy_names:
        metrics = report["policy_metrics"][policy_name]
        print(
            f"{policy_name}: combined_cost={metrics['total_combined_cost']}, "
            f"hit_rate={metrics['hit_rate']}, bytes_promoted={metrics['bytes_promoted']}"
        )
    print(
        "Scientific gates: "
        + ", ".join(
            f"gate{index}={gates[f'gate_{index}']['passed']}"
            for index in range(1, 6)
        )
    )
    print(f"Warnings: {len(report['warnings'])}")
    return int(not gates["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
