"""Command-line entry point for Prism Milestone 1 workload generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from prism.workload.config import WorkloadConfig, WorkloadConfigError
from prism.workload.generator import (
    OutputDirectoryError,
    generate_workload,
    persist_workload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic Prism Milestone 1 workload."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = WorkloadConfig.from_json(arguments.config)
        result = generate_workload(config)
        persist_workload(result, arguments.output_dir)
    except (OSError, OutputDirectoryError, WorkloadConfigError) as error:
        parser.error(str(error))

    summary = result.summary
    print("Generated Prism Milestone 1 workload")
    print(f"Output directory: {arguments.output_dir}")
    print(f"Seed: {summary.seed}")
    print(f"Windows: {summary.num_windows}")
    print(f"Records: {summary.num_records}")
    print(f"Sessions: {summary.total_sessions}")
    print(f"Requests: {summary.total_requests}")
    print(f"Events: {summary.total_events}")
    print(f"Bursts: {summary.total_bursts}")
    print(
        "Access sources: "
        f"baseline={summary.baseline_access_count}, "
        f"noise={summary.noise_access_count}, "
        f"working_set={summary.working_set_access_count}"
    )
    print(
        "Records in multiple working sets: "
        f"{summary.records_in_multiple_working_sets}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
