"""Command-line entry point for the frozen Milestone 5 sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from prism.experiments.config import ActionabilityManifest, ManifestError, load_manifest
from prism.experiments.runner import ExperimentRunError, run_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Prism Milestone 5 experiment sweep."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.print_manifest:
            if isinstance(manifest, ActionabilityManifest):
                payload = {
                    "regime_ids": [regime.id for regime in manifest.regimes],
                    "regimes": [
                        {
                            "id": regime.id,
                            "workload_overrides": regime.workload_overrides,
                            "field_diffs_from_baseline": regime.field_diffs_from_baseline,
                        }
                        for regime in manifest.regimes
                    ],
                    "horizons": list(manifest.horizons),
                    "seeds": list(manifest.seeds),
                    "experiment_ids": list(manifest.experiment_ids),
                    "common_eligible_window_protocol": manifest.common_windows,
                    "thesis_candidate_cells": list(manifest.thesis_candidate_cells),
                    "thesis_gate_thresholds": manifest.thesis_gate_thresholds,
                    "policy_display_names": [
                        policy.display_name for policy in manifest.policies
                    ],
                }
            else:
                payload = {
                    "variant_ids": [variant.id for variant in manifest.variants],
                    "seeds": list(manifest.seeds),
                    "experiment_ids": list(manifest.experiment_ids),
                    "variants": [
                        {
                            "id": variant.id,
                            "workload_overrides": variant.workload_overrides,
                            "capacity_fraction": variant.capacity_fraction,
                            "promotion_saved_read_equivalents": (
                                variant.promotion_saved_read_equivalents
                            ),
                        }
                        for variant in manifest.variants
                    ],
                    "policy_display_names": [
                        policy.display_name for policy in manifest.policies
                    ],
                }
            print(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.output_dir is None:
            parser.error("--output-dir is required unless --print-manifest is used")
        execution = run_experiments(
            arguments.manifest,
            arguments.output_dir,
            experiment_id=arguments.experiment_id,
            resume=arguments.resume,
            progress=print,
        )
    except (ManifestError, ExperimentRunError, OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Experiment status: completed={execution.completed_count}, "
        f"failed={execution.failed_count}, reused={len(execution.reused_experiment_ids)}"
    )
    return int(execution.failed_count > 0)


if __name__ == "__main__":
    raise SystemExit(main())
