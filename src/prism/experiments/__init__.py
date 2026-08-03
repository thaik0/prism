"""Reproducible multi-configuration evaluation for Prism Milestone 5."""

from prism.experiments.config import (
    ExperimentManifest,
    ManifestError,
    PolicyDefinition,
    VariantDefinition,
    load_manifest,
)
from prism.experiments.aggregate import (
    aggregate_experiment_root,
    descriptive_statistics,
    fixed_trajectory_crossover,
    write_aggregate_outputs,
)
from prism.experiments.materialize import (
    MaterializedRun,
    materialize_run,
    resolve_simulation_config,
    resolve_workload_config,
)
from prism.experiments.runner import (
    ExperimentExecution,
    ExperimentRunError,
    run_experiments,
)

__all__ = [
    "ExperimentManifest",
    "ExperimentExecution",
    "ExperimentRunError",
    "ManifestError",
    "MaterializedRun",
    "PolicyDefinition",
    "VariantDefinition",
    "aggregate_experiment_root",
    "descriptive_statistics",
    "fixed_trajectory_crossover",
    "load_manifest",
    "materialize_run",
    "resolve_simulation_config",
    "resolve_workload_config",
    "run_experiments",
    "write_aggregate_outputs",
]
