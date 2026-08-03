"""Controlled synthetic workloads for Prism Milestone 1."""

from prism.workload.config import WorkloadConfig, WorkloadConfigError
from prism.workload.generator import (
    OutputDirectoryError,
    generate_workload,
    persist_workload,
)
from prism.workload.models import ObservableEvent, WorkloadResult, WorkloadSummary

__all__ = [
    "ObservableEvent",
    "OutputDirectoryError",
    "WorkloadConfig",
    "WorkloadConfigError",
    "WorkloadResult",
    "WorkloadSummary",
    "generate_workload",
    "persist_workload",
]
