"""Demand windows and slow structural recovery for Prism Milestone 2."""

from prism.structure.config import StructureLearnerConfig, StructureLearnerConfigError
from prism.structure.demand import DemandMatrix, DemandMatrixError, build_demand_matrix
from prism.structure.evaluate import (
    RecoveryEvaluation,
    RecoveryEvaluationError,
    evaluate_recovery,
)
from prism.structure.learner import (
    LearnedStructure,
    StructureFitError,
    fit_structure,
    normalize_factors,
)
from prism.structure.persistence import (
    StructureOutputDirectoryError,
    StructureRunResult,
    run_structure_recovery,
)

__all__ = [
    "DemandMatrix",
    "DemandMatrixError",
    "LearnedStructure",
    "RecoveryEvaluation",
    "RecoveryEvaluationError",
    "StructureFitError",
    "StructureLearnerConfig",
    "StructureLearnerConfigError",
    "StructureOutputDirectoryError",
    "StructureRunResult",
    "build_demand_matrix",
    "evaluate_recovery",
    "fit_structure",
    "normalize_factors",
    "run_structure_recovery",
]
