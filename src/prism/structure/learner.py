"""Exactly one deterministic nonnegative matrix-factorization baseline."""

from __future__ import annotations

from dataclasses import dataclass
import warnings as python_warnings

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from prism.structure.config import (
    NMF_BETA_LOSS,
    NMF_INIT,
    NMF_SOLVER,
    StructureLearnerConfig,
)
from prism.structure.demand import DemandMatrix


class StructureFitError(ValueError):
    """Raised when NMF output violates the Milestone 2 factor contract."""


@dataclass(frozen=True)
class LearnedStructure:
    """Canonical normalized fuzzy structure and fit diagnostics."""

    activation_matrix: np.ndarray
    membership_matrix: np.ndarray
    factor_ids: np.ndarray
    window_ids: np.ndarray
    record_ids: np.ndarray
    converged: bool
    iteration_count: int
    sklearn_reconstruction_error: float
    convergence_warnings: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "activation_matrix",
            "membership_matrix",
            "factor_ids",
            "window_ids",
            "record_ids",
        ):
            value = np.array(getattr(self, field_name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)


def fit_structure(
    demand_matrix: DemandMatrix,
    config: StructureLearnerConfig,
) -> LearnedStructure:
    """Fit one fixed NMF model and return its normalized factorization."""

    if not isinstance(demand_matrix, DemandMatrix):
        raise TypeError("demand_matrix must be a DemandMatrix")
    if not isinstance(config, StructureLearnerConfig):
        raise TypeError("config must be a StructureLearnerConfig")
    config.validate_dimensions(*demand_matrix.X.shape)

    X = np.asarray(demand_matrix.X, dtype=np.float64, order="C")
    model = NMF(
        n_components=config.n_components,
        init=NMF_INIT,
        solver=NMF_SOLVER,
        beta_loss=NMF_BETA_LOSS,
        max_iter=config.max_iter,
        tol=config.tolerance,
        random_state=config.fit_seed,
    )
    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always", ConvergenceWarning)
        activation_raw = model.fit_transform(X)
    membership_raw = np.asarray(model.components_, dtype=np.float64)

    convergence_messages = tuple(
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    )
    reached_iteration_limit = int(model.n_iter_) >= config.max_iter
    if reached_iteration_limit and not convergence_messages:
        convergence_messages = (
            f"NMF reached the configured iteration limit ({config.max_iter})",
        )
    converged = not convergence_messages and not reached_iteration_limit

    activation, membership, normalization_warnings = normalize_factors(
        activation_raw, membership_raw
    )
    return LearnedStructure(
        activation_matrix=activation,
        membership_matrix=membership,
        factor_ids=np.arange(config.n_components, dtype=np.int64),
        window_ids=demand_matrix.window_ids,
        record_ids=demand_matrix.record_ids,
        converged=converged,
        iteration_count=int(model.n_iter_),
        sklearn_reconstruction_error=float(model.reconstruction_err_),
        convergence_warnings=convergence_messages,
        warnings=normalization_warnings,
    )


def normalize_factors(
    activation_raw: np.ndarray, membership_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Normalize membership rows and reciprocally rescale activation columns."""

    activation = np.array(activation_raw, dtype=np.float64, copy=True)
    membership = np.array(membership_raw, dtype=np.float64, copy=True)
    if activation.ndim != 2 or membership.ndim != 2:
        raise StructureFitError("activation and membership must be matrices")
    if activation.shape[1] != membership.shape[0]:
        raise StructureFitError("activation and membership factor counts must agree")
    if not np.all(np.isfinite(activation)) or not np.all(np.isfinite(membership)):
        raise StructureFitError("learned factors must contain only finite values")
    if np.any(activation < 0.0) or np.any(membership < 0.0):
        raise StructureFitError("learned factors must be nonnegative")

    warnings: list[str] = []
    scales = membership.sum(axis=1, dtype=np.float64)
    for factor_id, scale in enumerate(scales):
        if scale == 0.0:
            warnings.append(
                f"learned factor {factor_id} has zero-sum membership and was left unnormalized"
            )
            continue
        membership[factor_id, :] /= scale
        activation[:, factor_id] *= scale
    return activation, membership, tuple(warnings)
