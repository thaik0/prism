"""Strict configuration for the Milestone 2 slow structure learner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping


NMF_ALGORITHM = "sklearn.decomposition.NMF"
NMF_INIT = "nndsvda"
NMF_SOLVER = "cd"
NMF_BETA_LOSS = "frobenius"
MAX_RANDOM_SEED = 2**32 - 1


class StructureLearnerConfigError(ValueError):
    """Raised when the slow-learner configuration is malformed."""


@dataclass(frozen=True)
class StructureLearnerConfig:
    """Fully specified configurable inputs for exactly one NMF fit."""

    n_components: int
    fit_seed: int
    max_iter: int
    tolerance: float

    def __post_init__(self) -> None:
        _require_integer("n_components", self.n_components, minimum=1)
        _require_integer(
            "fit_seed", self.fit_seed, minimum=0, maximum=MAX_RANDOM_SEED
        )
        _require_integer("max_iter", self.max_iter, minimum=1)
        if isinstance(self.tolerance, bool) or not isinstance(
            self.tolerance, (int, float)
        ):
            raise StructureLearnerConfigError(
                "tolerance must be a finite positive number"
            )
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise StructureLearnerConfigError(
                "tolerance must be a finite positive number"
            )
        object.__setattr__(self, "tolerance", tolerance)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StructureLearnerConfig:
        if not isinstance(raw, Mapping):
            raise StructureLearnerConfigError(
                "learner configuration root must be a JSON object"
            )
        expected = {field.name for field in fields(cls)}
        supplied = set(raw)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing:
            raise StructureLearnerConfigError(
                "missing required learner configuration fields: "
                + ", ".join(missing)
            )
        if unknown:
            raise StructureLearnerConfigError(
                "unknown learner configuration fields: " + ", ".join(unknown)
            )
        return cls(**dict(raw))

    @classmethod
    def from_json(cls, path: str | Path) -> StructureLearnerConfig:
        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise StructureLearnerConfigError(
                f"invalid JSON in {config_path}: {error.msg}"
            ) from error
        return cls.from_dict(raw)

    def validate_dimensions(self, num_windows: int, num_records: int) -> None:
        """Reject a factor count that cannot fit the source matrix."""

        limit = min(num_windows, num_records)
        if self.n_components > limit:
            raise StructureLearnerConfigError(
                "n_components must not exceed min(num_windows, num_records) "
                f"({limit}); got {self.n_components}"
            )

    def validate_representative_factor_count(
        self, planted_factor_count: int
    ) -> None:
        """Enforce the controlled representative experiment's supplied K."""

        if self.n_components != planted_factor_count:
            raise StructureLearnerConfigError(
                "representative n_components must equal source num_working_sets "
                f"({planted_factor_count}); got {self.n_components}"
            )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def to_resolved_dict(self) -> dict[str, int | float | str]:
        return {
            **self.to_dict(),
            "algorithm": NMF_ALGORITHM,
            "init": NMF_INIT,
            "solver": NMF_SOLVER,
            "beta_loss": NMF_BETA_LOSS,
        }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructureLearnerConfigError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _require_integer(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StructureLearnerConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise StructureLearnerConfigError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise StructureLearnerConfigError(f"{name} must be at most {maximum}")
