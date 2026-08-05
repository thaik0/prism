from __future__ import annotations

import math

import pytest

from prism.structure.config import (
    NMF_ALGORITHM,
    NMF_BETA_LOSS,
    NMF_INIT,
    NMF_SOLVER,
    StructureLearnerConfig,
    StructureLearnerConfigError,
)


def valid_config() -> dict[str, object]:
    return {
        "n_components": 2,
        "fit_seed": 17,
        "max_iter": 500,
        "tolerance": 1e-4,
    }


def test_structure_config_resolves_fixed_nmf_choices() -> None:
    config = StructureLearnerConfig.from_dict(valid_config())

    assert config.to_resolved_dict() == {
        **valid_config(),
        "algorithm": NMF_ALGORITHM,
        "init": NMF_INIT,
        "solver": NMF_SOLVER,
        "beta_loss": NMF_BETA_LOSS,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.pop("fit_seed"), "missing.*fit_seed"),
        (lambda raw: raw.update({"solver": "mu"}), "unknown.*solver"),
        (lambda raw: raw.update({"n_components": True}), "n_components"),
        (lambda raw: raw.update({"n_components": 0}), "n_components"),
        (lambda raw: raw.update({"fit_seed": 1.5}), "fit_seed"),
        (lambda raw: raw.update({"fit_seed": -1}), "fit_seed"),
        (lambda raw: raw.update({"max_iter": 0}), "max_iter"),
        (lambda raw: raw.update({"tolerance": True}), "tolerance"),
        (lambda raw: raw.update({"tolerance": 0.0}), "tolerance"),
        (lambda raw: raw.update({"tolerance": math.inf}), "tolerance"),
    ],
)
def test_invalid_structure_config_is_rejected(mutation, message: str) -> None:
    raw = valid_config()
    mutation(raw)

    with pytest.raises(StructureLearnerConfigError, match=message):
        StructureLearnerConfig.from_dict(raw)


def test_component_dimension_and_representative_k_validation() -> None:
    config = StructureLearnerConfig.from_dict(valid_config())
    config.validate_dimensions(2, 4)
    config.validate_representative_factor_count(2)

    with pytest.raises(StructureLearnerConfigError, match="must not exceed"):
        config.validate_dimensions(1, 4)
    with pytest.raises(StructureLearnerConfigError, match="num_working_sets"):
        config.validate_representative_factor_count(3)
