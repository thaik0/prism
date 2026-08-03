from __future__ import annotations

import math

import pytest

from prism.predictor import PredictorConfig, PredictorConfigError


def valid_config() -> dict[str, object]:
    return {
        "fit_seed": 7,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "activation_max_iter": 1000,
        "activation_tolerance": 1e-6,
        "intensity_ridge_alpha": 1.0,
        "calibration_bins": 10,
    }


def test_chronological_split_boundaries_and_target_assignment() -> None:
    config = PredictorConfig.from_dict(valid_config())
    split = config.split_boundaries(1000)

    assert split.train_end == 600
    assert split.validation_end == 800
    assert split.split_code_for_target(599) == 0
    assert split.split_code_for_target(600) == 1
    assert split.split_code_for_target(799) == 1
    assert split.split_code_for_target(800) == 2
    assert split.to_dict()["training_target_windows"] == [3, 600]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.pop("fit_seed"), "missing.*fit_seed"),
        (lambda raw: raw.update({"model": "tree"}), "unknown.*model"),
        (lambda raw: raw.update({"fit_seed": True}), "fit_seed"),
        (lambda raw: raw.update({"train_fraction": 0.0}), "train_fraction"),
        (lambda raw: raw.update({"validation_fraction": 1.0}), "validation_fraction"),
        (
            lambda raw: raw.update(
                {"train_fraction": 0.8, "validation_fraction": 0.2}
            ),
            "plus validation_fraction",
        ),
        (lambda raw: raw.update({"activation_max_iter": 0}), "activation_max_iter"),
        (lambda raw: raw.update({"activation_tolerance": math.inf}), "finite"),
        (lambda raw: raw.update({"intensity_ridge_alpha": -1.0}), "ridge_alpha"),
        (lambda raw: raw.update({"calibration_bins": 1}), "calibration_bins"),
    ],
)
def test_invalid_predictor_configuration_is_rejected(mutation, message: str) -> None:
    raw = valid_config()
    mutation(raw)

    with pytest.raises(PredictorConfigError, match=message):
        PredictorConfig.from_dict(raw)


def test_too_short_source_splits_are_rejected() -> None:
    config = PredictorConfig.from_dict(valid_config())

    with pytest.raises(PredictorConfigError, match="lagged training"):
        config.split_boundaries(6)
