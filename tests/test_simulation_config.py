from __future__ import annotations

import json
import math

import pytest

from prism.simulation import SimulationConfig, SimulationConfigError
from tests.conftest import REPOSITORY_ROOT


REPRESENTATIVE = REPOSITORY_ROOT / "configs" / "milestone4_simulation.json"


def _raw() -> dict[str, int | float]:
    return {
        "fast_capacity_bytes": 100,
        "fast_read_cost": 1.0,
        "slow_read_cost": 10.0,
        "promotion_cost_per_byte": 0.01,
    }


def test_representative_configuration_has_frozen_static_values() -> None:
    config = SimulationConfig.from_json(REPRESENTATIVE)

    assert config.fast_capacity_bytes == 154728
    assert config.fast_read_cost == 1.0
    assert config.slow_read_cost == 10.0
    assert config.promotion_cost_per_byte == 0.0017976630380505342
    assert config.promotion_cost_per_byte * 10013 == pytest.approx(18.0)
    config.validate_record_sizes([1024, 10013, 16384])


def test_missing_unknown_duplicate_and_invalid_fields_are_rejected(tmp_path) -> None:
    raw = _raw()
    del raw["fast_read_cost"]
    with pytest.raises(SimulationConfigError, match="missing.*fast_read_cost"):
        SimulationConfig.from_dict(raw)

    raw = _raw()
    raw["extra"] = 1
    with pytest.raises(SimulationConfigError, match="unknown.*extra"):
        SimulationConfig.from_dict(raw)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"fast_capacity_bytes": 10, "fast_capacity_bytes": 20, '
        '"fast_read_cost": 1, "slow_read_cost": 2, '
        '"promotion_cost_per_byte": 0}\n',
        encoding="utf-8",
    )
    with pytest.raises(SimulationConfigError, match="duplicate"):
        SimulationConfig.from_json(duplicate)


@pytest.mark.parametrize(
    "updates",
    [
        {"fast_capacity_bytes": True},
        {"fast_capacity_bytes": 0},
        {"fast_capacity_bytes": 1.5},
        {"fast_read_cost": True},
        {"fast_read_cost": -1.0},
        {"slow_read_cost": math.inf},
        {"promotion_cost_per_byte": math.nan},
        {"promotion_cost_per_byte": -0.1},
        {"slow_read_cost": 1.0},
    ],
)
def test_invalid_configuration_values_are_rejected(updates) -> None:
    raw = _raw()
    raw.update(updates)
    with pytest.raises(SimulationConfigError):
        SimulationConfig.from_dict(raw)


def test_capacity_must_hold_at_least_one_valid_record() -> None:
    config = SimulationConfig.from_dict(_raw())
    with pytest.raises(SimulationConfigError, match="cannot hold any"):
        config.validate_record_sizes([101, 200])
    with pytest.raises(SimulationConfigError, match="positive integers"):
        config.validate_record_sizes([True])
