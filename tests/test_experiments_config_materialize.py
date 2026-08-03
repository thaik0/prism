from __future__ import annotations

import json
import math

import pytest

from prism.experiments import (
    ManifestError,
    load_manifest,
    resolve_simulation_config,
    resolve_workload_config,
)
from tests.conftest import REPOSITORY_ROOT


MANIFEST = REPOSITORY_ROOT / "configs" / "milestone5_experiments.json"


def test_frozen_manifest_has_exact_order_ids_policies_and_transformations() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.seeds == (1729, 2718, 31415)
    assert len(manifest.variants) == 12
    assert len(manifest.experiment_ids) == 36
    assert manifest.experiment_ids[0] == "baseline__seed_1729"
    assert manifest.experiment_ids[-1] == "context_weak__seed_31415"
    assert len(manifest.policies) == 11
    assert manifest.policies[3].display_name == "Predictive Greedy (Prism)"
    assert manifest.variant("noise_low").workload_overrides == {
        "noise_access_weight": 0.175
    }
    assert manifest.variant("burst_short").workload_overrides == {
        "burst_duration_min_windows": 1,
        "burst_duration_max_windows": 2,
    }


def test_manifest_duplicate_missing_family_unknown_and_invalid_values_are_rejected(
    tmp_path,
) -> None:
    raw = json.loads(MANIFEST.read_text())
    raw["variants"][1]["id"] = "baseline"
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(duplicate)

    raw = json.loads(MANIFEST.read_text())
    raw["variants"][10]["workload_overrides"]["precursor_probability_scale"] = 1.1
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(invalid)

    raw = json.loads(MANIFEST.read_text())
    raw["variants"][6]["workload_overrides"] = {"burst_access_weight": 9.0}
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="unknown workload"):
        load_manifest(unknown)


def test_materialization_changes_only_seed_and_exact_variant_fields() -> None:
    manifest = load_manifest(MANIFEST)
    weak = resolve_workload_config(manifest, manifest.variant("context_weak"), 2718)

    assert weak.seed == 2718
    assert weak.precursor_probability_scale == pytest.approx(0.4125)
    assert weak.spontaneous_activation_probability == 0.16
    assert weak.noise_access_weight == 0.35
    assert weak.burst_duration_min_windows == 2
    assert weak.burst_duration_max_windows == 5


def test_trace_specific_capacity_and_median_promotion_cost_are_exact() -> None:
    manifest = load_manifest(MANIFEST)
    sizes = [1, 20, 39]
    simulation = resolve_simulation_config(manifest.variant("capacity_10"), sizes)
    assert simulation.fast_capacity_bytes == math.floor(0.10 * sum(sizes))
    assert simulation.promotion_cost_per_byte == pytest.approx(18.0 / 20.0)

    free = resolve_simulation_config(manifest.variant("promotion_0"), sizes)
    assert free.promotion_cost_per_byte == 0.0
