from __future__ import annotations

import pytest

from prism.experiments.aggregate import (
    _hypothesis_status,
    _paired_comparisons,
    descriptive_statistics,
    fixed_trajectory_crossover,
)
from prism.experiments.config import load_manifest
from tests.conftest import REPOSITORY_ROOT


MANIFEST = REPOSITORY_ROOT / "configs" / "milestone5_experiments.json"


def test_descriptive_statistics_use_sample_standard_deviation_and_retain_extrema() -> None:
    result = descriptive_statistics([1.0, 2.0, 3.0])
    assert result == {
        "count": 3,
        "mean": 2.0,
        "median": 2.0,
        "sample_standard_deviation_ddof_1": 1.0,
        "minimum": 1.0,
        "maximum": 3.0,
    }
    assert descriptive_statistics([4.0])["sample_standard_deviation_ddof_1"] is None


def test_paired_comparisons_match_only_identical_variant_and_seed() -> None:
    manifest = load_manifest(MANIFEST)
    rows = {}
    for variant in manifest.variants:
        for seed in manifest.seeds:
            policies = {
                policy.id: {"total_combined_cost": 100.0}
                for policy in manifest.policies
            }
            policies["predictive_greedy"]["total_combined_cost"] = float(seed)
            policies["validation_final_frozen"]["total_combined_cost"] = float(seed + 1)
            rows[(variant.id, seed)] = {"policy_results": policies}

    comparisons = _paired_comparisons(rows, manifest)
    frozen = next(
        item
        for item in comparisons
        if item["right_policy_id"] == "validation_final_frozen"
    )
    baseline = frozen["variants"][0]
    assert baseline["paired_differences"] == [
        {"seed": 1729, "difference": -1.0},
        {"seed": 2718, "difference": -1.0},
        {"seed": 31415, "difference": -1.0},
    ]
    assert baseline["left_costs_less_count"] == 3
    assert baseline["equal_within_tolerance_count"] == 0
    assert baseline["left_costs_more_count"] == 0


def test_fixed_trajectory_break_even_handles_defined_zero_and_meaningless_cases() -> None:
    defined = fixed_trajectory_crossover(80.0, 20, 100.0, 10)
    assert defined["promotion_cost_per_byte_crossover"] == 2.0
    assert defined["meaningful"]

    zero = fixed_trajectory_crossover(80.0, 10, 100.0, 10)
    assert zero["promotion_cost_per_byte_crossover"] is None
    assert "equal bytes" in zero["explanation"]

    negative = fixed_trajectory_crossover(100.0, 20, 80.0, 10)
    assert negative["promotion_cost_per_byte_crossover"] is None
    assert not negative["meaningful"]


def test_hypothesis_summary_rules_are_deterministic() -> None:
    variants = ["baseline", "context_strong"]
    assert _hypothesis_status(
        {"baseline": [-1, 1, -1], "context_strong": [-1, -2, -3]},
        variants,
        True,
    ) == "supported"
    assert _hypothesis_status(
        {"baseline": [1, 1, -1], "context_strong": [1, 1, 1]},
        variants,
        True,
    ) == "not_supported"
    assert _hypothesis_status(
        {"baseline": [-1, -1, 1], "context_strong": [-1, 1, 1]},
        variants,
        True,
    ) == "mixed"
    assert _hypothesis_status(
        {"baseline": [-1], "context_strong": []}, variants, True
    ) == "insufficient_data"
