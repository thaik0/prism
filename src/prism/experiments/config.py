"""Strict frozen experiment-manifest validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from prism.workload import WorkloadConfig


EXPECTED_SEEDS = (1729, 2718, 31415)
ACTIONABILITY_REGIMES = ("baseline", "sparse", "very_sparse")
ACTIONABILITY_HORIZONS = (1, 2, 4)
ACTIONABILITY_CANDIDATE_CELLS = (
    "sparse__h2",
    "sparse__h4",
    "very_sparse__h2",
    "very_sparse__h4",
)
EXPECTED_VARIANTS = (
    "baseline",
    "capacity_10",
    "capacity_40",
    "promotion_0",
    "promotion_1",
    "promotion_4",
    "noise_low",
    "noise_high",
    "burst_short",
    "burst_long",
    "context_strong",
    "context_weak",
)
EXPECTED_POLICIES = (
    ("lru", "LRU"),
    ("lfu", "LFU"),
    ("recent_demand_greedy", "Recent-Demand Greedy"),
    ("predictive_greedy", "Predictive Greedy (Prism)"),
    ("training_popularity_static", "Training-Popularity Static (Prism)"),
    ("validation_final_frozen", "Validation-Final Frozen (Prism)"),
    ("recent_state_only", "Recent-State-Only (Prism ablation)"),
    (
        "activation_intensity_only",
        "Activation/Intensity-Only (Prism ablation)",
    ),
    (
        "residual_baseline_only",
        "Residual-Baseline-Only (Prism ablation)",
    ),
    ("oracle_greedy", "Oracle Greedy"),
    ("oracle_exact", "Oracle Exact"),
)
EXPECTED_FAMILIES = {
    "baseline": {"baseline"},
    "capacity": {"capacity_10", "capacity_40"},
    "promotion": {"promotion_0", "promotion_1", "promotion_4"},
    "noise": {"noise_low", "noise_high"},
    "duration": {"burst_short", "burst_long"},
    "context": {"context_strong", "context_weak"},
}
MANIFEST_FIELDS = {
    "schema_version",
    "base_workload_config",
    "structure_config",
    "predictor_config",
    "base_simulation_settings",
    "seeds",
    "variant_order",
    "variants",
    "policy_order",
    "experiment_id_format",
    "expected_unique_variant_count",
    "expected_total_run_count",
}
VARIANT_FIELDS = {
    "id",
    "family",
    "workload_overrides",
    "capacity_fraction",
    "promotion_saved_read_equivalents",
}


class ManifestError(ValueError):
    """Raised when the frozen Milestone 5 design is malformed."""


@dataclass(frozen=True)
class PolicyDefinition:
    id: str
    display_name: str


@dataclass(frozen=True)
class VariantDefinition:
    id: str
    family: str
    workload_overrides: dict[str, int | float]
    capacity_fraction: float
    promotion_saved_read_equivalents: float


@dataclass(frozen=True)
class ExperimentManifest:
    path: Path
    sha256: str
    raw: dict[str, Any]
    base_workload_config: Path
    structure_config: Path
    predictor_config: Path
    seeds: tuple[int, ...]
    variants: tuple[VariantDefinition, ...]
    policies: tuple[PolicyDefinition, ...]

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{variant.id}__seed_{seed}"
            for variant in self.variants
            for seed in self.seeds
        )

    def variant(self, variant_id: str) -> VariantDefinition:
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise ManifestError(f"unknown variant ID: {variant_id}")


@dataclass(frozen=True)
class RegimeDefinition:
    id: str
    workload_overrides: dict[str, int | float]
    field_diffs_from_baseline: dict[str, dict[str, int | float]]


@dataclass(frozen=True)
class ActionabilityManifest:
    """Frozen Milestone 5.5 regime/horizon experiment design."""

    path: Path
    sha256: str
    raw: dict[str, Any]
    base_workload_config: Path
    structure_config: Path
    predictor_config: Path
    seeds: tuple[int, ...]
    regimes: tuple[RegimeDefinition, ...]
    horizons: tuple[int, ...]
    policies: tuple[PolicyDefinition, ...]
    common_windows: dict[str, Any]
    thesis_candidate_cells: tuple[str, ...]
    thesis_gate_thresholds: dict[str, Any]

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{regime.id}__h{horizon}__seed_{seed}"
            for regime in self.regimes
            for horizon in self.horizons
            for seed in self.seeds
        )

    def regime(self, regime_id: str) -> RegimeDefinition:
        for regime in self.regimes:
            if regime.id == regime_id:
                return regime
        raise ManifestError(f"unknown regime ID: {regime_id}")


def load_manifest(path: str | Path) -> ExperimentManifest | ActionabilityManifest:
    """Load and validate the exact committed Milestone 5 experiment design."""

    manifest_path = Path(path)
    raw_bytes = manifest_path.read_bytes()
    try:
        raw = json.loads(raw_bytes, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ManifestError(f"invalid manifest JSON: {error.msg}") from error
    if isinstance(raw, dict) and raw.get("schema_version") == 2:
        return _load_actionability_manifest(manifest_path, raw_bytes, raw)
    if not isinstance(raw, dict) or set(raw) != MANIFEST_FIELDS:
        raise ManifestError("manifest fields do not match the schema")
    if raw["schema_version"] != 1:
        raise ManifestError("manifest schema_version must equal 1")
    if raw["experiment_id_format"] != "<variant_id>__seed_<seed>":
        raise ManifestError("experiment_id_format is invalid")
    if raw["expected_unique_variant_count"] != 12:
        raise ManifestError("expected_unique_variant_count must equal 12")
    if raw["expected_total_run_count"] != 36:
        raise ManifestError("expected_total_run_count must equal 36")
    seeds = _integer_tuple("seeds", raw["seeds"])
    if seeds != EXPECTED_SEEDS:
        raise ManifestError(f"seeds must equal {list(EXPECTED_SEEDS)} in order")
    if tuple(raw["variant_order"]) != EXPECTED_VARIANTS:
        raise ManifestError("variant_order must contain the exact twelve variants")

    root = manifest_path.resolve().parent.parent
    base_path = _config_path(root, raw["base_workload_config"], "base_workload_config")
    structure_path = _config_path(root, raw["structure_config"], "structure_config")
    predictor_path = _config_path(root, raw["predictor_config"], "predictor_config")
    base = WorkloadConfig.from_json(base_path)
    _validate_base_settings(raw["base_simulation_settings"])

    raw_variants = raw["variants"]
    if not isinstance(raw_variants, list) or len(raw_variants) != 12:
        raise ManifestError("variants must contain exactly twelve entries")
    variants = tuple(_variant(item) for item in raw_variants)
    ids = tuple(item.id for item in variants)
    if len(set(ids)) != len(ids):
        raise ManifestError("variant IDs must be unique")
    if ids != EXPECTED_VARIANTS:
        raise ManifestError("variants must follow exact variant_order")
    for family, expected_ids in EXPECTED_FAMILIES.items():
        actual = {item.id for item in variants if item.family == family}
        if actual != expected_ids:
            raise ManifestError(f"variant family {family} is missing or invalid")
    _validate_exact_transformations(base, variants)

    policies = _policies(raw["policy_order"])
    if tuple((item.id, item.display_name) for item in policies) != EXPECTED_POLICIES:
        raise ManifestError("policy_order must contain the exact eleven policies")
    return ExperimentManifest(
        path=manifest_path.resolve(),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw=raw,
        base_workload_config=base_path,
        structure_config=structure_path,
        predictor_config=predictor_path,
        seeds=seeds,
        variants=variants,
        policies=policies,
    )


def _load_actionability_manifest(
    manifest_path: Path, raw_bytes: bytes, raw: dict[str, Any]
) -> ActionabilityManifest:
    expected_fields = {
        "schema_version",
        "base_workload_config",
        "structure_config",
        "predictor_config",
        "base_simulation_settings",
        "regime_order",
        "regimes",
        "horizons",
        "seeds",
        "common_eligible_window_protocol",
        "policy_order",
        "thesis_candidate_cells",
        "thesis_gate_thresholds",
        "experiment_id_format",
        "expected_total_run_count",
    }
    if set(raw) != expected_fields:
        raise ManifestError("Milestone 5.5 manifest fields do not match the schema")
    if raw["experiment_id_format"] != "<regime>__h<horizon>__seed_<seed>":
        raise ManifestError("Milestone 5.5 experiment_id_format is invalid")
    if raw["expected_total_run_count"] != 27:
        raise ManifestError("Milestone 5.5 expected_total_run_count must equal 27")
    seeds = _integer_tuple("seeds", raw["seeds"])
    if seeds != EXPECTED_SEEDS:
        raise ManifestError(f"seeds must equal {list(EXPECTED_SEEDS)} in order")
    horizons = _integer_tuple("horizons", raw["horizons"])
    if horizons != ACTIONABILITY_HORIZONS:
        raise ManifestError("horizons must equal [1, 2, 4] in order")
    if tuple(raw["regime_order"]) != ACTIONABILITY_REGIMES:
        raise ManifestError("regime_order must equal baseline, sparse, very_sparse")

    root = manifest_path.resolve().parent.parent
    base_path = _config_path(root, raw["base_workload_config"], "base_workload_config")
    structure_path = _config_path(root, raw["structure_config"], "structure_config")
    predictor_path = _config_path(root, raw["predictor_config"], "predictor_config")
    base = WorkloadConfig.from_json(base_path)
    _validate_base_settings(raw["base_simulation_settings"])
    regimes = _actionability_regimes(raw["regimes"], base)
    policies = _policies(raw["policy_order"])
    if tuple((item.id, item.display_name) for item in policies) != EXPECTED_POLICIES:
        raise ManifestError("policy_order must contain the exact eleven policies")
    common_windows = raw["common_eligible_window_protocol"]
    expected_windows = {
        "h_max": 4,
        "train_end": 600,
        "validation_start": 600,
        "validation_end": 800,
        "validation_evaluation_windows": [600, 797],
        "validation_carry_only_windows": [797, 800],
        "test_start": 800,
        "trace_end": 1000,
        "test_evaluation_windows": [800, 997],
        "final_excluded_tail_windows": [997, 1000],
    }
    if common_windows != expected_windows:
        raise ManifestError("common eligible-window protocol is not the frozen protocol")
    candidate_cells = tuple(raw["thesis_candidate_cells"])
    if candidate_cells != ACTIONABILITY_CANDIDATE_CELLS:
        raise ManifestError("thesis candidate cells are not the precommitted four")
    thresholds = raw["thesis_gate_thresholds"]
    expected_thresholds = {
        "gate_a_mean_difference_fraction_minimum": 0.10,
        "gate_a_seed_difference_fraction_minimum": 0.10,
        "gate_a_minimum_passing_seed_count": 2,
        "gate_b_minimum_seed_win_count": 2,
        "gate_c_minimum_seed_win_count": 2,
        "gate_c_transition_coverage_absolute_improvement_minimum": 0.05,
        "gate_d_minimum_pre_demand_promotions": 10,
        "gate_d_minimum_repayment_fraction": 0.50,
        "gate_d_aggregate_net_value_strictly_positive": True,
        "numerical_tolerance": 1e-9,
    }
    if thresholds != expected_thresholds:
        raise ManifestError("thesis gate thresholds are not the frozen thresholds")
    return ActionabilityManifest(
        path=manifest_path.resolve(),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw=raw,
        base_workload_config=base_path,
        structure_config=structure_path,
        predictor_config=predictor_path,
        seeds=seeds,
        regimes=regimes,
        horizons=horizons,
        policies=policies,
        common_windows=dict(common_windows),
        thesis_candidate_cells=candidate_cells,
        thesis_gate_thresholds=dict(thresholds),
    )


def _actionability_regimes(
    raw: Any, base: WorkloadConfig
) -> tuple[RegimeDefinition, ...]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ManifestError("regimes must contain exactly three entries")
    expected_overrides: dict[str, dict[str, int | float]] = {
        "baseline": {
            "precursor_probability_scale": base.precursor_probability_scale,
            "spontaneous_activation_probability": base.spontaneous_activation_probability,
            "post_burst_cooldown_windows": 0,
        },
        "sparse": {
            "precursor_probability_scale": 0.5 * base.precursor_probability_scale,
            "spontaneous_activation_probability": 0.5 * base.spontaneous_activation_probability,
            "post_burst_cooldown_windows": base.burst_duration_max_windows,
        },
        "very_sparse": {
            "precursor_probability_scale": 0.25 * base.precursor_probability_scale,
            "spontaneous_activation_probability": 0.25 * base.spontaneous_activation_probability,
            "post_burst_cooldown_windows": 2 * base.burst_duration_max_windows,
        },
    }
    regimes: list[RegimeDefinition] = []
    for expected_id, item in zip(ACTIONABILITY_REGIMES, raw, strict=True):
        if not isinstance(item, dict) or set(item) != {
            "id", "workload_overrides", "field_diffs_from_baseline"
        }:
            raise ManifestError("regime entries have invalid fields")
        if item["id"] != expected_id or item["workload_overrides"] != expected_overrides[expected_id]:
            raise ManifestError(f"regime {expected_id} has incorrect frozen values")
        baseline_values = expected_overrides["baseline"]
        diffs = {
            name: {"baseline": baseline_values[name], "resolved": value}
            for name, value in expected_overrides[expected_id].items()
            if value != baseline_values[name]
        }
        if item["field_diffs_from_baseline"] != diffs:
            raise ManifestError(f"regime {expected_id} has incorrect field diffs")
        resolved = base.to_resolved_dict()
        resolved.update(item["workload_overrides"])
        WorkloadConfig.from_dict(resolved)
        regimes.append(RegimeDefinition(expected_id, dict(item["workload_overrides"]), diffs))
    return tuple(regimes)


def _variant(raw: Any) -> VariantDefinition:
    if not isinstance(raw, dict) or set(raw) != VARIANT_FIELDS:
        raise ManifestError("each variant must contain the exact variant fields")
    variant_id = _nonempty_string("variant.id", raw["id"])
    family = _nonempty_string("variant.family", raw["family"])
    overrides = raw["workload_overrides"]
    if not isinstance(overrides, dict):
        raise ManifestError("workload_overrides must be an object")
    allowed = {
        "noise_access_weight",
        "burst_duration_min_windows",
        "burst_duration_max_windows",
        "precursor_probability_scale",
        "spontaneous_activation_probability",
    }
    if not set(overrides) <= allowed:
        raise ManifestError("unknown workload transformation field")
    capacity = _number("capacity_fraction", raw["capacity_fraction"], minimum=0.0)
    if not 0.0 < capacity <= 1.0:
        raise ManifestError("capacity_fraction must be in (0, 1]")
    promotion = _number(
        "promotion_saved_read_equivalents",
        raw["promotion_saved_read_equivalents"],
        minimum=0.0,
    )
    return VariantDefinition(variant_id, family, dict(overrides), capacity, promotion)


def _validate_exact_transformations(
    base: WorkloadConfig, variants: tuple[VariantDefinition, ...]
) -> None:
    expected: dict[str, tuple[dict[str, int | float], float, float]] = {
        "baseline": ({}, 0.25, 2.0),
        "capacity_10": ({}, 0.10, 2.0),
        "capacity_40": ({}, 0.40, 2.0),
        "promotion_0": ({}, 0.25, 0.0),
        "promotion_1": ({}, 0.25, 1.0),
        "promotion_4": ({}, 0.25, 4.0),
        "noise_low": ({"noise_access_weight": 0.5 * base.noise_access_weight}, 0.25, 2.0),
        "noise_high": ({"noise_access_weight": 2.0 * base.noise_access_weight}, 0.25, 2.0),
        "burst_short": (
            {
                "burst_duration_min_windows": max(1, math.floor(0.5 * base.burst_duration_min_windows)),
                "burst_duration_max_windows": max(
                    max(1, math.floor(0.5 * base.burst_duration_min_windows)),
                    math.floor(0.5 * base.burst_duration_max_windows),
                ),
            },
            0.25,
            2.0,
        ),
        "burst_long": (
            {
                "burst_duration_min_windows": 2 * base.burst_duration_min_windows,
                "burst_duration_max_windows": 2 * base.burst_duration_max_windows,
            },
            0.25,
            2.0,
        ),
        "context_strong": (
            {
                "precursor_probability_scale": 1.25 * base.precursor_probability_scale,
                "spontaneous_activation_probability": 0.5 * base.spontaneous_activation_probability,
            },
            0.25,
            2.0,
        ),
        "context_weak": (
            {
                "precursor_probability_scale": 0.75 * base.precursor_probability_scale,
                "spontaneous_activation_probability": 2.0 * base.spontaneous_activation_probability,
            },
            0.25,
            2.0,
        ),
    }
    for variant in variants:
        overrides, capacity, promotion = expected[variant.id]
        if variant.workload_overrides != overrides:
            raise ManifestError(f"variant {variant.id} has incorrect transformed values")
        if variant.capacity_fraction != capacity:
            raise ManifestError(f"variant {variant.id} has incorrect capacity fraction")
        if variant.promotion_saved_read_equivalents != promotion:
            raise ManifestError(f"variant {variant.id} has incorrect promotion value")
        raw = base.to_dict()
        raw.update(overrides)
        WorkloadConfig.from_dict(raw)


def _validate_base_settings(raw: Any) -> None:
    expected = {
        "capacity_fraction": 0.25,
        "fast_read_cost": 1.0,
        "slow_read_cost": 10.0,
        "promotion_saved_read_equivalents": 2.0,
    }
    if raw != expected:
        raise ManifestError("base_simulation_settings must contain exact frozen values")


def _policies(raw: Any) -> tuple[PolicyDefinition, ...]:
    if not isinstance(raw, list):
        raise ManifestError("policy_order must be a list")
    result = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"id", "display_name"}:
            raise ManifestError("policy definitions have invalid fields")
        result.append(
            PolicyDefinition(
                _nonempty_string("policy.id", item["id"]),
                _nonempty_string("policy.display_name", item["display_name"]),
            )
        )
    return tuple(result)


def _config_path(root: Path, raw: Any, name: str) -> Path:
    relative = Path(_nonempty_string(name, raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestError(f"{name} must be a repository-relative path")
    path = root / relative
    if not path.is_file():
        path = Path.cwd() / relative
    if not path.is_file():
        raise ManifestError(f"{name} does not exist: {relative}")
    return path


def _integer_tuple(name: str, raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw
    ):
        raise ManifestError(f"{name} must be an integer list")
    return tuple(raw)


def _nonempty_string(name: str, raw: Any) -> str:
    if not isinstance(raw, str) or not raw:
        raise ManifestError(f"{name} must be a nonempty string")
    return raw


def _number(name: str, raw: Any, *, minimum: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ManifestError(f"{name} must be a finite number")
    value = float(raw)
    if not math.isfinite(value) or value < minimum:
        raise ManifestError(f"{name} must be finite and at least {minimum}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate manifest field: {key}")
        result[key] = value
    return result
