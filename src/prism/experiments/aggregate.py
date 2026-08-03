"""Deterministic seed aggregation, paired comparisons, and hypothesis evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from prism.experiments.config import ExperimentManifest


PAIRED_RIGHT_POLICIES = (
    "validation_final_frozen",
    "training_popularity_static",
    "recent_state_only",
    "activation_intensity_only",
    "residual_baseline_only",
    "recent_demand_greedy",
    "lru",
    "lfu",
    "oracle_greedy",
)
NUMERICAL_TOLERANCE = 1e-9
AGGREGATE_METRICS = (
    "total_access_cost",
    "total_promotion_cost",
    "total_combined_cost",
    "mean_combined_cost_per_access",
    "hit_rate",
    "slow_reads",
    "bytes_promoted",
    "wasted_promoted_bytes",
    "burst_start_window_combined_cost",
    "first_two_window_combined_cost",
    "target_set_change_count",
    "test_promotion_count",
    "record_window_disagreements_from_frozen",
    "mean_pre_transition_realized_demand_coverage",
    "mean_pre_transition_supported_record_coverage",
)


def aggregate_experiment_root(
    output_dir: str | Path, manifest: ExperimentManifest
) -> dict[str, Any]:
    """Aggregate only completed matching variant/seed runs."""

    root = Path(output_dir)
    index = _load_json(root / "experiment_index.json")
    completed_entries = [row for row in index["runs"] if row["status"] == "completed"]
    failed_entries = [row for row in index["runs"] if row["status"] == "failed"]
    seed_results = [
        _seed_result(root, entry, manifest) for entry in completed_entries
    ]
    seed_by_key = {
        (row["variant_id"], row["seed"]): row for row in seed_results
    }
    variant_results = _variant_aggregations(seed_results, manifest)
    paired = _paired_comparisons(seed_by_key, manifest)
    break_even = _break_even(seed_by_key, manifest)
    hypotheses = _hypotheses(paired, manifest)
    report = {
        "schema_version": 1,
        "source_manifest_sha256": manifest.sha256,
        "completed_run_count": len(completed_entries),
        "failed_run_count": len(failed_entries),
        "pending_run_count": sum(row["status"] == "pending" for row in index["runs"]),
        "seed_level_policy_results": seed_results,
        "variant_level_aggregations": variant_results,
        "paired_policy_comparisons": paired,
        "static_control_comparisons": _comparison_subset(
            paired,
            {"validation_final_frozen", "training_popularity_static"},
        ),
        "ablation_comparisons": _comparison_subset(
            paired,
            {
                "recent_state_only",
                "activation_intensity_only",
                "residual_baseline_only",
            },
        ),
        "migration_cost_analysis": {
            "behavioral_variant_order": [
                "promotion_0",
                "promotion_1",
                "baseline",
                "promotion_4",
            ],
            "behavioral_results": _variant_subset(
                variant_results,
                ["promotion_0", "promotion_1", "baseline", "promotion_4"],
            ),
            "fixed_trajectory_break_even": break_even,
            "caveat": (
                "Fixed-trajectory accounting holds realized access and promotion "
                "trajectories constant. Behavioral promotion-cost runs are primary "
                "because changing migration cost can change placement decisions."
            ),
        },
        "capacity_analysis": _family_analysis(
            seed_results, variant_results, ["capacity_10", "baseline", "capacity_40"]
        ),
        "context_analysis": _family_analysis(
            seed_results, variant_results, ["context_strong", "baseline", "context_weak"]
        ),
        "noise_analysis": _family_analysis(
            seed_results, variant_results, ["noise_low", "baseline", "noise_high"]
        ),
        "duration_analysis": _family_analysis(
            seed_results, variant_results, ["burst_short", "baseline", "burst_long"]
        ),
        "dynamic_action_diagnostics": [
            {
                "experiment_id": row["experiment_id"],
                "variant_id": row["variant_id"],
                "seed": row["seed"],
                "predictive_greedy": row["dynamic_action"]["predictive_greedy_summary"],
            }
            for row in seed_results
        ],
        "pre_transition_coverage": [
            {
                "experiment_id": row["experiment_id"],
                "variant_id": row["variant_id"],
                "seed": row["seed"],
                "aggregate_by_policy": row["pre_transition"],
            }
            for row in seed_results
        ],
        "oracle_regret": _oracle_regret(seed_results),
        "hypothesis_summaries": hypotheses,
        "hypothesis_rule": {
            "supported": (
                "The expected direction occurs for all three seeds in at least one "
                "relevant non-baseline configuration and relevant configurations are "
                "not uniformly majority-contradictory."
            ),
            "not_supported": (
                "No relevant configuration shows the expected direction for a majority of seeds."
            ),
            "mixed": "Evidence varies across relevant seeds or configurations.",
            "insufficient_data": "At least one required variant lacks all three seed values.",
        },
        "failed_runs": [
            {
                "experiment_id": row["experiment_id"],
                "variant_id": row["variant_id"],
                "seed": row["seed"],
                "stage_reached": row["stage_reached"],
                "failure": row["failure"],
            }
            for row in failed_entries
        ],
        "warnings": [
            "Three seeds provide limited uncertainty estimation; confidence intervals and significance tests are not reported.",
            "Scientific gate failures are experimental outcomes and do not exclude engineering-valid completed runs.",
        ],
        "limitations": [
            "Costs are simulated units rather than measured storage latency.",
            "Oracle policies are one-window myopic policies, not full-horizon optima.",
            "Oracle regret terms are descriptive and are not claimed to be an exact additive decomposition.",
        ],
    }
    return report


def write_aggregate_outputs(
    output_dir: str | Path, manifest: ExperimentManifest
) -> dict[str, Any]:
    from prism.experiments.report import render_aggregate_tables

    root = Path(output_dir)
    report = aggregate_experiment_root(root, manifest)
    _write_json(root / "aggregate_report.json", report)
    (root / "aggregate_tables.md").write_text(
        render_aggregate_tables(report, manifest), encoding="utf-8"
    )
    return report


def descriptive_statistics(values: Iterable[float]) -> dict[str, Any]:
    resolved = [float(value) for value in values]
    if not resolved:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sample_standard_deviation_ddof_1": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(resolved),
        "mean": math.fsum(resolved) / len(resolved),
        "median": statistics.median(resolved),
        "sample_standard_deviation_ddof_1": (
            statistics.stdev(resolved) if len(resolved) >= 2 else None
        ),
        "minimum": min(resolved),
        "maximum": max(resolved),
    }


def fixed_trajectory_crossover(
    left_access_cost: float,
    left_bytes_promoted: int,
    right_access_cost: float,
    right_bytes_promoted: int,
) -> dict[str, Any]:
    denominator = left_bytes_promoted - right_bytes_promoted
    numerator = right_access_cost - left_access_cost
    if denominator == 0:
        return {
            "promotion_cost_per_byte_crossover": None,
            "meaningful": False,
            "explanation": "undefined because both trajectories promote equal bytes",
        }
    crossover = numerator / denominator
    if not math.isfinite(crossover) or crossover < 0.0:
        return {
            "promotion_cost_per_byte_crossover": None,
            "meaningful": False,
            "explanation": "the accounting crossover is negative or non-finite",
        }
    return {
        "promotion_cost_per_byte_crossover": crossover,
        "meaningful": True,
        "explanation": "local fixed-trajectory accounting crossover",
    }


def _seed_result(
    root: Path, entry: dict[str, Any], manifest: ExperimentManifest
) -> dict[str, Any]:
    run_root = root / "runs" / entry["experiment_id"]
    simulation = _load_json(run_root / "simulation" / "evaluation_report.json")
    structure = _load_json(run_root / "structure" / "recovery_report.json")
    predictor = _load_json(run_root / "predictor" / "evaluation_report.json")
    workload = _load_json(run_root / "workload" / "workload_validation.json")
    resolved_simulation = _load_json(run_root / "resolved_simulation_config.json")
    transitions = simulation["transition_metrics"]
    dynamic = simulation["causal_diagnostics"]["dynamic_action"]
    pre_transition = simulation["causal_diagnostics"]["pre_transition"][
        "aggregate_by_policy"
    ]
    policies = {}
    for policy in manifest.policies:
        metrics = dict(simulation["policy_metrics"][policy.id])
        metrics.update(
            {
                "burst_start_window_combined_cost": transitions[
                    "burst_start_windows"
                ][policy.id]["combined_cost"],
                "first_two_window_combined_cost": transitions[
                    "first_two_windows"
                ][policy.id]["combined_cost"],
                "target_set_change_count": dynamic["summary_by_policy"]
                .get(policy.id, {})
                .get("target_set_change_count"),
                "test_promotion_count": metrics["promotion_count"],
                "record_window_disagreements_from_frozen": (
                    dynamic["predictive_greedy_summary"]["comparisons"][
                        "validation_final_frozen"
                    ]["record_window_residency_disagreements"]
                    if policy.id == "predictive_greedy"
                    else None
                ),
                "mean_pre_transition_realized_demand_coverage": pre_transition[
                    policy.id
                ]["mean_realized_demand_coverage_fraction"],
                "mean_pre_transition_supported_record_coverage": pre_transition[
                    policy.id
                ]["mean_supported_record_coverage_fraction"],
            }
        )
        policies[policy.id] = {
            "display_name": policy.display_name,
            **metrics,
        }
    return {
        "experiment_id": entry["experiment_id"],
        "variant_id": entry["variant_id"],
        "seed": entry["seed"],
        "resolved_simulation_config": resolved_simulation,
        "policy_results": policies,
        "dynamic_action": dynamic,
        "pre_transition": pre_transition,
        "projection_test_metrics": simulation["projection"]["diagnostics"]["test"],
        "missed_oracle_opportunities": _missed_oracle_summary(
            simulation["causal_diagnostics"]["pre_transition"]["per_burst"]
        ),
        "scientific_gate_outcomes": {
            "workload_demonstrations": workload["demonstration_checks"][
                "all_required_demonstrations_passed"
            ],
            "workload_intensity_signal": workload["intensity_predictability"][
                "signal_gate"
            ]["all_required_conditions_passed"],
            "structure_recovery": structure["representative_gate"]["passed"],
            "predictor": predictor["scientific_gates"],
            "simulation": simulation["scientific_gates"],
        },
    }


def _variant_aggregations(
    rows: list[dict[str, Any]], manifest: ExperimentManifest
) -> dict[str, Any]:
    result = {}
    for variant in manifest.variants:
        selected = [row for row in rows if row["variant_id"] == variant.id]
        policy_output = {}
        for policy in manifest.policies:
            metric_output = {}
            for metric in AGGREGATE_METRICS:
                individual = [
                    {"seed": row["seed"], "value": row["policy_results"][policy.id][metric]}
                    for row in selected
                    if row["policy_results"][policy.id][metric] is not None
                ]
                metric_output[metric] = {
                    "individual_seed_values": individual,
                    **descriptive_statistics(item["value"] for item in individual),
                }
            policy_output[policy.id] = {
                "display_name": policy.display_name,
                "metrics": metric_output,
            }
        result[variant.id] = {
            "completed_seed_count": len(selected),
            "policies": policy_output,
        }
    return result


def _paired_comparisons(
    rows: dict[tuple[str, int], dict[str, Any]], manifest: ExperimentManifest
) -> list[dict[str, Any]]:
    output = []
    for right in PAIRED_RIGHT_POLICIES:
        comparison = {
            "comparison_id": f"predictive_greedy_minus_{right}",
            "left_policy_id": "predictive_greedy",
            "right_policy_id": right,
            "sign_convention": "left total cost - right total cost; negative favors left",
            "variants": [],
        }
        for variant in manifest.variants:
            differences = []
            for seed in manifest.seeds:
                row = rows.get((variant.id, seed))
                if row is None:
                    continue
                left = row["policy_results"]["predictive_greedy"]["total_combined_cost"]
                right_value = row["policy_results"][right]["total_combined_cost"]
                differences.append({"seed": seed, "difference": left - right_value})
            values = [row["difference"] for row in differences]
            comparison["variants"].append(
                {
                    "variant_id": variant.id,
                    "paired_differences": differences,
                    **descriptive_statistics(values),
                    "left_costs_less_count": sum(value < -NUMERICAL_TOLERANCE for value in values),
                    "equal_within_tolerance_count": sum(abs(value) <= NUMERICAL_TOLERANCE for value in values),
                    "left_costs_more_count": sum(value > NUMERICAL_TOLERANCE for value in values),
                    "numerical_tolerance": NUMERICAL_TOLERANCE,
                }
            )
        output.append(comparison)
    return output


def _break_even(
    rows: dict[tuple[str, int], dict[str, Any]], manifest: ExperimentManifest
) -> list[dict[str, Any]]:
    output = []
    for seed in manifest.seeds:
        row = rows.get(("baseline", seed))
        if row is None:
            continue
        left = row["policy_results"]["predictive_greedy"]
        for right_name in (
            "recent_demand_greedy",
            "validation_final_frozen",
            "training_popularity_static",
            "recent_state_only",
        ):
            right = row["policy_results"][right_name]
            output.append(
                {
                    "experiment_id": row["experiment_id"],
                    "left_policy_id": "predictive_greedy",
                    "right_policy_id": right_name,
                    **fixed_trajectory_crossover(
                        left["total_access_cost"],
                        left["bytes_promoted"],
                        right["total_access_cost"],
                        right["bytes_promoted"],
                    ),
                    "diagnostic_type": "fixed_trajectory_accounting_only",
                }
            )
    return output


def _hypotheses(
    paired: list[dict[str, Any]], manifest: ExperimentManifest
) -> dict[str, Any]:
    by_right = {item["right_policy_id"]: item for item in paired}
    specs = {
        "dynamic_value": ("validation_final_frozen", [item.id for item in manifest.variants], True),
        "fast_predictor_contribution": ("recent_state_only", ["context_strong", "baseline", "context_weak"], True),
        "beyond_static_popularity": ("training_popularity_static", ["capacity_10", "baseline", "burst_short", "burst_long"], True),
        "factor_and_residual_contribution": ("residual_baseline_only", [item.id for item in manifest.variants], True),
        "capacity_pressure": ("training_popularity_static", ["capacity_10", "baseline", "capacity_40"], True),
        "noise_tolerance": ("recent_demand_greedy", ["noise_low", "baseline", "noise_high"], True),
        "burst_duration": ("training_popularity_static", ["burst_short", "baseline", "burst_long"], True),
        "oracle_regret": ("oracle_greedy", [item.id for item in manifest.variants], False),
    }
    result = {}
    for name, (right, variants, expected_negative) in specs.items():
        rows = {
            row["variant_id"]: [item["difference"] for item in row["paired_differences"]]
            for row in by_right[right]["variants"]
            if row["variant_id"] in variants
        }
        status = _hypothesis_status(rows, variants, expected_negative)
        result[name] = {
            "status": status,
            "comparison": f"predictive_greedy minus {right}",
            "expected_direction": "negative" if expected_negative else "positive",
            "relevant_variants": variants,
            "seed_differences_by_variant": rows,
        }
    return result


def _hypothesis_status(
    rows: dict[str, list[float]], variants: list[str], expected_negative: bool
) -> str:
    if any(len(rows.get(variant, [])) != 3 for variant in variants):
        return "insufficient_data"
    expected = lambda value: value < -NUMERICAL_TOLERANCE if expected_negative else value > NUMERICAL_TOLERANCE
    majority = [sum(expected(value) for value in rows[variant]) >= 2 for variant in variants]
    if not any(majority):
        return "not_supported"
    all_seed_nonbaseline = any(
        variant != "baseline" and all(expected(value) for value in rows[variant])
        for variant in variants
    )
    uniformly_contradictory = all(not value for value in majority)
    if all_seed_nonbaseline and not uniformly_contradictory:
        return "supported"
    return "mixed"


def _missed_oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = [row["missed_oracle_opportunities"] for row in rows]
    return {
        "burst_count": len(items),
        "record_opportunity_count": sum(item["record_count"] for item in items),
        "bytes": sum(item["bytes"] for item in items),
        "actual_current_window_accesses": sum(
            item["actual_current_window_accesses"] for item in items
        ),
        "additional_access_cost": math.fsum(
            item["additional_access_cost"] for item in items
        ),
    }


def _oracle_regret(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        predictive = row["policy_results"]["predictive_greedy"]
        oracle = row["policy_results"]["oracle_greedy"]
        exact = row["policy_results"]["oracle_exact"]
        result.append(
            {
                "experiment_id": row["experiment_id"],
                "variant_id": row["variant_id"],
                "seed": row["seed"],
                "predictive_minus_oracle_greedy_total_cost": predictive["total_combined_cost"] - oracle["total_combined_cost"],
                "access_cost_difference": predictive["total_access_cost"] - oracle["total_access_cost"],
                "promotion_cost_difference": predictive["total_promotion_cost"] - oracle["total_promotion_cost"],
                "oracle_greedy_minus_exact_cost": oracle["total_combined_cost"] - exact["total_combined_cost"],
                "missed_oracle_opportunities": row["missed_oracle_opportunities"],
                "projection_test_record_rmse": row["projection_test_metrics"]["record_demand"]["predictive"]["pooled"]["rmse"],
                "decomposition_note": "descriptive terms; not an exact additive decomposition",
            }
        )
    return result


def _comparison_subset(
    paired: list[dict[str, Any]], right_ids: set[str]
) -> list[dict[str, Any]]:
    return [item for item in paired if item["right_policy_id"] in right_ids]


def _variant_subset(
    variants: dict[str, Any], ordered_ids: list[str]
) -> dict[str, Any]:
    return {variant_id: variants[variant_id] for variant_id in ordered_ids}


def _family_analysis(
    seed_results: list[dict[str, Any]],
    variants: dict[str, Any],
    ordered_ids: list[str],
) -> dict[str, Any]:
    return {
        "variant_order": ordered_ids,
        "aggregations": _variant_subset(variants, ordered_ids),
        "scientific_gate_outcomes": [
            {
                "experiment_id": row["experiment_id"],
                "variant_id": row["variant_id"],
                "seed": row["seed"],
                "outcomes": row["scientific_gate_outcomes"],
            }
            for row in seed_results
            if row["variant_id"] in ordered_ids
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"aggregate source must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
