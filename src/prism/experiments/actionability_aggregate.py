"""Deterministic Milestone 5.5 aggregation and precommitted thesis gates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from prism.experiments.config import ActionabilityManifest


def write_actionability_outputs(
    output_dir: str | Path, manifest: ActionabilityManifest
) -> dict[str, Any]:
    root = Path(output_dir)
    report, thesis = aggregate_actionability_root(root, manifest)
    _write_json(root / "aggregate_report.json", report)
    _write_json(root / "thesis_gate_report.json", thesis)
    (root / "aggregate_tables.md").write_text(
        _render_tables(report, thesis), encoding="utf-8"
    )
    return report


def aggregate_actionability_root(
    root: Path, manifest: ActionabilityManifest
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = _load_json(root / "experiment_index.json")
    completed = [row for row in index["runs"] if row["status"] == "completed"]
    failed = [row for row in index["runs"] if row["status"] == "failed"]
    runs = [_run_row(root, row) for row in completed]
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in runs:
        by_cell.setdefault((row["regime"], row["horizon"]), []).append(row)
    for rows in by_cell.values():
        rows.sort(key=lambda row: row["seed"])

    regime_rows = _regime_diagnostics(completed)
    separation = _regime_separation(regime_rows)
    cell_gates = []
    for cell in manifest.thesis_candidate_cells:
        regime, raw_horizon = cell.rsplit("__h", 1)
        rows = by_cell.get((regime, int(raw_horizon)), [])
        cell_gates.append(_cell_gates(cell, rows, manifest.thesis_gate_thresholds))
    engineering_complete = len(completed) == 27 and not failed
    candidate_complete = all(row["seed_count"] == 3 for row in cell_gates)
    passing_cells = [row["cell"] for row in cell_gates if row["passed"]]
    thesis_status = decide_thesis_status(
        cell_gates,
        engineering_complete=engineering_complete and candidate_complete,
        regime_separation_sufficient=separation["meaningful_aggregate_separation"],
    )
    thesis = {
        "schema_version": 1,
        "source_manifest_sha256": manifest.sha256,
        "candidate_cells": list(manifest.thesis_candidate_cells),
        "gate_thresholds": manifest.thesis_gate_thresholds,
        "numerical_tolerance": manifest.thesis_gate_thresholds["numerical_tolerance"],
        "cell_gates": cell_gates,
        "passing_cells": passing_cells,
        "engineering_complete": engineering_complete,
        "regime_separation_sufficient": separation["meaningful_aggregate_separation"],
        "thesis_status": thesis_status,
        "decision_text": _decision_text(thesis_status),
    }
    report = {
        "schema_version": 1,
        "source_manifest_sha256": manifest.sha256,
        "completed_run_count": len(completed),
        "failed_run_count": len(failed),
        "pending_run_count": len(index["runs"]) - len(completed) - len(failed),
        "run_results": runs,
        "regime_diagnostics": regime_rows,
        "regime_separation": separation,
        "horizon_predictor_metrics": _horizon_metrics(runs),
        "policy_metrics_by_cell": _policy_metrics(by_cell),
        "paired_policy_comparisons": _paired_policy_comparisons(by_cell),
        "factor_movement": [
            _select(row, "experiment_id", "factor_movement") for row in runs
        ],
        "record_rank_turnover": [
            _select(row, "experiment_id", "record_rank_turnover") for row in runs
        ],
        "controller_rejection_reasons": [
            _select(row, "experiment_id", "controller_summary") for row in runs
        ],
        "component_shares": [
            _select(row, "experiment_id", "component_shares") for row in runs
        ],
        "selection_margins": [
            _select(row, "experiment_id", "selection_margins") for row in runs
        ],
        "oracle_target_agreement": [
            _select(row, "experiment_id", "oracle_agreement") for row in runs
        ],
        "proactive_promotion_repayment": [
            _select(row, "experiment_id", "promotion_repayment") for row in runs
        ],
        "candidate_cell_gates": cell_gates,
        "thesis_status": thesis_status,
        "warnings": separation["warnings"],
        "limitations": [
            "Costs are simulated units rather than measured storage latency.",
            "Three fixed seeds provide descriptive evidence only.",
            "Oracle agreement is matched-horizon and myopic, not trajectory-optimal causal regret.",
        ],
        "failed_runs": [
            {
                "experiment_id": row["experiment_id"],
                "stage_reached": row["stage_reached"],
                "failure": row["failure"],
            }
            for row in failed
        ],
    }
    return report, thesis


def decide_thesis_status(
    cell_gates: Sequence[Mapping[str, Any]],
    *,
    engineering_complete: bool,
    regime_separation_sufficient: bool,
) -> str:
    """Apply the precommitted final decision rule without reading experiment files."""
    if not engineering_complete or not regime_separation_sufficient:
        return "insufficient_evidence"
    if any(bool(row["passed"]) for row in cell_gates):
        return "actionable_predictive_tiering_demonstrated"
    return "stable_cost_aware_tiering_reframe"


def _run_row(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = root / "runs" / str(entry["experiment_id"])
    simulation = _load_json(run_dir / "simulation" / "evaluation_report.json")
    actionability = _load_json(run_dir / "actionability" / "actionability_report.json")
    predictor = _load_json(run_dir / "predictor" / "evaluation_report.json")
    causal = simulation["causal_diagnostics"]
    predictive = simulation["policy_metrics"]["predictive_greedy"]
    frozen = simulation["policy_metrics"]["validation_final_frozen"]
    recent = simulation["policy_metrics"]["recent_state_only"]
    disagreement = causal["dynamic_action"]["predictive_greedy_summary"]["comparisons"]["validation_final_frozen"]
    transition = actionability["transition_coverage"]["aggregate_by_policy"]
    return {
        "experiment_id": entry["experiment_id"],
        "regime": entry["regime"],
        "horizon": entry["horizon"],
        "seed": entry["seed"],
        "predictor_scientific_gates": predictor["scientific_gates"],
        "activation_test_metrics": predictor["activation"]["test"],
        "intensity_test_metrics": predictor["intensity"]["test"],
        "projection_coefficients": simulation["projection"]["coefficients"],
        "policy_metrics": simulation["policy_metrics"],
        "predictive_frozen_difference_fraction": disagreement["windows_differing_fraction"],
        "predictive_cost": predictive["total_combined_cost"],
        "frozen_cost": frozen["total_combined_cost"],
        "recent_state_cost": recent["total_combined_cost"],
        "predictive_access_cost": predictive["total_access_cost"],
        "frozen_access_cost": frozen["total_access_cost"],
        "predictive_transition_coverage": transition["predictive_greedy"],
        "frozen_transition_coverage": transition["validation_final_frozen"],
        "factor_movement": actionability["factor_forecast_movement"],
        "record_rank_turnover": actionability["record_projection_movement"]["summary"],
        "record_forecast_metrics": actionability["record_projection_movement"]["cumulative_record_forecast_metrics"],
        "controller_summary": actionability["controller_actionability"]["test_summary"],
        "component_shares": _mean_component_shares(
            actionability["record_projection_movement"]["component_shares_by_window"]
        ),
        "selection_margins": _mean_selection_margins(
            actionability["controller_actionability"]["per_window"]
        ),
        "oracle_agreement": actionability["matched_horizon_oracle_agreement"]["summary"],
        "promotion_repayment": actionability["promotion_repayment"],
    }


def _cell_gates(
    cell: str,
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    seed_count = len(rows)
    fractions = [float(row["predictive_frozen_difference_fraction"]) for row in rows]
    gate_a = {
        "seed_values": [
            {"seed": row["seed"], "difference_fraction": row["predictive_frozen_difference_fraction"]}
            for row in rows
        ],
        "mean_difference_fraction": _mean(fractions),
    }
    gate_a["passing_seed_count"] = sum(
        value >= thresholds["gate_a_seed_difference_fraction_minimum"]
        for value in fractions
    )
    gate_a["passed"] = bool(
        seed_count == 3
        and gate_a["mean_difference_fraction"] >= thresholds["gate_a_mean_difference_fraction_minimum"]
        and gate_a["passing_seed_count"] >= thresholds["gate_a_minimum_passing_seed_count"]
    )

    comparisons = {}
    for name, key in (
        ("validation_final_frozen", "frozen_cost"),
        ("recent_state_only", "recent_state_cost"),
    ):
        differences = [float(row["predictive_cost"] - row[key]) for row in rows]
        comparisons[name] = {
            "paired_differences_predictive_minus_comparator": [
                {"seed": row["seed"], "difference": difference}
                for row, difference in zip(rows, differences, strict=True)
            ],
            "predictive_win_count": sum(value < 0 for value in differences),
            "mean_difference": _mean(differences),
            "passed": bool(
                seed_count == 3
                and sum(value < 0 for value in differences) >= thresholds["gate_b_minimum_seed_win_count"]
                and _mean(differences) < 0
            ),
        }
    gate_b = {
        "comparators": comparisons,
        "passed": all(item["passed"] for item in comparisons.values()),
    }

    access_differences = [
        float(row["predictive_access_cost"] - row["frozen_access_cost"])
        for row in rows
    ]
    route_1 = {
        "paired_access_cost_differences": [
            {"seed": row["seed"], "difference": difference}
            for row, difference in zip(rows, access_differences, strict=True)
        ],
        "predictive_win_count": sum(value < 0 for value in access_differences),
        "mean_difference": _mean(access_differences),
    }
    route_1["passed"] = bool(
        seed_count == 3
        and route_1["predictive_win_count"] >= thresholds["gate_c_minimum_seed_win_count"]
        and route_1["mean_difference"] < 0
    )
    coverage_differences = [
        float(row["predictive_transition_coverage"] - row["frozen_transition_coverage"])
        for row in rows
        if row["predictive_transition_coverage"] is not None and row["frozen_transition_coverage"] is not None
    ]
    route_2 = {
        "seed_coverage": [
            {
                "seed": row["seed"],
                "predictive": row["predictive_transition_coverage"],
                "frozen": row["frozen_transition_coverage"],
                "difference": (
                    row["predictive_transition_coverage"] - row["frozen_transition_coverage"]
                    if row["predictive_transition_coverage"] is not None and row["frozen_transition_coverage"] is not None else None
                ),
            }
            for row in rows
        ],
        "mean_difference": _mean(coverage_differences),
        "predictive_win_count": sum(value > 0 for value in coverage_differences),
    }
    route_2["passed"] = bool(
        len(coverage_differences) == 3
        and route_2["mean_difference"] >= thresholds["gate_c_transition_coverage_absolute_improvement_minimum"]
        and route_2["predictive_win_count"] >= thresholds["gate_c_minimum_seed_win_count"]
    )
    gate_c = {
        "route_1_access_cost": route_1,
        "route_2_transition_coverage": route_2,
        "passed_route": "access_cost" if route_1["passed"] else "transition_coverage" if route_2["passed"] else None,
        "passed": bool(route_1["passed"] or route_2["passed"]),
    }

    promotion_rows = [row["promotion_repayment"] for row in rows]
    pre_demand = sum(int(row["pre_demand_promotions"]) for row in promotion_rows)
    repaid = sum(int(row["repaid_promotions"]) for row in promotion_rows)
    cost = math.fsum(float(row["total_promotion_cost"]) for row in promotion_rows)
    savings = math.fsum(float(row["total_realized_savings"]) for row in promotion_rows)
    gate_d = {
        "seed_values": [
            {"seed": source["seed"], **summary}
            for source, summary in zip(rows, promotion_rows, strict=True)
        ],
        "total_test_promotions": sum(int(row["total_test_promotions"]) for row in promotion_rows),
        "pre_demand_promotions": pre_demand,
        "repaid_promotions": repaid,
        "repayment_fraction": repaid / pre_demand if pre_demand else None,
        "total_promotion_cost": cost,
        "total_realized_savings": savings,
        "aggregate_net_value": savings - cost,
    }
    gate_d["passed"] = bool(
        seed_count == 3
        and pre_demand >= thresholds["gate_d_minimum_pre_demand_promotions"]
        and gate_d["repayment_fraction"] is not None
        and gate_d["repayment_fraction"] >= thresholds["gate_d_minimum_repayment_fraction"]
        and gate_d["aggregate_net_value"] > 0
    )
    result = {
        "cell": cell,
        "seed_count": seed_count,
        "gate_a_dynamic_behavior": gate_a,
        "gate_b_combined_cost": gate_b,
        "gate_c_non_migration_value": gate_c,
        "gate_d_useful_promotions": gate_d,
    }
    result["passed"] = all(
        result[name]["passed"]
        for name in (
            "gate_a_dynamic_behavior",
            "gate_b_combined_cost",
            "gate_c_non_migration_value",
            "gate_d_useful_promotions",
        )
    )
    return result


def _regime_diagnostics(completed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for entry in completed:
        key = (entry["regime"], entry["seed"])
        if key in seen:
            continue
        seen.add(key)
        values = entry["scientific_gate_outcomes"]["regime_sparsity"]
        rows.append({"regime": entry["regime"], "seed": entry["seed"], **values})
    order = {"baseline": 0, "sparse": 1, "very_sparse": 2}
    rows.sort(key=lambda row: (order[row["regime"]], row["seed"]))
    return rows


def _regime_separation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    means = {}
    for regime in ("baseline", "sparse", "very_sparse"):
        subset = [row for row in rows if row["regime"] == regime]
        means[regime] = {
            "burst_start_rate": _mean([row["burst_starts_per_100_windows"] for row in subset]),
            "multi_active_fraction": _mean([row["fraction_windows_with_at_least_two_active"] for row in subset]),
            "median_dormant_interval": _mean([
                row["pooled_dormant_intervals"]["median"]
                for row in subset if row["pooled_dormant_intervals"]["median"] is not None
            ]),
        }
    burst_values = [means[name]["burst_start_rate"] for name in ("baseline", "sparse", "very_sparse")]
    active_values = [means[name]["multi_active_fraction"] for name in ("baseline", "sparse", "very_sparse")]
    dormant_values = [means[name]["median_dormant_interval"] for name in ("baseline", "sparse", "very_sparse")]
    checks = {
        "burst_start_rate_strictly_decreases": _strict_direction(burst_values, decreasing=True),
        "multi_active_fraction_strictly_decreases": _strict_direction(active_values, decreasing=True),
        "median_dormant_interval_strictly_increases": _strict_direction(dormant_values, decreasing=False),
    }
    passed = all(checks.values())
    return {
        "three_seed_means": means,
        "directional_checks": checks,
        "meaningful_aggregate_separation": passed,
        "warnings": [] if passed else ["Realized regimes did not satisfy every precommitted separation direction; thesis evidence is insufficient without retuning."],
    }


def _horizon_metrics(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": row["experiment_id"],
            "regime": row["regime"],
            "horizon": row["horizon"],
            "seed": row["seed"],
            "predictor_scientific_gates": row["predictor_scientific_gates"],
            "activation_brier": row["activation_test_metrics"]["all_examples"]["context_plus_state_logistic"]["pooled"]["brier_score"],
            "conditional_intensity_rmse": row["intensity_test_metrics"]["context_plus_state_ridge"]["pooled"]["rmse"],
            "projection_coefficients": row["projection_coefficients"],
            "factor_test_metrics": row["factor_movement"]["test"],
            "record_test_metrics": row["record_forecast_metrics"]["test"],
        }
        for row in runs
    ]


def _policy_metrics(by_cell: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for (regime, horizon), rows in by_cell.items():
        policies = rows[0]["policy_metrics"] if rows else {}
        result.append({
            "regime": regime,
            "horizon": horizon,
            "policies": {
                policy: {
                    "mean_access_cost": _mean([row["policy_metrics"][policy]["total_access_cost"] for row in rows]),
                    "mean_promotion_cost": _mean([row["policy_metrics"][policy]["total_promotion_cost"] for row in rows]),
                    "mean_combined_cost": _mean([row["policy_metrics"][policy]["total_combined_cost"] for row in rows]),
                    "mean_hit_rate": _mean([row["policy_metrics"][policy]["hit_rate"] for row in rows]),
                }
                for policy in policies
            },
        })
    return result


def _paired_policy_comparisons(
    by_cell: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows = []
    for (regime, horizon), cell_rows in by_cell.items():
        for comparator, key in (
            ("validation_final_frozen", "frozen_cost"),
            ("recent_state_only", "recent_state_cost"),
        ):
            differences = [float(row["predictive_cost"] - row[key]) for row in cell_rows]
            rows.append({
                "regime": regime,
                "horizon": horizon,
                "comparator": comparator,
                "seed_differences_predictive_minus_comparator": [
                    {"seed": row["seed"], "difference": difference}
                    for row, difference in zip(cell_rows, differences, strict=True)
                ],
                "mean_difference": _mean(differences),
                "predictive_win_count": sum(value < 0 for value in differences),
            })
    return rows


def _mean_component_shares(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("continuation", "activation_intensity", "factor_intercept", "residual_baseline")
    return {
        name: _mean([
            row["shares"][name]
            for row in rows if row["shares"] is not None
        ])
        for name in names
    }


def _mean_selection_margins(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    test = [row["selection_margins"] for row in rows if row["period"] == "test"]
    names = (
        "minimum_selected_net_benefit",
        "minimum_selected_benefit_density",
        "maximum_rejected_positive_net_benefit",
        "maximum_rejected_positive_benefit_density",
        "selected_versus_rejected_density_margin",
        "smallest_positive_nonresident_net_benefit",
        "closest_nonresident_distance_to_zero",
    )
    return {
        name: _mean([row[name] for row in test if row[name] is not None])
        for name in names
    }


def _render_tables(report: Mapping[str, Any], thesis: Mapping[str, Any]) -> str:
    lines = [
        "# Milestone 5.5 Aggregate Tables", "",
        "## Run completion", "",
        "| Completed | Failed | Pending |", "|---:|---:|---:|",
        f"| {report['completed_run_count']} | {report['failed_run_count']} | {report['pending_run_count']} |", "",
        "## Realized regime sparsity", "",
        "| Regime | Seed | Burst starts / 100 | Active-window fraction | Multi-active fraction | Dormant median |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["regime_diagnostics"]:
        lines.append(f"| {row['regime']} | {row['seed']} | {_fmt(row['burst_starts_per_100_windows'])} | {_fmt(row['fraction_windows_with_active_burst'])} | {_fmt(row['fraction_windows_with_at_least_two_active'])} | {_fmt(row['pooled_dormant_intervals']['median'])} |")
    lines.extend(["", "## Predictor and projection metrics by horizon", "", "| Experiment | Activation Brier | Conditional-intensity RMSE | Factor MAE | Factor RMSE | Record MAE | Record RMSE |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in report["horizon_predictor_metrics"]:
        lines.append(f"| {row['experiment_id']} | {_fmt(row['activation_brier'])} | {_fmt(row['conditional_intensity_rmse'])} | {_fmt(row['factor_test_metrics']['mae'])} | {_fmt(row['factor_test_metrics']['rmse'])} | {_fmt(row['record_test_metrics']['mae'])} | {_fmt(row['record_test_metrics']['rmse'])} |")
    lines.extend(["", "## Policy cost by cell", "", "| Regime | H | Policy | Access | Promotion | Combined | Hit rate |", "|---|---:|---|---:|---:|---:|---:|"])
    for cell in report["policy_metrics_by_cell"]:
        for policy, values in cell["policies"].items():
            lines.append(f"| {cell['regime']} | {cell['horizon']} | {policy} | {_fmt(values['mean_access_cost'])} | {_fmt(values['mean_promotion_cost'])} | {_fmt(values['mean_combined_cost'])} | {_fmt(values['mean_hit_rate'])} |")
    lines.extend(["", "## Predictive versus frozen and recent-state-only", "", "Differences are Predictive Greedy (Prism) minus comparator; negative favors Predictive.", "", "| Regime | H | Comparator | Seed differences | Mean | Predictive wins |", "|---|---:|---|---|---:|---:|"])
    for row in report["paired_policy_comparisons"]:
        differences = ", ".join(f"{item['seed']}:{_fmt(item['difference'])}" for item in row["seed_differences_predictive_minus_comparator"])
        lines.append(f"| {row['regime']} | {row['horizon']} | {row['comparator']} | {differences} | {_fmt(row['mean_difference'])} | {row['predictive_win_count']} |")
    lines.extend(["", "## Predictive actionability diagnostics", "", "| Experiment | Rank change | Movement-cost rejects | Capacity rejects | Oracle Jaccard | Pre-demand | Repaid fraction | Net value |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    rank = {row["experiment_id"]: row["record_rank_turnover"] for row in report["record_rank_turnover"]}
    controller = {row["experiment_id"]: row["controller_summary"] for row in report["controller_rejection_reasons"]}
    oracle = {row["experiment_id"]: row["oracle_agreement"] for row in report["oracle_target_agreement"]}
    promotion = {row["experiment_id"]: row["promotion_repayment"] for row in report["proactive_promotion_repayment"]}
    for experiment_id in rank:
        lines.append(f"| {experiment_id} | {_fmt(rank[experiment_id]['mean_normalized_rank_change'])} | {controller[experiment_id]['movement_cost_rejected_record_count']} | {controller[experiment_id]['capacity_rejected_record_count']} | {_fmt(oracle[experiment_id]['mean_target_jaccard'])} | {promotion[experiment_id]['pre_demand_promotions']} | {_fmt(promotion[experiment_id]['repayment_fraction'])} | {_fmt(promotion[experiment_id]['aggregate_net_value'])} |")
    lines.extend(["", "## Gates A–D", "", "| Candidate cell | A | B | C | C route | D | Overall |", "|---|---|---|---|---|---|---|"])
    for row in thesis["cell_gates"]:
        lines.append(f"| {row['cell']} | {row['gate_a_dynamic_behavior']['passed']} | {row['gate_b_combined_cost']['passed']} | {row['gate_c_non_migration_value']['passed']} | {row['gate_c_non_migration_value']['passed_route'] or '—'} | {row['gate_d_useful_promotions']['passed']} | {row['passed']} |")
    lines.extend(["", "## Final thesis decision", "", f"`{thesis['thesis_status']}`", "", thesis["decision_text"], ""])
    return "\n".join(lines)


def _decision_text(status: str) -> str:
    if status == "actionable_predictive_tiering_demonstrated":
        return "Prism demonstrates actionable predictive storage tiering under at least one precommitted controlled regime and cumulative decision horizon."
    if status == "stable_cost_aware_tiering_reframe":
        return "The current architecture does not demonstrate dynamic predictive actionability under the tested precommitted sparse regimes and cumulative horizons. Prism should be reframed as learned latent-demand structure for stable, cost-aware storage tiering."
    return "Engineering failures or ineffective realized regime separation prevent the precommitted candidate cells from supporting either thesis conclusion."


def _select(row: Mapping[str, Any], id_key: str, value_key: str) -> dict[str, Any]:
    return {id_key: row[id_key], value_key: row[value_key]}


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(float(value) for value in values) / len(values) if values else None


def _strict_direction(values: Sequence[float | None], *, decreasing: bool) -> bool:
    if len(values) != 3 or any(value is None for value in values):
        return False
    first, second, third = (float(value) for value in values)
    return first > second > third if decreasing else first < second < third


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
