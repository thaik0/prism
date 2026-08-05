"""Deterministic readable Markdown tables for Milestone 5."""

from __future__ import annotations

from typing import Any

from prism.experiments.config import ExperimentManifest


def render_aggregate_tables(
    report: dict[str, Any], manifest: ExperimentManifest
) -> str:
    lines = [
        "# Milestone 5 Aggregate Tables",
        "",
        "Three seeds provide limited uncertainty estimation; no confidence intervals or significance tests are reported.",
        "",
        "## Run completion",
        "",
        "| Completed | Failed | Pending |",
        "|---:|---:|---:|",
        f"| {report['completed_run_count']} | {report['failed_run_count']} | {report['pending_run_count']} |",
        "",
    ]
    for metric, title in (
        ("total_combined_cost", "Per-variant mean total cost"),
        ("hit_rate", "Per-variant mean hit rate"),
    ):
        lines.extend([f"## {title}", "", "| Variant | Policy | Mean | Median | Sample SD | Min | Max |", "|---|---|---:|---:|---:|---:|---:|"])
        for variant in manifest.variants:
            for policy in manifest.policies:
                row = report["variant_level_aggregations"][variant.id]["policies"][policy.id]["metrics"][metric]
                lines.append(
                    f"| {variant.id} | {policy.display_name} | {_format(row['mean'])} | {_format(row['median'])} | {_format(row['sample_standard_deviation_ddof_1'])} | {_format(row['minimum'])} | {_format(row['maximum'])} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Paired Predictive Greedy (Prism) comparisons",
            "",
            "Differences use left total cost minus right total cost; negative favors Predictive Greedy (Prism).",
            "",
            "| Variant | Right policy | Mean difference | Wins | Equal | Losses |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    display = {policy.id: policy.display_name for policy in manifest.policies}
    for comparison in report["paired_policy_comparisons"]:
        for row in comparison["variants"]:
            lines.append(
                f"| {row['variant_id']} | {display[comparison['right_policy_id']]} | {_format(row['mean'])} | {row['left_costs_less_count']} | {row['equal_within_tolerance_count']} | {row['left_costs_more_count']} |"
            )
    lines.append("")

    lines.extend(
        [
            "## Migration and dynamic action",
            "",
            "| Experiment | Dynamic test action | Target changes | Test promotions | Bytes promoted |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["dynamic_action_diagnostics"]:
        summary = row["predictive_greedy"]
        lines.append(
            f"| {row['experiment_id']} | {str(summary['dynamic_test_action']).lower()} | {summary['target_set_change_count']} | {summary['test_promotion_count']} | {summary['test_promotion_bytes']} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Pre-transition coverage",
            "",
            "| Experiment | Policy | Support coverage | Realized-demand coverage | Useful first-two-window promotions |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["pre_transition_coverage"]:
        for policy in manifest.policies:
            values = row["aggregate_by_policy"][policy.id]
            lines.append(
                f"| {row['experiment_id']} | {policy.display_name} | {_format(values['mean_supported_record_coverage_fraction'])} | {_format(values['mean_realized_demand_coverage_fraction'])} | {values['useful_first_two_window_promotion_count']} |"
            )
    lines.append("")

    lines.extend(
        [
            "## Hypothesis outcomes",
            "",
            "| Hypothesis | Status | Comparison |",
            "|---|---|---|",
        ]
    )
    for name, item in report["hypothesis_summaries"].items():
        lines.append(f"| {name} | {item['status']} | {item['comparison']} |")
    lines.extend(["", "## Failed runs", ""])
    if report["failed_runs"]:
        lines.extend(["| Experiment | Stage | Failure |", "|---|---|---|"])
        for row in report["failed_runs"]:
            failure = row["failure"] or {}
            lines.append(
                f"| {row['experiment_id']} | {row['stage_reached']} | {failure.get('type')}: {failure.get('message')} |"
            )
    else:
        lines.append("No engineering failures.")
    lines.append("")
    return "\n".join(lines)


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
