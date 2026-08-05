"""Post-replay dynamic-action and planted-transition diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from prism.simulation.config import SimulationConfig
from prism.simulation.replay import BOUNDARY_POLICIES, PolicyReplayResult


def evaluate_causal_diagnostics(
    replay: PolicyReplayResult,
    observable_demand: np.ndarray,
    record_ids: Sequence[int],
    record_sizes: Sequence[int],
    hidden_ground_truth: Mapping[str, Any],
    config: SimulationConfig,
) -> dict[str, Any]:
    """Evaluate residency behavior after replay; never influence policy actions."""

    demand = np.asarray(observable_demand, dtype=np.int64)
    ids = tuple(int(value) for value in record_ids)
    sizes = tuple(int(value) for value in record_sizes)
    if demand.shape[1] != len(ids) or len(sizes) != len(ids):
        raise ValueError("causal diagnostic record inputs are incompatible")
    if replay.pre_window_resident_indicator.shape != (
        len(replay.policy_names),
        len(replay.test_window_ids),
        len(ids),
    ):
        raise ValueError("causal diagnostics require complete residency traces")
    record_index = {record_id: index for index, record_id in enumerate(ids)}
    size_array = np.asarray(sizes, dtype=np.int64)
    policy_index = {name: index for index, name in enumerate(replay.policy_names)}

    dynamic_windows: dict[str, list[dict[str, Any]]] = {}
    dynamic_summaries: dict[str, dict[str, Any]] = {}
    for policy_name in replay.policy_names:
        if policy_name not in BOUNDARY_POLICIES:
            continue
        index = policy_index[policy_name]
        previous = replay.previous_target_indicator[index]
        rows = []
        for position, window_id in enumerate(replay.test_window_ids):
            current = replay.pre_window_resident_indicator[index, position]
            added = current & ~previous
            removed = previous & ~current
            intersection = current & previous
            union = current | previous
            previous_bytes = int(np.sum(size_array[previous], dtype=np.int64))
            overlap_bytes = int(np.sum(size_array[intersection], dtype=np.int64))
            rows.append(
                {
                    "window_id": int(window_id),
                    "target_set_changed": bool(np.any(current != previous)),
                    "records_added": int(np.sum(added)),
                    "records_removed": int(np.sum(removed)),
                    "bytes_added": int(np.sum(size_array[added], dtype=np.int64)),
                    "bytes_removed": int(np.sum(size_array[removed], dtype=np.int64)),
                    "jaccard_similarity": (
                        int(np.sum(intersection)) / int(np.sum(union))
                        if np.any(union)
                        else 1.0
                    ),
                    "resident_byte_overlap": {
                        "bytes": overlap_bytes,
                        "previous_target_bytes": previous_bytes,
                        "fraction": (
                            overlap_bytes / previous_bytes if previous_bytes else 1.0
                        ),
                    },
                    "promotion_count": int(
                        replay.per_window["promotion_count"][index, position]
                    ),
                    "promotion_bytes": int(
                        replay.per_window["bytes_promoted"][index, position]
                    ),
                    "eviction_count": int(
                        replay.per_window["eviction_count"][index, position]
                    ),
                    "eviction_bytes": int(
                        replay.per_window["bytes_evicted"][index, position]
                    ),
                }
            )
            previous = current
        dynamic_windows[policy_name] = rows
        changed = sum(row["target_set_changed"] for row in rows)
        promotions = sum(row["promotion_count"] > 0 for row in rows)
        dynamic_summaries[policy_name] = {
            "test_window_count": len(rows),
            "target_set_change_count": changed,
            "target_set_change_fraction": changed / len(rows) if rows else None,
            "windows_with_promotion_count": promotions,
            "windows_with_promotion_fraction": promotions / len(rows) if rows else None,
            "test_promotion_count": sum(row["promotion_count"] for row in rows),
            "test_promotion_bytes": sum(row["promotion_bytes"] for row in rows),
        }

    predictive = replay.pre_window_resident_indicator[policy_index["predictive_greedy"]]
    disagreements = {}
    for other in (
        "validation_final_frozen",
        "recent_state_only",
        "residual_baseline_only",
        "activation_intensity_only",
    ):
        other_trace = replay.pre_window_resident_indicator[policy_index[other]]
        per_window = np.sum(predictive != other_trace, axis=1)
        disagreements[other] = {
            "windows_differing_count": int(np.sum(per_window > 0)),
            "windows_differing_fraction": float(np.mean(per_window > 0)),
            "record_window_residency_disagreements": int(np.sum(per_window)),
            "complete_test_residency_trace_identical": bool(
                np.array_equal(predictive, other_trace)
            ),
        }
    predictive_summary = {
        **dynamic_summaries["predictive_greedy"],
        "comparisons": disagreements,
        "identical_to_any_ablation": any(
            disagreements[name]["complete_test_residency_trace_identical"]
            for name in (
                "recent_state_only",
                "activation_intensity_only",
                "residual_baseline_only",
            )
        ),
        "dynamic_test_action": not disagreements["validation_final_frozen"][
            "complete_test_residency_trace_identical"
        ],
    }

    transition_rows = _transition_rows(
        replay,
        demand,
        ids,
        sizes,
        record_index,
        hidden_ground_truth,
        config,
        policy_index,
    )
    return {
        "hidden_truth_usage": "post-replay diagnostics only",
        "dynamic_action": {
            "per_window_by_policy": dynamic_windows,
            "summary_by_policy": dynamic_summaries,
            "predictive_greedy_summary": predictive_summary,
        },
        "pre_transition": {
            "per_burst": transition_rows,
            "aggregate_by_policy": _aggregate_transition_rows(
                transition_rows, replay.policy_names
            ),
        },
    }


def _transition_rows(
    replay: PolicyReplayResult,
    demand: np.ndarray,
    ids: tuple[int, ...],
    sizes: tuple[int, ...],
    record_index: dict[int, int],
    hidden: Mapping[str, Any],
    config: SimulationConfig,
    policy_index: dict[str, int],
) -> list[dict[str, Any]]:
    memberships = hidden.get("working_set_memberships")
    bursts = hidden.get("bursts")
    if not isinstance(memberships, list) or not isinstance(bursts, list):
        raise ValueError("hidden transition truth is incomplete")
    membership_by_factor: dict[int, dict[int, float]] = {}
    for row in memberships:
        membership_by_factor[int(row["working_set_id"])] = {
            int(item["record_id"]): float(item["weight"])
            for item in row["members"]
        }
    test_index = {
        int(window): position for position, window in enumerate(replay.test_window_ids)
    }
    size_map = dict(zip(ids, sizes, strict=True))
    result = []
    for burst in bursts:
        start = int(burst["start_window"])
        if start not in test_index:
            continue
        position = test_index[start]
        factor_id = int(burst["working_set_id"])
        weights = membership_by_factor[factor_id]
        support = set(weights)
        support_indices = np.asarray(
            [record_index[item] for item in sorted(support)], dtype=np.int64
        )
        realized_windows = [start]
        if start + 1 in test_index:
            realized_windows.append(start + 1)
        realized = np.sum(demand[realized_windows], axis=0, dtype=np.int64)
        current = demand[start]
        policy_rows = {}
        for policy_name in replay.policy_names:
            index = policy_index[policy_name]
            resident = replay.pre_window_resident_indicator[index, position]
            resident_support = [
                item for item in sorted(support) if resident[record_index[item]]
            ]
            support_accesses = int(np.sum(realized[support_indices], dtype=np.int64))
            resident_accesses = sum(
                int(realized[record_index[item]]) for item in resident_support
            )
            row = {
                "supported_record_count": len(support),
                "supported_records_resident": len(resident_support),
                "supported_record_coverage_fraction": (
                    len(resident_support) / len(support) if support else None
                ),
                "supported_bytes": sum(size_map[item] for item in support),
                "supported_bytes_resident": sum(
                    size_map[item] for item in resident_support
                ),
                "supported_byte_coverage_fraction": (
                    sum(size_map[item] for item in resident_support)
                    / sum(size_map[item] for item in support)
                    if support
                    else None
                ),
                "membership_weighted_coverage": math.fsum(
                    weights[item] for item in resident_support
                ),
                "realized_support_accesses_s_and_s_plus_1": support_accesses,
                "pre_transition_resident_realized_accesses": resident_accesses,
                "pre_transition_realized_demand_coverage_fraction": (
                    resident_accesses / support_accesses if support_accesses else None
                ),
            }
            if policy_name in BOUNDARY_POLICIES:
                promoted = replay.per_window_promotion_indicator[index, position]
                promoted_support = [
                    item for item in sorted(support) if promoted[record_index[item]]
                ]
                useful_current = [
                    item
                    for item in promoted_support
                    if current[record_index[item]] > 0
                ]
                useful_two = [
                    item
                    for item in promoted_support
                    if realized[record_index[item]] > 0
                ]
                row["proactive_useful_changes"] = {
                    "promoted_supported_record_count": len(promoted_support),
                    "promoted_supported_bytes": sum(
                        size_map[item] for item in promoted_support
                    ),
                    "accessed_in_start_window_count": len(useful_current),
                    "accessed_in_start_window_bytes": sum(
                        size_map[item] for item in useful_current
                    ),
                    "accessed_in_first_two_windows_count": len(useful_two),
                    "accessed_in_first_two_windows_bytes": sum(
                        size_map[item] for item in useful_two
                    ),
                }
            else:
                row["proactive_useful_changes"] = None
            policy_rows[policy_name] = row

        predictive_resident = replay.pre_window_resident_indicator[
            policy_index["predictive_greedy"], position
        ]
        oracle_resident = replay.pre_window_resident_indicator[
            policy_index["oracle_greedy"], position
        ]
        missed = oracle_resident & ~predictive_resident
        missed_ids = [ids[index] for index in np.flatnonzero(missed)]
        missed_accesses = int(np.sum(current[missed], dtype=np.int64))
        result.append(
            {
                "burst_id": int(burst["burst_id"]),
                "working_set_id": factor_id,
                "start_window": start,
                "policy_metrics": policy_rows,
                "missed_oracle_opportunities": {
                    "record_ids": missed_ids,
                    "record_count": len(missed_ids),
                    "bytes": sum(size_map[item] for item in missed_ids),
                    "actual_current_window_accesses": missed_accesses,
                    "additional_access_cost": missed_accesses
                    * (config.slow_read_cost - config.fast_read_cost),
                },
            }
        )
    return result


def _aggregate_transition_rows(
    rows: list[dict[str, Any]], policy_names: Sequence[str]
) -> dict[str, Any]:
    result = {}
    for policy_name in policy_names:
        policy_rows = [row["policy_metrics"][policy_name] for row in rows]
        realized = [
            row["pre_transition_realized_demand_coverage_fraction"]
            for row in policy_rows
            if row["pre_transition_realized_demand_coverage_fraction"] is not None
        ]
        result[policy_name] = {
            "burst_count": len(policy_rows),
            "mean_supported_record_coverage_fraction": (
                math.fsum(row["supported_record_coverage_fraction"] for row in policy_rows)
                / len(policy_rows)
                if policy_rows
                else None
            ),
            "mean_supported_byte_coverage_fraction": (
                math.fsum(row["supported_byte_coverage_fraction"] for row in policy_rows)
                / len(policy_rows)
                if policy_rows
                else None
            ),
            "mean_membership_weighted_coverage": (
                math.fsum(row["membership_weighted_coverage"] for row in policy_rows)
                / len(policy_rows)
                if policy_rows
                else None
            ),
            "mean_realized_demand_coverage_fraction": (
                math.fsum(realized) / len(realized) if realized else None
            ),
            "useful_start_window_promotion_count": sum(
                row["proactive_useful_changes"]["accessed_in_start_window_count"]
                for row in policy_rows
                if row["proactive_useful_changes"] is not None
            ),
            "useful_first_two_window_promotion_count": sum(
                row["proactive_useful_changes"]["accessed_in_first_two_windows_count"]
                for row in policy_rows
                if row["proactive_useful_changes"] is not None
            ),
        }
    return result
