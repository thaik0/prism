"""Projection diagnostics, transition metrics, and fixed scientific gates."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from prism.simulation.projection import ProjectionResult
from prism.simulation.replay import PolicyReplayResult


EXACT_GATE_ABSOLUTE_TOLERANCE = 1e-9


def evaluate_projection_diagnostics(
    projection: ProjectionResult,
    observable_demand: np.ndarray,
    prediction_available: np.ndarray,
    validation_start: int,
    test_start: int,
) -> dict[str, Any]:
    """Describe calibrated factor and record forecasts without fitting."""

    X = np.asarray(observable_demand, dtype=np.float64)
    available = np.asarray(prediction_available, dtype=np.bool_)
    report: dict[str, Any] = {}
    warnings: list[str] = []
    for period, start, end in (
        ("validation", validation_start, test_start),
        ("test", test_start, len(X)),
    ):
        mask = available & (np.arange(len(X)) >= start) & (np.arange(len(X)) < end)
        windows = np.flatnonzero(mask)
        actual_factor = projection.observable_factor_demand[windows]
        predicted_factor = projection.predicted_factor_demand[windows]
        recent_factor = projection.observable_factor_demand[windows - 1]
        actual_record = X[windows]
        predicted_record = projection.predicted_record_demand[windows]
        recent_record = X[windows - 1]
        report[period] = {
            "target_window_count": len(windows),
            "target_window_ids": windows.tolist(),
            "factor_demand": {
                "predictive": _matrix_metrics(
                    actual_factor,
                    predicted_factor,
                    f"{period} predictive factor demand",
                    warnings,
                    per_column=True,
                ),
                "recent_baseline": _matrix_metrics(
                    actual_factor,
                    recent_factor,
                    f"{period} recent factor demand",
                    warnings,
                    per_column=True,
                ),
            },
            "record_demand": {
                "predictive": _matrix_metrics(
                    actual_record,
                    predicted_record,
                    f"{period} predictive record demand",
                    warnings,
                    per_column=False,
                ),
                "recent_baseline": _matrix_metrics(
                    actual_record,
                    recent_record,
                    f"{period} recent record demand",
                    warnings,
                    per_column=False,
                ),
                "per_window_total_demand_error": {
                    "predictive": _total_error_rows(
                        windows, actual_record, predicted_record
                    ),
                    "recent_baseline": _total_error_rows(
                        windows, actual_record, recent_record
                    ),
                },
            },
        }
    report["warnings"] = warnings
    return report


def evaluate_policy_results(
    replay: PolicyReplayResult,
    bursts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate controlled transitions and evaluate the five fixed gates."""

    test_start = int(replay.test_window_ids[0])
    test_end = int(replay.test_window_ids[-1]) + 1
    test_bursts = [
        dict(burst)
        for burst in bursts
        if test_start <= int(burst["start_window"]) < test_end
    ]
    burst_start_windows = sorted(
        {int(burst["start_window"]) for burst in test_bursts}
    )
    first_two_windows = sorted(
        {
            window_id
            for burst in test_bursts
            for window_id in (
                int(burst["start_window"]),
                int(burst["start_window"]) + 1,
            )
            if test_start <= window_id < test_end
        }
    )
    transitions = {
        "test_burst_start_count": len(test_bursts),
        "unique_burst_start_window_count": len(burst_start_windows),
        "unique_first_two_window_count": len(first_two_windows),
        "burst_start_window_ids": burst_start_windows,
        "first_two_window_ids": first_two_windows,
        "burst_start_windows": _aggregate_windows(replay, burst_start_windows),
        "first_two_windows": _aggregate_windows(replay, first_two_windows),
        "per_burst_descriptive_costs": [
            {
                "burst_id": int(burst["burst_id"]),
                "working_set_id": int(burst["working_set_id"]),
                "start_window": int(burst["start_window"]),
                "window_ids": [
                    window_id
                    for window_id in (
                        int(burst["start_window"]),
                        int(burst["start_window"]) + 1,
                    )
                    if test_start <= window_id < test_end
                ],
                "policy_costs": _aggregate_windows(
                    replay,
                    [
                        window_id
                        for window_id in (
                            int(burst["start_window"]),
                            int(burst["start_window"]) + 1,
                        )
                        if test_start <= window_id < test_end
                    ],
                ),
            }
            for burst in test_bursts
        ],
    }
    metrics = replay.policy_metrics
    start_costs = transitions["burst_start_windows"]
    gate_1 = _strict_less(
        "predictive_greedy",
        metrics["predictive_greedy"]["total_combined_cost"],
        "recent_demand_greedy",
        metrics["recent_demand_greedy"]["total_combined_cost"],
    )
    gate_2_lru = _strict_less(
        "predictive_greedy",
        metrics["predictive_greedy"]["total_combined_cost"],
        "lru",
        metrics["lru"]["total_combined_cost"],
    )
    gate_2_lfu = _strict_less(
        "predictive_greedy",
        metrics["predictive_greedy"]["total_combined_cost"],
        "lfu",
        metrics["lfu"]["total_combined_cost"],
    )
    gate_2 = {
        "name": "reactive-policy comparison",
        "lru_comparison": gate_2_lru,
        "lfu_comparison": gate_2_lfu,
        "passed": bool(gate_2_lru["passed"] and gate_2_lfu["passed"]),
    }
    gate_3 = _strict_less(
        "predictive_greedy",
        start_costs["predictive_greedy"]["combined_cost"],
        "recent_demand_greedy",
        start_costs["recent_demand_greedy"]["combined_cost"],
    )
    gate_4 = _strict_less(
        "oracle_greedy",
        metrics["oracle_greedy"]["total_combined_cost"],
        "recent_demand_greedy",
        metrics["recent_demand_greedy"]["total_combined_cost"],
    )
    gate_5 = _less_or_equal(
        "oracle_exact",
        metrics["oracle_exact"]["total_combined_cost"],
        "oracle_greedy",
        metrics["oracle_greedy"]["total_combined_cost"],
        EXACT_GATE_ABSOLUTE_TOLERANCE,
    )
    gates = {
        "gate_1": {"name": "predictive value holding controller fixed", **gate_1},
        "gate_2": gate_2,
        "gate_3": {"name": "transition advantage", **gate_3},
        "gate_4": {"name": "predictive opportunity exists", **gate_4},
        "gate_5": {"name": "exact-controller sanity", **gate_5},
    }
    gates["all_passed"] = all(gates[f"gate_{index}"]["passed"] for index in range(1, 6))
    oracle_ordering = _less_or_equal(
        "oracle_greedy",
        metrics["oracle_greedy"]["total_combined_cost"],
        "predictive_greedy",
        metrics["predictive_greedy"]["total_combined_cost"],
        EXACT_GATE_ABSOLUTE_TOLERANCE,
    )
    oracle_exact_cost = metrics["oracle_exact"]["total_combined_cost"]
    oracle_greedy_cost = metrics["oracle_greedy"]["total_combined_cost"]
    warnings = []
    if not gates["all_passed"]:
        warnings.append("one or more Milestone 4 test scientific gates failed")
    if not oracle_ordering["passed"]:
        warnings.append(
            "oracle greedy costs more than predictive greedy under different myopic trajectories"
        )
    return {
        "transition_metrics": transitions,
        "scientific_gates": gates,
        "oracle_ordering_diagnostic": oracle_ordering,
        "controller_diagnostics": {
            "exact_solver_windows": list(replay.exact_solver_diagnostics),
            "all_exact_windows_optimal": all(
                row["solver_status"].startswith("optimal")
                for row in replay.exact_solver_diagnostics
            ),
            "oracle_greedy_total_combined_cost": oracle_greedy_cost,
            "oracle_exact_total_combined_cost": oracle_exact_cost,
            "greedy_minus_exact_cost": oracle_greedy_cost - oracle_exact_cost,
            "greedy_approximation_gap_fraction": (
                (oracle_greedy_cost - oracle_exact_cost) / oracle_exact_cost
                if oracle_exact_cost
                else None
            ),
            "trajectory_warning": (
                "Oracle policies use the same one-window objective but maintain "
                "independent residency trajectories; this is not a full-horizon optimum."
            ),
        },
        "warnings": warnings,
    }


def _matrix_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    name: str,
    warnings: list[str],
    *,
    per_column: bool,
) -> dict[str, Any]:
    errors = predicted - actual
    pooled = {
        "value_count": int(actual.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson_correlation": _pearson(
            actual.reshape(-1), predicted.reshape(-1), name, warnings
        ),
    }
    result: dict[str, Any] = {"pooled": pooled}
    if per_column:
        result["per_factor"] = [
            {
                "learned_factor_id": factor_id,
                "mae": float(np.mean(np.abs(errors[:, factor_id]))),
                "rmse": float(
                    np.sqrt(np.mean(np.square(errors[:, factor_id])))
                ),
                "pearson_correlation": _pearson(
                    actual[:, factor_id],
                    predicted[:, factor_id],
                    f"{name} factor {factor_id}",
                    warnings,
                ),
            }
            for factor_id in range(actual.shape[1])
        ]
    return result


def _total_error_rows(
    windows: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> list[dict[str, int | float]]:
    actual_totals = np.sum(actual, axis=1)
    predicted_totals = np.sum(predicted, axis=1)
    return [
        {
            "target_window_id": int(window_id),
            "actual_total": float(actual_total),
            "predicted_total": float(predicted_total),
            "error": float(predicted_total - actual_total),
            "absolute_error": float(abs(predicted_total - actual_total)),
        }
        for window_id, actual_total, predicted_total in zip(
            windows, actual_totals, predicted_totals, strict=True
        )
    ]


def _pearson(
    left: np.ndarray,
    right: np.ndarray,
    name: str,
    warnings: list[str],
) -> float | None:
    if len(left) < 2:
        warnings.append(f"{name} correlation is undefined with fewer than two values")
        return None
    left_deviation = left - np.mean(left)
    right_deviation = right - np.mean(right)
    denominator = float(np.linalg.norm(left_deviation) * np.linalg.norm(right_deviation))
    if denominator == 0.0:
        warnings.append(f"{name} correlation is undefined because an input is constant")
        return None
    return min(1.0, max(-1.0, float(np.dot(left_deviation, right_deviation) / denominator)))


def _aggregate_windows(
    replay: PolicyReplayResult,
    window_ids: Sequence[int],
) -> dict[str, dict[str, int | float]]:
    index = {int(window): position for position, window in enumerate(replay.test_window_ids)}
    positions = [index[int(window)] for window in window_ids]
    result: dict[str, dict[str, int | float]] = {}
    for policy_index, policy_name in enumerate(replay.policy_names):
        result[policy_name] = {
            "access_cost": float(
                math.fsum(replay.per_window["access_cost"][policy_index, positions])
            ),
            "promotion_cost": float(
                math.fsum(replay.per_window["promotion_cost"][policy_index, positions])
            ),
            "combined_cost": float(
                math.fsum(replay.per_window["combined_cost"][policy_index, positions])
            ),
            "hits": int(np.sum(replay.per_window["hit_count"][policy_index, positions])),
            "misses": int(np.sum(replay.per_window["miss_count"][policy_index, positions])),
        }
    return result


def _strict_less(
    left_name: str, left: float, right_name: str, right: float
) -> dict[str, Any]:
    difference = right - left
    return {
        "left_name": left_name,
        "left_value": left,
        "operator": "<",
        "right_name": right_name,
        "right_value": right,
        "right_minus_left": difference,
        "percentage_improvement_over_right": difference / right if right else None,
        "passed": bool(left < right),
    }


def _less_or_equal(
    left_name: str,
    left: float,
    right_name: str,
    right: float,
    tolerance: float,
) -> dict[str, Any]:
    difference = right - left
    return {
        "left_name": left_name,
        "left_value": left,
        "operator": "<=",
        "right_name": right_name,
        "right_value": right,
        "right_minus_left": difference,
        "percentage_improvement_over_right": difference / right if right else None,
        "absolute_tolerance": tolerance,
        "passed": bool(left <= right + tolerance),
    }
