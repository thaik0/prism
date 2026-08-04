"""Horizon-aligned simulation and predictive-actionability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from prism.experiments.actionability import CommonWindowProtocol
from prism.simulation.config import SimulationConfig
from prism.simulation.controllers import greedy_placement, record_benefits
from prism.simulation.diagnostics import evaluate_causal_diagnostics
from prism.simulation.evaluate import evaluate_policy_results
from prism.simulation.persistence import _write_policy_traces
from prism.simulation.projection import (
    ProjectionResult,
    build_record_demand_variants,
    fit_record_demand_projection,
)
from prism.simulation.replay import POLICY_DISPLAY_NAMES, PolicyReplayResult, run_policy_replay
from prism.workload.models import ObservableEvent


REJECTION_CODES = {
    "nonpositive_gross_savings": 0,
    "movement_cost": 1,
    "capacity": 2,
    "selected": 3,
    "oversized": 4,
}


@dataclass(frozen=True)
class HorizonSimulationRun:
    projection: ProjectionResult
    replay: PolicyReplayResult
    simulation_report: dict[str, Any]
    actionability_report: dict[str, Any]
    promotion_episodes: tuple[dict[str, Any], ...]


def run_horizon_simulation(
    observable_events: Sequence[ObservableEvent],
    observable_demand: np.ndarray,
    record_ids: Sequence[int],
    record_sizes: Sequence[int],
    membership_matrix: np.ndarray,
    factor_ids: Sequence[int],
    activation_probability: np.ndarray,
    conditional_intensity: np.ndarray,
    prediction_available: np.ndarray,
    config: SimulationConfig,
    protocol: CommonWindowProtocol,
    horizon: int,
    hidden_ground_truth: Mapping[str, Any],
    output_dir: str | Path,
    actionability_dir: str | Path,
    *,
    regime: str,
    seed: int,
) -> HorizonSimulationRun:
    """Fit cumulative projection, replay aligned policies, diagnose, and persist."""

    X = np.asarray(observable_demand, dtype=np.float64)
    available = np.asarray(prediction_available, dtype=np.bool_)
    training_targets = np.flatnonzero(
        available & (np.arange(len(X)) < protocol.train_end)
    )
    projection = fit_record_demand_projection(
        X,
        membership_matrix,
        activation_probability,
        conditional_intensity,
        available,
        protocol.train_end,
        forecast_horizon_windows=horizon,
        training_target_window_ids=training_targets,
    )
    variants = build_record_demand_variants(
        projection,
        activation_probability,
        conditional_intensity,
        available,
    )
    replay = run_policy_replay(
        observable_events,
        X,
        record_ids,
        record_sizes,
        projection.predicted_record_demand,
        available,
        config,
        protocol.validation_start,
        protocol.test_start,
        forecast_variants=variants,
        forecast_horizon_windows=horizon,
        validation_decision_end=protocol.validation_evaluation_end,
        test_evaluation_end=protocol.test_evaluation_end,
    )
    bursts = hidden_ground_truth.get("bursts")
    if not isinstance(bursts, list):
        raise ValueError("hidden bursts are required for actionability evaluation")
    policy_evaluation = evaluate_policy_results(replay, bursts)
    causal = evaluate_causal_diagnostics(
        replay,
        X,
        record_ids,
        record_sizes,
        hidden_ground_truth,
        config,
    )
    diagnostics, arrays, promotion_episodes = _actionability_diagnostics(
        X,
        np.asarray(record_ids, dtype=np.int64),
        np.asarray(record_sizes, dtype=np.int64),
        np.asarray(factor_ids, dtype=np.int64),
        projection,
        activation_probability,
        conditional_intensity,
        available,
        replay,
        config,
        protocol,
        horizon,
        hidden_ground_truth,
        regime,
        seed,
    )
    simulation_report = {
        "schema_version": 2,
        "forecast_horizon_windows": horizon,
        "common_eligible_window_protocol": protocol.to_dict(),
        "projection": {
            "coefficients": [
                {
                    "learned_factor_id": int(factor_id),
                    "a_recent_factor_demand": float(projection.model.factor_coefficients[i, 0]),
                    "b_activation_intensity": float(projection.model.factor_coefficients[i, 1]),
                    "c_constant": float(projection.model.factor_coefficients[i, 2]),
                    "residual_norm": float(projection.model.factor_residual_norms[i]),
                }
                for i, factor_id in enumerate(factor_ids)
            ],
            "training_target_window_ids": training_targets.tolist(),
        },
        "policy_metrics": replay.policy_metrics,
        "causal_diagnostics": causal,
        **policy_evaluation,
        "simulator_invariants": {
            "identical_observable_events": True,
            "independent_policy_state": True,
            "capacity_violations": replay.capacity_violations,
            "cumulative_controller_demand": True,
            "rolling_boundary_decisions": True,
            "carry_only_costs_excluded": True,
            "final_tail_costs_excluded": True,
        },
    }
    destination = Path(output_dir)
    action_destination = Path(actionability_dir)
    for path in (destination, action_destination):
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"output directory must be empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    _write_json(
        destination / "simulation_config.json",
        {
            "schema_version": 2,
            "forecast_horizon_windows": horizon,
            "resolved_simulation_configuration": config.to_dict(),
            "common_eligible_window_protocol": protocol.to_dict(),
            "policy_display_names": POLICY_DISPLAY_NAMES,
        },
    )
    np.savez(
        destination / "projection_model.npz",
        membership_matrix=projection.model.membership_matrix,
        record_ids=np.asarray(record_ids, dtype=np.int64),
        factor_ids=np.asarray(factor_ids, dtype=np.int64),
        factor_calibration_coefficients=projection.model.factor_coefficients,
        factor_calibration_residual_norms=projection.model.factor_residual_norms,
        residual_record_baseline=projection.model.residual_record_baseline,
        training_target_window_ids=projection.model.training_target_window_ids,
    )
    _write_policy_traces(destination / "policy_traces.npz", replay)
    _write_json(destination / "evaluation_report.json", simulation_report)
    np.savez(action_destination / "factor_diagnostics.npz", **arrays["factor"])
    np.savez(action_destination / "record_diagnostics.npz", **arrays["record"])
    np.savez(action_destination / "controller_diagnostics.npz", **arrays["controller"])
    _write_json(action_destination / "promotion_episodes.json", list(promotion_episodes))
    _write_json(action_destination / "actionability_report.json", diagnostics)
    return HorizonSimulationRun(
        projection,
        replay,
        simulation_report,
        diagnostics,
        promotion_episodes,
    )


def _actionability_diagnostics(
    X: np.ndarray,
    record_ids: np.ndarray,
    record_sizes: np.ndarray,
    factor_ids: np.ndarray,
    projection: ProjectionResult,
    probability: np.ndarray,
    intensity: np.ndarray,
    available: np.ndarray,
    replay: PolicyReplayResult,
    config: SimulationConfig,
    protocol: CommonWindowProtocol,
    horizon: int,
    hidden: Mapping[str, Any],
    regime: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], tuple[dict[str, Any], ...]]:
    windows = np.flatnonzero(
        available & (np.arange(len(available)) >= protocol.validation_start)
    )
    coefficients = projection.model.factor_coefficients
    current = projection.observable_factor_demand[windows - 1]
    continuation = current * coefficients[:, 0]
    activation = probability[windows] * intensity[windows] * coefficients[:, 1]
    intercept = np.broadcast_to(coefficients[:, 2], continuation.shape).copy()
    predicted = projection.predicted_factor_demand[windows]
    actual = projection.cumulative_future_factor_demand[windows]
    if actual is None:
        raise ValueError("cumulative factor targets are missing")
    factor_report = _factor_report(
        windows, current, continuation, activation, intercept, predicted, actual,
        protocol,
    )

    membership = projection.model.membership_matrix
    continuation_record = continuation @ membership
    activation_record = activation @ membership
    intercept_record = intercept @ membership
    residual_record = np.broadcast_to(
        projection.model.residual_record_baseline,
        (len(windows), len(record_ids)),
    ).copy()
    record_forecast = projection.predicted_record_demand[windows]
    record_report = _record_report(
        windows,
        record_forecast,
        continuation_record,
        activation_record,
        intercept_record,
        residual_record,
        record_ids,
        record_sizes,
        config,
    )

    controller_report, controller_arrays, targets_by_window = _controller_report(
        windows,
        projection.predicted_record_demand,
        record_ids,
        record_sizes,
        config,
        protocol,
    )
    _verify_predictive_targets(replay, targets_by_window, record_ids, protocol)
    oracle_report = _oracle_agreement(
        replay, X, record_ids, record_sizes, config, horizon
    )
    episodes = _promotion_episodes(
        targets_by_window,
        X,
        record_ids,
        record_sizes,
        config,
        protocol,
        horizon,
        hidden,
        regime,
        seed,
    )
    transition = _transition_coverage(
        replay, X, record_ids, hidden, horizon
    )
    report = {
        "schema_version": 1,
        "regime": regime,
        "forecast_horizon_windows": horizon,
        "seed": seed,
        "factor_forecast_movement": factor_report,
        "record_projection_movement": record_report,
        "controller_actionability": controller_report,
        "matched_horizon_oracle_agreement": oracle_report,
        "promotion_repayment": _promotion_summary(episodes),
        "transition_coverage": transition,
        "rejection_reason_codes": REJECTION_CODES,
        "numerical_tolerance": 1e-9,
        "hidden_truth_usage": "evaluation-only transition associations",
    }
    arrays = {
        "factor": {
            "target_window_ids": windows.astype(np.int64),
            "factor_ids": factor_ids,
            "current_factor_demand": current,
            "continuation_contribution": continuation,
            "activation_intensity_contribution": activation,
            "factor_intercept_contribution": intercept,
            "predicted_cumulative_factor_demand": predicted,
            "actual_cumulative_factor_demand": actual,
            "oracle_cumulative_factor_demand": actual,
        },
        "record": {
            "target_window_ids": windows.astype(np.int64),
            "record_ids": record_ids,
            "continuation_record_contribution": continuation_record,
            "activation_intensity_record_contribution": activation_record,
            "factor_intercept_record_contribution": intercept_record,
            "residual_record_baseline_contribution": residual_record,
            "predicted_cumulative_record_demand": record_forecast,
        },
        "controller": controller_arrays,
    }
    return report, arrays, episodes


def _factor_report(
    windows: np.ndarray,
    current: np.ndarray,
    continuation: np.ndarray,
    activation: np.ndarray,
    intercept: np.ndarray,
    predicted: np.ndarray,
    actual: np.ndarray,
    protocol: CommonWindowProtocol,
) -> dict[str, Any]:
    rows = {}
    for period, start, end in (
        ("validation", protocol.validation_start, protocol.validation_evaluation_end),
        ("test", protocol.test_start, protocol.test_evaluation_end),
    ):
        mask = (windows >= start) & (windows < end)
        error = predicted[mask] - actual[mask]
        changes = np.abs(np.diff(predicted[mask], axis=0))
        totals = [
            float(np.sum(component[mask]))
            for component in (continuation, activation, intercept)
        ]
        denominator = math.fsum(totals)
        rows[period] = {
            "target_window_count": int(np.sum(mask)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "mean_absolute_consecutive_change": float(np.mean(changes)) if changes.size else None,
            "median_absolute_consecutive_change": float(np.median(changes)) if changes.size else None,
            "fraction_changing_beyond_tolerance": float(np.mean(changes > 1e-9)) if changes.size else None,
            "contribution_shares": {
                "continuation": totals[0] / denominator if denominator else None,
                "activation_intensity": totals[1] / denominator if denominator else None,
                "intercept": totals[2] / denominator if denominator else None,
            },
            "per_factor": [
                {
                    "factor_id": factor,
                    "mae": float(np.mean(np.abs(error[:, factor]))),
                    "rmse": float(np.sqrt(np.mean(error[:, factor] ** 2))),
                }
                for factor in range(predicted.shape[1])
            ],
        }
    return rows


def _record_report(
    windows: np.ndarray,
    forecast: np.ndarray,
    continuation: np.ndarray,
    activation: np.ndarray,
    intercept: np.ndarray,
    residual: np.ndarray,
    record_ids: np.ndarray,
    record_sizes: np.ndarray,
    config: SimulationConfig,
) -> dict[str, Any]:
    rank_rows = []
    previous_ranks = None
    previous_candidates: set[int] | None = None
    component_rows = []
    for position, window in enumerate(windows):
        components = (continuation[position], activation[position], intercept[position], residual[position])
        totals = [float(np.sum(values)) for values in components]
        total = float(np.sum(forecast[position]))
        component_rows.append({
            "window_id": int(window),
            "shares": {
                name: value / total if total else None
                for name, value in zip(
                    ("continuation", "activation_intensity", "factor_intercept", "residual_baseline"),
                    totals,
                    strict=True,
                )
            } if total else None,
            "null_share_reason": None if total else "total_projected_demand_is_zero",
        })
        ranks = rankdata(-forecast[position], method="average")
        candidates = _diagnostic_candidate_set(
            forecast[position], record_ids, record_sizes, config
        )
        if previous_ranks is not None and previous_candidates is not None:
            correlation = spearmanr(previous_ranks, ranks).statistic
            additions = sorted(candidates - previous_candidates)
            removals = sorted(previous_candidates - candidates)
            union = candidates | previous_candidates
            rank_rows.append({
                "window_id": int(window),
                "spearman_rank_correlation": float(correlation) if np.isfinite(correlation) else None,
                "mean_normalized_absolute_rank_change": float(np.mean(np.abs(ranks - previous_ranks) / max(1, len(record_ids) - 1))),
                "median_normalized_absolute_rank_change": float(np.median(np.abs(ranks - previous_ranks) / max(1, len(record_ids) - 1))),
                "top_capacity_candidate_jaccard": len(candidates & previous_candidates) / len(union) if union else 1.0,
                "candidate_additions": additions,
                "candidate_removals": removals,
            })
        previous_ranks = ranks
        previous_candidates = candidates
    return {
        "component_shares_by_window": component_rows,
        "rank_turnover_by_consecutive_window": rank_rows,
        "top_capacity_candidate_definition": "gross-benefit-density order, ignoring residency and promotion cost, greedily filled by bytes",
        "summary": {
            "mean_spearman": _mean_optional(row["spearman_rank_correlation"] for row in rank_rows),
            "mean_normalized_rank_change": _mean_optional(row["mean_normalized_absolute_rank_change"] for row in rank_rows),
            "mean_candidate_jaccard": _mean_optional(row["top_capacity_candidate_jaccard"] for row in rank_rows),
        },
    }


def _diagnostic_candidate_set(
    forecast: np.ndarray,
    record_ids: np.ndarray,
    record_sizes: np.ndarray,
    config: SimulationConfig,
) -> set[int]:
    savings = config.slow_read_cost - config.fast_read_cost
    gross = forecast * savings
    ordered = sorted(
        range(len(record_ids)),
        key=lambda index: (
            -gross[index] / record_sizes[index],
            -gross[index],
            int(record_sizes[index]),
            int(record_ids[index]),
        ),
    )
    selected: set[int] = set()
    used = 0
    for index in ordered:
        size = int(record_sizes[index])
        if gross[index] > 0 and used + size <= config.fast_capacity_bytes:
            selected.add(int(record_ids[index]))
            used += size
    return selected


def _controller_report(
    windows: np.ndarray,
    forecasts: np.ndarray,
    record_ids: np.ndarray,
    record_sizes: np.ndarray,
    config: SimulationConfig,
    protocol: CommonWindowProtocol,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[int, set[int]]]:
    size_map = dict(zip(record_ids.tolist(), record_sizes.tolist(), strict=True))
    resident: set[int] = set()
    targets: dict[int, set[int]] = {}
    gross_rows = []
    promotion_rows = []
    net_rows = []
    density_rows = []
    current_rows = []
    selected_rows = []
    rejection_rows = []
    reports = []
    previous: set[int] = set()
    for window in windows:
        window_id = int(window)
        if window_id == protocol.test_start:
            previous = set(resident)
        forecast = forecasts[window_id]
        benefits = record_benefits(
            forecast,
            record_ids,
            size_map,
            resident,
            config.fast_read_cost,
            config.slow_read_cost,
            config.promotion_cost_per_byte,
        )
        selection = greedy_placement(benefits, size_map, config.fast_capacity_bytes)
        target = set(selection.target_record_ids)
        gross = forecast * (config.slow_read_cost - config.fast_read_cost)
        promotion = np.asarray([
            0.0 if int(record_id) in resident else int(size) * config.promotion_cost_per_byte
            for record_id, size in zip(record_ids, record_sizes, strict=True)
        ])
        net = np.asarray([benefits[int(record_id)] for record_id in record_ids])
        density = net / record_sizes
        current_indicator = np.asarray([int(record_id) in resident for record_id in record_ids])
        selected_indicator = np.asarray([int(record_id) in target for record_id in record_ids])
        reasons = np.empty(len(record_ids), dtype=np.int8)
        for index, record_id in enumerate(record_ids):
            if record_sizes[index] > config.fast_capacity_bytes:
                reason = "oversized"
            elif gross[index] <= 0:
                reason = "nonpositive_gross_savings"
            elif not current_indicator[index] and net[index] <= 0:
                reason = "movement_cost"
            elif selected_indicator[index]:
                reason = "selected"
            else:
                reason = "capacity"
            reasons[index] = REJECTION_CODES[reason]
        added = target - resident
        removed = resident - target
        union = target | resident
        positive_rejected = (net > 0) & ~selected_indicator & (record_sizes <= config.fast_capacity_bytes)
        selected_positive = selected_indicator & (net > 0)
        nonresident_positive = (~current_indicator) & (net > 0)
        movement = reasons == REJECTION_CODES["movement_cost"]
        capacity = reasons == REJECTION_CODES["capacity"]
        reports.append({
            "window_id": window_id,
            "period": "training" if window_id < protocol.train_end else "validation" if window_id < protocol.test_start else "test",
            "profitable_resident_candidates": int(np.sum(current_indicator & (net > 0))),
            "profitable_nonresident_candidates": int(np.sum(nonresident_positive)),
            "selected_profitable_nonresident_candidates": int(np.sum(nonresident_positive & selected_indicator)),
            "unselected_profitable_nonresident_candidates": int(np.sum(nonresident_positive & ~selected_indicator)),
            "movement_cost_rejection": {
                "record_count": int(np.sum(movement)),
                "bytes": int(np.sum(record_sizes[movement])),
                "gross_expected_savings": float(np.sum(gross[movement])),
                "promotion_cost": float(np.sum(promotion[movement])),
                "foregone_net_value_before_movement_cost": float(np.sum(gross[movement])),
            },
            "capacity_rejection": {
                "record_count": int(np.sum(capacity)),
                "bytes": int(np.sum(record_sizes[capacity])),
                "total_positive_net_benefit": float(np.sum(net[capacity])),
                "maximum_rejected_net_benefit": _max_or_none(net[capacity]),
                "maximum_rejected_benefit_density": _max_or_none(density[capacity]),
            },
            "selection_margins": {
                "minimum_selected_net_benefit": _min_or_none(net[selected_positive]),
                "minimum_selected_benefit_density": _min_or_none(density[selected_positive]),
                "maximum_rejected_positive_net_benefit": _max_or_none(net[positive_rejected]),
                "maximum_rejected_positive_benefit_density": _max_or_none(density[positive_rejected]),
                "selected_versus_rejected_density_margin": (
                    _min_or_none(density[selected_positive]) - _max_or_none(density[positive_rejected])
                    if np.any(selected_positive) and np.any(positive_rejected) else None
                ),
                "smallest_positive_nonresident_net_benefit": _min_or_none(net[nonresident_positive]),
                "closest_nonresident_distance_to_zero": _min_or_none(np.abs(net[~current_indicator])),
            },
            "actions": {
                "target_set_changed": target != resident,
                "records_added": len(added),
                "records_removed": len(removed),
                "bytes_added": sum(size_map[item] for item in added),
                "bytes_removed": sum(size_map[item] for item in removed),
                "promotions": len(added),
                "evictions": len(removed),
                "consecutive_target_set_jaccard": len(target & resident) / len(union) if union else 1.0,
            },
        })
        gross_rows.append(gross)
        promotion_rows.append(promotion)
        net_rows.append(net)
        density_rows.append(density)
        current_rows.append(current_indicator)
        selected_rows.append(selected_indicator)
        rejection_rows.append(reasons)
        resident = target
        targets[window_id] = set(target)
    test_rows = [row for row in reports if row["period"] == "test"]
    report = {
        "per_window": reports,
        "test_summary": {
            "target_set_change_count": sum(row["actions"]["target_set_changed"] for row in test_rows),
            "promotion_count": sum(row["actions"]["promotions"] for row in test_rows),
            "movement_cost_rejected_record_count": sum(row["movement_cost_rejection"]["record_count"] for row in test_rows),
            "capacity_rejected_record_count": sum(row["capacity_rejection"]["record_count"] for row in test_rows),
        },
    }
    arrays = {
        "target_window_ids": windows.astype(np.int64),
        "record_ids": record_ids,
        "gross_expected_savings": np.stack(gross_rows),
        "promotion_cost": np.stack(promotion_rows),
        "net_benefit": np.stack(net_rows),
        "benefit_density": np.stack(density_rows),
        "current_residency": np.stack(current_rows),
        "selected_target": np.stack(selected_rows),
        "rejection_reason_code": np.stack(rejection_rows),
    }
    return report, arrays, targets


def _verify_predictive_targets(
    replay: PolicyReplayResult,
    targets: Mapping[int, set[int]],
    record_ids: np.ndarray,
    protocol: CommonWindowProtocol,
) -> None:
    index = replay.policy_names.index("predictive_greedy")
    for position, window in enumerate(replay.test_window_ids):
        expected = np.asarray([int(record_id) in targets[int(window)] for record_id in record_ids])
        if not np.array_equal(expected, replay.pre_window_resident_indicator[index, position]):
            raise ValueError("diagnostic controller changed predictive policy residency")


def _oracle_agreement(
    replay: PolicyReplayResult,
    X: np.ndarray,
    record_ids: np.ndarray,
    record_sizes: np.ndarray,
    config: SimulationConfig,
    horizon: int,
) -> dict[str, Any]:
    predictive_index = replay.policy_names.index("predictive_greedy")
    oracle_index = replay.policy_names.index("oracle_greedy")
    rows = []
    for position, window in enumerate(replay.test_window_ids):
        predictive = replay.pre_window_resident_indicator[predictive_index, position]
        oracle = replay.pre_window_resident_indicator[oracle_index, position]
        intersection = predictive & oracle
        union = predictive | oracle
        demand = np.sum(X[int(window) : int(window) + horizon], axis=0)
        predictive_covered = float(np.sum(demand[predictive]))
        oracle_covered = float(np.sum(demand[oracle]))
        total = float(np.sum(demand))
        missed = oracle & ~predictive
        rows.append({
            "window_id": int(window),
            "target_set_jaccard": int(np.sum(intersection)) / int(np.sum(union)) if np.any(union) else 1.0,
            "byte_weighted_residency_overlap": int(np.sum(record_sizes[intersection])) / int(np.sum(record_sizes[oracle])) if np.any(oracle) else 1.0,
            "predictive_selected_bytes": int(np.sum(record_sizes[predictive])),
            "oracle_selected_bytes": int(np.sum(record_sizes[oracle])),
            "intersection_bytes": int(np.sum(record_sizes[intersection])),
            "oracle_selected_records_missed": record_ids[missed].tolist(),
            "predictive_selected_records_absent_from_oracle": record_ids[predictive & ~oracle].tolist(),
            "predictive_actual_horizon_accesses_covered": predictive_covered,
            "oracle_actual_horizon_accesses_covered": oracle_covered,
            "oracle_demand_weighted_target_coverage": predictive_covered / oracle_covered if oracle_covered else None,
            "missed_oracle_access_demand": float(np.sum(demand[missed])),
            "estimated_additional_access_cost": float(np.sum(demand[missed])) * (config.slow_read_cost - config.fast_read_cost),
            "total_horizon_demand": total,
        })
    return {
        "per_window": rows,
        "summary": {
            "mean_target_jaccard": _mean_optional(row["target_set_jaccard"] for row in rows),
            "mean_byte_overlap": _mean_optional(row["byte_weighted_residency_overlap"] for row in rows),
            "total_missed_oracle_access_demand": math.fsum(row["missed_oracle_access_demand"] for row in rows),
            "estimated_additional_access_cost": math.fsum(row["estimated_additional_access_cost"] for row in rows),
        },
        "causal_caveat": "Path-dependent residency and promotion cost prevent exact additive causal regret.",
    }


def _promotion_episodes(
    targets: Mapping[int, set[int]],
    X: np.ndarray,
    record_ids: np.ndarray,
    record_sizes: np.ndarray,
    config: SimulationConfig,
    protocol: CommonWindowProtocol,
    horizon: int,
    hidden: Mapping[str, Any],
    regime: str,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    index = {int(record_id): i for i, record_id in enumerate(record_ids)}
    memberships = {
        int(row["working_set_id"]): {
            int(item["record_id"]): float(item["weight"])
            for item in row["members"]
        }
        for row in hidden["working_set_memberships"]
    }
    bursts = [dict(row) for row in hidden["bursts"]]
    previous = targets[protocol.validation_evaluation_end - 1]
    rows = []
    for window in range(protocol.test_start, protocol.test_evaluation_end):
        target = targets[window]
        for record_id in sorted(target - previous):
            stop = min(window + horizon, protocol.test_evaluation_end)
            for future in range(window + 1, stop):
                if record_id not in targets[future]:
                    stop = future
                    break
            access_windows = [
                future for future in range(window, stop)
                if X[future, index[record_id]] > 0
            ]
            qualifying = int(sum(X[future, index[record_id]] for future in range(window, stop)))
            promotion_cost = int(record_sizes[index[record_id]]) * config.promotion_cost_per_byte
            savings = qualifying * (config.slow_read_cost - config.fast_read_cost)
            associated = [
                burst for burst in bursts
                if window <= int(burst["start_window"]) < window + horizon
                and record_id in memberships[int(burst["working_set_id"])]
            ]
            current = [burst for burst in associated if int(burst["start_window"]) == window]
            later = [burst for burst in associated if int(burst["start_window"]) > window]
            rows.append({
                "regime": regime,
                "horizon": horizon,
                "seed": seed,
                "promotion_window": window,
                "record_id": record_id,
                "record_size": int(record_sizes[index[record_id]]),
                "promotion_cost": promotion_cost,
                "access_windows": access_windows,
                "qualifying_fast_access_count": qualifying,
                "realized_savings": savings,
                "realized_net_value": savings - promotion_cost,
                "repaid": savings + 1e-9 >= promotion_cost,
                "pre_demand_promotion": qualifying > 0,
                "planted_support_associations": sorted({int(burst["working_set_id"]) for burst in associated}),
                "supports_burst_starting_current_window": bool(current),
                "supports_burst_starting_later_within_horizon": bool(later),
                "receives_demand_from_associated_working_set": bool(associated and qualifying),
                "improves_membership_weighted_coverage": bool(associated and record_id not in previous),
                "improves_actual_horizon_demand_coverage": qualifying > 0,
                "accessed_in_first_target_window": bool(X[window, index[record_id]] > 0),
                "accessed_later_within_horizon": any(future > window for future in access_windows),
                "observation_end_exclusive": stop,
            })
        previous = target
    return tuple(rows)


def _promotion_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    useful = [row for row in rows if row["pre_demand_promotion"]]
    repaid = [row for row in useful if row["repaid"]]
    return {
        "total_test_promotions": len(rows),
        "pre_demand_promotions": len(useful),
        "repaid_promotions": len(repaid),
        "repayment_fraction": len(repaid) / len(useful) if useful else None,
        "total_promotion_cost": math.fsum(float(row["promotion_cost"]) for row in useful),
        "total_realized_savings": math.fsum(float(row["realized_savings"]) for row in useful),
        "aggregate_net_value": math.fsum(float(row["realized_net_value"]) for row in useful),
    }


def _transition_coverage(
    replay: PolicyReplayResult,
    X: np.ndarray,
    record_ids: np.ndarray,
    hidden: Mapping[str, Any],
    horizon: int,
) -> dict[str, Any]:
    memberships = {
        int(row["working_set_id"]): {int(item["record_id"]) for item in row["members"]}
        for row in hidden["working_set_memberships"]
    }
    record_index = {int(record_id): i for i, record_id in enumerate(record_ids)}
    policy_index = {name: i for i, name in enumerate(replay.policy_names)}
    window_position = {int(window): i for i, window in enumerate(replay.test_window_ids)}
    rows = []
    for burst in hidden["bursts"]:
        start = int(burst["start_window"])
        if start not in window_position:
            continue
        support_indices = np.asarray(
            [record_index[record] for record in memberships[int(burst["working_set_id"])]],
            dtype=np.int64,
        )
        actual = np.sum(X[start : start + horizon], axis=0)
        denominator = float(np.sum(actual[support_indices]))
        coverage = {}
        for policy in ("predictive_greedy", "validation_final_frozen"):
            resident = replay.pre_window_resident_indicator[
                policy_index[policy], window_position[start]
            ]
            numerator = float(np.sum(actual[support_indices][resident[support_indices]]))
            coverage[policy] = numerator / denominator if denominator else None
        rows.append({
            "burst_id": int(burst["burst_id"]),
            "start_window": start,
            "working_set_id": int(burst["working_set_id"]),
            "actual_supported_access_count": denominator,
            "coverage": coverage,
        })
    return {
        "per_burst": rows,
        "aggregate_by_policy": {
            policy: _weighted_coverage(rows, policy)
            for policy in ("predictive_greedy", "validation_final_frozen")
        },
    }


def _weighted_coverage(rows: Sequence[Mapping[str, Any]], policy: str) -> float | None:
    numerator = math.fsum(
        float(row["actual_supported_access_count"]) * float(row["coverage"][policy])
        for row in rows if row["coverage"][policy] is not None
    )
    denominator = math.fsum(
        float(row["actual_supported_access_count"])
        for row in rows if row["coverage"][policy] is not None
    )
    return numerator / denominator if denominator else None


def _mean_optional(values: Sequence[float | None] | Any) -> float | None:
    resolved = [float(value) for value in values if value is not None]
    return math.fsum(resolved) / len(resolved) if resolved else None


def _max_or_none(values: np.ndarray) -> float | None:
    return float(np.max(values)) if values.size else None


def _min_or_none(values: np.ndarray) -> float | None:
    return float(np.min(values)) if values.size else None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
