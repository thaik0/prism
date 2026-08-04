from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prism.llm_sim.catalog import build_block_catalog
from prism.llm_sim.config import IntegrationConfig, RequestSplit, ResolvedBudget
from prism.llm_sim.demand import LogicalDemand, build_logical_demand
from prism.llm_sim.model import DemandForecasts, fit_demand_forecasts
from prism.llm_sim.policies import PolicyError, PolicyRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = IntegrationConfig.from_json(
    PROJECT_ROOT / "configs/milestone8_llmservingsim.json"
)


def _small_inputs(tmp_path: Path):
    a = list(range(16))
    b = list(range(16, 32))
    c = list(range(32, 48))
    rows = [
        (a + b, 0),
        (a, 1),
        (a + c, 2),
        (a + b, 3),
        (a + c, 4),
        (a + b, 5),
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "input_toks": len(tokens),
                    "output_toks": 1,
                    "arrival_time_ns": arrival,
                    "input_tok_ids": tokens,
                    "output_tok_ids": [100 + arrival],
                }
            )
            + "\n"
            for tokens, arrival in rows
        ),
        encoding="utf-8",
    )
    budget = ResolvedBudget(
        total_gpu_bytes=1000,
        weight_bytes=100,
        kv_bytes_per_token=1,
        block_size_tokens=16,
        block_bytes=16,
        post_weight_blocks=56,
        active_request_reservation_blocks=1,
        remaining_blocks=55,
        reusable_gpu_blocks=2,
        host_capacity_bytes=1600,
        host_capacity_blocks=100,
    )
    catalog = build_block_catalog(
        trace,
        namespace="test",
        block_size_tokens=16,
        block_bytes=budget.block_bytes,
    )
    split = RequestSplit(2, 4, 6)
    demand = build_logical_demand(catalog, split)
    prediction = demand.demand_matrix.astype(np.float64)
    forecasts = DemandForecasts(
        predicted_record_demand=prediction,
        prediction_available=np.ones(6, dtype=np.bool_),
        learned_membership=np.zeros((4, len(demand.block_ids))),
        training_seen_indices=np.flatnonzero(demand.training_seen),
        activation_threshold=np.zeros(4),
        fit_status="fixture",
        structure_converged=True,
        structure_iterations=1,
    )
    return catalog, demand, forecasts, split, budget


def _run_to(
    runtime: PolicyRuntime,
    demand: LogicalDemand,
    request_id: int,
    *,
    eligible: set[str],
    resident: set[str],
):
    decision = None
    for current in range(request_id + 1):
        decision = runtime.before_request(
            current, eligible_block_ids=eligible, resident_block_ids=resident
        )
        runtime.after_request(current)
        eligible.update(
            demand.block_ids[index] for index in demand.ordered_block_indices[current]
        )
        if decision.target_block_ids is not None:
            resident.clear()
            resident.update(decision.target_block_ids)
    assert decision is not None
    return decision


def test_full_model_fit_is_training_only_and_cold_start_zero() -> None:
    budget = CONFIG.resolve_budget()
    catalog = build_block_catalog(
        CONFIG.trace_path(tiny=False),
        namespace=f"{CONFIG.model}|{CONFIG.dtype}|b{CONFIG.block_size_tokens}",
        block_size_tokens=CONFIG.block_size_tokens,
        block_bytes=budget.block_bytes,
    )
    demand = build_logical_demand(catalog, CONFIG.full_split)
    first = fit_demand_forecasts(demand, CONFIG.full_split, CONFIG)

    changed_matrix = np.array(demand.demand_matrix, copy=True)
    changed_matrix[CONFIG.full_split.validation_end :] *= 7
    changed = LogicalDemand(
        block_ids=demand.block_ids,
        request_ids=demand.request_ids,
        demand_matrix=changed_matrix,
        ordered_block_indices=demand.ordered_block_indices,
        logical_stream_sha256=demand.logical_stream_sha256,
        training_seen=demand.training_seen,
        cold_start=demand.cold_start,
    )
    second = fit_demand_forecasts(changed, CONFIG.full_split, CONFIG)

    assert first.fit_status == "fitted_training_only"
    assert first.structure_converged
    np.testing.assert_array_equal(first.learned_membership, second.learned_membership)
    np.testing.assert_allclose(
        first.predicted_record_demand[CONFIG.full_split.validation_end],
        second.predicted_record_demand[CONFIG.full_split.validation_end],
    )
    assert np.all(first.predicted_record_demand[:, demand.cold_start] == 0.0)


def test_tiny_degenerate_fit_is_explicit_not_hidden() -> None:
    budget = CONFIG.resolve_budget()
    catalog = build_block_catalog(
        CONFIG.trace_path(tiny=True),
        namespace="tiny",
        block_size_tokens=CONFIG.block_size_tokens,
        block_bytes=budget.block_bytes,
    )
    demand = build_logical_demand(catalog, CONFIG.tiny_split)
    forecasts = fit_demand_forecasts(
        demand, CONFIG.tiny_split, CONFIG, allow_degenerate_smoke=True
    )
    assert forecasts.fit_status in {
        "fitted_training_only",
        "degenerate_tiny_smoke_no_predictive_target",
    }
    if forecasts.fit_status.startswith("degenerate"):
        assert not forecasts.prediction_available.any()


def test_native_lru_adapter_path_never_supplies_a_target(tmp_path: Path) -> None:
    catalog, demand, forecasts, split, budget = _small_inputs(tmp_path)
    runtime = PolicyRuntime(
        "llmservingsim_lru", catalog, demand, forecasts, split, budget, CONFIG
    )
    decision = _run_to(runtime, demand, 5, eligible=set(), resident=set())
    assert decision.target_block_ids is None
    assert not decision.revealed_current_request


def test_lfu_ties_and_capacity_retain_prefix_closure(tmp_path: Path) -> None:
    catalog, demand, forecasts, split, budget = _small_inputs(tmp_path)
    runtime = PolicyRuntime("lfu", catalog, demand, forecasts, split, budget, CONFIG)
    decision = _run_to(runtime, demand, 3, eligible=set(), resident=set())
    assert decision.target_block_ids is not None
    assert len(decision.target_block_ids) == 2
    selected = set(decision.target_block_ids)
    for block_id in selected:
        parent = catalog.by_id[block_id].parent_block_id
        assert parent is None or parent in selected


def test_static_and_validation_final_freeze_targets(tmp_path: Path) -> None:
    catalog, demand, forecasts, split, budget = _small_inputs(tmp_path)
    static = PolicyRuntime(
        "training_popularity_static_prism",
        catalog,
        demand,
        forecasts,
        split,
        budget,
        CONFIG,
    )
    eligible: set[str] = set()
    resident: set[str] = set()
    static_targets = []
    for request_id in range(6):
        decision = static.before_request(
            request_id, eligible_block_ids=eligible, resident_block_ids=resident
        )
        static.after_request(request_id)
        eligible.update(
            demand.block_ids[index] for index in demand.ordered_block_indices[request_id]
        )
        if decision.target_block_ids is not None:
            static_targets.append(decision.target_block_ids)
            resident = set(decision.target_block_ids)
    assert len(set(static_targets)) == 1

    frozen = PolicyRuntime(
        "validation_final_frozen_prism",
        catalog,
        demand,
        forecasts,
        split,
        budget,
        CONFIG,
    )
    eligible = set()
    resident = set()
    targets = []
    for request_id in range(6):
        decision = frozen.before_request(
            request_id, eligible_block_ids=eligible, resident_block_ids=resident
        )
        frozen.after_request(request_id)
        eligible.update(
            demand.block_ids[index] for index in demand.ordered_block_indices[request_id]
        )
        if decision.target_block_ids is not None:
            targets.append((request_id, decision.target_block_ids, decision.frozen))
            resident = set(decision.target_block_ids)
    assert targets[-1][1] == targets[-2][1]
    assert targets[-1][2]


def test_predictive_timing_and_oracle_isolation(tmp_path: Path) -> None:
    catalog, demand, forecasts, split, budget = _small_inputs(tmp_path)
    predictive = PolicyRuntime(
        "predictive_greedy_prism",
        catalog,
        demand,
        forecasts,
        split,
        budget,
        CONFIG,
    )
    predicted = _run_to(predictive, demand, 4, eligible=set(), resident=set())
    assert not predicted.revealed_current_request

    oracle = PolicyRuntime(
        "oracle_greedy", catalog, demand, forecasts, split, budget, CONFIG
    )
    oracle_decision = _run_to(oracle, demand, 4, eligible=set(), resident=set())
    assert oracle_decision.revealed_current_request


def test_policy_rejects_reveal_order_and_over_budget_runtime(tmp_path: Path) -> None:
    catalog, demand, forecasts, split, budget = _small_inputs(tmp_path)
    runtime = PolicyRuntime("lfu", catalog, demand, forecasts, split, budget, CONFIG)
    with pytest.raises(PolicyError, match="strict chronological"):
        runtime.before_request(1, eligible_block_ids=(), resident_block_ids=())
    all_ids = set(demand.block_ids)
    with pytest.raises(PolicyError, match="exceeds the fixed budget"):
        runtime.before_request(
            0, eligible_block_ids=all_ids, resident_block_ids=all_ids
        )
