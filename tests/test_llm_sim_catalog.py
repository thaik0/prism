from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from prism.llm_sim.catalog import (
    CatalogError,
    ancestors,
    build_block_catalog,
    validate_prefix_closed,
)
from prism.llm_sim.config import IntegrationConfig, RequestSplit
from prism.llm_sim.demand import LogicalDemandError, build_logical_demand


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/milestone8_llmservingsim.json"


def _write_trace(path: Path) -> None:
    rows = [
        {
            "input_toks": 32,
            "output_toks": 2,
            "arrival_time_ns": 10,
            "input_tok_ids": list(range(32)),
            "output_tok_ids": [100, 101],
        },
        {
            "input_toks": 32,
            "output_toks": 2,
            "arrival_time_ns": 20,
            "input_tok_ids": list(range(16)) + list(range(40, 56)),
            "output_tok_ids": [102, 103],
        },
        {
            "input_toks": 16,
            "output_toks": 1,
            "arrival_time_ns": 30,
            "input_tok_ids": list(range(100, 116)),
            "output_tok_ids": [104],
        },
        {
            "input_toks": 8,
            "output_toks": 1,
            "arrival_time_ns": 40,
            "input_tok_ids": list(range(8)),
            "output_tok_ids": [105],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_frozen_config_resolves_simulator_memory_budget() -> None:
    config = IntegrationConfig.from_json(CONFIG_PATH)
    budget = config.resolve_budget()

    assert config.policy_ids == (
        "llmservingsim_lru",
        "lfu",
        "training_popularity_static_prism",
        "validation_final_frozen_prism",
        "predictive_greedy_prism",
        "oracle_greedy",
    )
    assert budget.total_gpu_bytes == 96 * 1024**3
    assert budget.weight_bytes == 16_060_522_496
    assert budget.kv_bytes_per_token == 131_072
    assert budget.block_bytes == 2_097_152
    assert budget.post_weight_blocks == 41_493
    assert budget.remaining_blocks == 39_445
    assert budget.reusable_gpu_blocks == 9_861
    assert budget.host_capacity_blocks >= 14_722


def test_catalog_uses_stable_native_hash_path_identity(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    catalog = build_block_catalog(
        trace, namespace="model|bf16|b16", block_size_tokens=16, block_bytes=200
    )

    assert len(catalog.requests) == 4
    assert len(catalog.blocks) == 4
    first_request = catalog.requests[0]
    second_request = catalog.requests[1]
    assert first_request.ordered_block_ids[0] == second_request.ordered_block_ids[0]
    assert first_request.ordered_block_ids[1] != second_request.ordered_block_ids[1]
    child = first_request.ordered_block_ids[1]
    assert ancestors(child, catalog) == (first_request.ordered_block_ids[0],)
    assert catalog.by_id[first_request.ordered_block_ids[0]].native_block_hash == hash(
        tuple(range(16))
    )


def test_native_integer_tuple_hash_is_seed_independent() -> None:
    code = "print(hash(tuple(range(16))))"
    values = []
    for seed in ("1", "987654"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            capture_output=True,
            env={"PYTHONHASHSEED": seed},
        )
        values.append(result.stdout.strip())
    assert values[0] == values[1]


def test_metadata_only_catalog_does_not_store_frequency(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    catalog = build_block_catalog(
        trace, namespace="n", block_size_tokens=16, block_bytes=200
    )
    serialized = catalog.to_dict()

    assert "frequency" not in json.dumps(serialized)
    assert "arrival_time_ns" not in json.dumps(serialized)
    assert serialized["blocks"][0].keys() == {
        "block_id",
        "native_block_hash",
        "parent_block_id",
        "position",
        "token_count",
        "byte_size",
        "token_ids",
    }


def test_logical_demand_is_ordered_policy_independent_and_classifies_cold_start(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    catalog = build_block_catalog(
        trace, namespace="n", block_size_tokens=16, block_bytes=200
    )
    split = RequestSplit(training_end=2, validation_end=3, request_count=4)
    first = build_logical_demand(catalog, split)
    second = build_logical_demand(catalog, split)

    np.testing.assert_array_equal(first.demand_matrix, second.demand_matrix)
    assert first.ordered_block_indices == second.ordered_block_indices
    assert first.logical_stream_sha256 == second.logical_stream_sha256
    assert int(first.training_seen.sum()) == 3
    assert int(first.cold_start.sum()) == 1
    assert first.demand_matrix[3].sum() == 0


def test_prefix_closed_target_is_enforced(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    catalog = build_block_catalog(
        trace, namespace="n", block_size_tokens=16, block_bytes=200
    )
    parent, child = catalog.requests[0].ordered_block_ids

    assert validate_prefix_closed((child, parent), catalog) == tuple(sorted((parent, child)))
    with pytest.raises(CatalogError, match="not prefix-closed"):
        validate_prefix_closed((child,), catalog)
    with pytest.raises(CatalogError, match="unknown"):
        validate_prefix_closed(("missing",), catalog)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda row: row.pop("input_tok_ids"), "must contain input_tok_ids"),
        (lambda row: row.update(input_toks=31), "matching input_toks"),
        (lambda row: row.update(arrival_time_ns=-1), "must be nonnegative"),
    ],
)
def test_catalog_rejects_malformed_trace(tmp_path: Path, mutation, message: str) -> None:
    row = {
        "input_toks": 32,
        "output_toks": 2,
        "arrival_time_ns": 10,
        "input_tok_ids": list(range(32)),
        "output_tok_ids": [100, 101],
    }
    mutation(row)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(CatalogError, match=message):
        build_block_catalog(
            trace, namespace="n", block_size_tokens=16, block_bytes=200
        )


def test_split_length_mismatch_is_rejected(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    catalog = build_block_catalog(
        trace, namespace="n", block_size_tokens=16, block_bytes=200
    )
    with pytest.raises(LogicalDemandError, match="expected 5"):
        build_logical_demand(catalog, RequestSplit(2, 4, 5))
