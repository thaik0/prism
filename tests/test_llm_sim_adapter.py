from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism.llm_sim.adapter import AdapterError, LLMServingSimAdapter
from prism.llm_sim.catalog import build_block_catalog
from prism.llm_sim.config import ResolvedBudget


class _Node:
    def __init__(self, key=(), parent=None):
        self.key = list(key)
        self.parent = parent
        self.children = {}
        self.lock_ref = 0


class _PageCache:
    """Page-per-node radix fixture exposing the pinned upstream methods."""

    def __init__(self, block_size: int):
        self.page_size = block_size
        self.root_node = _Node()
        self.root_node.lock_ref = 1
        self.removed = []

    def insert(self, tokens):
        node = self.root_node
        for start in range(0, len(tokens), self.page_size):
            page = tuple(tokens[start : start + self.page_size])
            child = node.children.get(page)
            if child is None:
                child = _Node(page, node)
                node.children[page] = child
            node = child

    def match_prefix(self, tokens):
        del tokens

    def _collect_leaves(self):
        leaves = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            else:
                leaves.append(node)
        return leaves

    def _delete_leaf(self, node):
        key = next(key for key, value in node.parent.children.items() if value is node)
        del node.parent.children[key]

    def _record_remove_event(self, node):
        self.removed.append(tuple(node.key))

    def protected_size(self):
        return sum(
            len(node.key)
            for node in self._nodes()
            if node is not self.root_node and node.lock_ref > 0
        )

    def _nodes(self):
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())


class _Memory:
    def __init__(self, npu, host):
        self.npu_prefix_cache = npu
        self.second_tier_prefix_cache = host
        self.events_applied = 0

    def apply_kv_cache_events(self):
        self.events_applied += 1


def _fixture(tmp_path: Path, reusable_blocks: int = 2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    a = list(range(16))
    b = list(range(16, 32))
    c = list(range(32, 48))
    d = list(range(48, 64))
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "input_toks": len(tokens),
                    "output_toks": 1,
                    "arrival_time_ns": request_id,
                    "input_tok_ids": tokens,
                    "output_tok_ids": [100 + request_id],
                }
            )
            + "\n"
            for request_id, tokens in enumerate((a + b, a + c, d))
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
        reusable_gpu_blocks=reusable_blocks,
        host_capacity_bytes=1600,
        host_capacity_blocks=100,
    )
    catalog = build_block_catalog(
        trace, namespace="adapter-test", block_size_tokens=16, block_bytes=16
    )
    npu = _PageCache(16)
    host = _PageCache(16)
    for tokens in (a + b, a + c, d):
        npu.insert(tokens)
        host.insert(tokens)
    return catalog, budget, _Memory(npu, host)


def test_adapter_protects_pinned_pages_and_prunes_only_unpinned(tmp_path: Path) -> None:
    catalog, budget, memory = _fixture(tmp_path)
    first_path = catalog.requests[0].ordered_block_ids
    shared, pinned_leaf = first_path
    # Mirror native inc_lock_ref: the active leaf and every ancestor are pinned.
    pinned_node = memory.npu_prefix_cache.root_node.children[tuple(range(16))].children[
        tuple(range(16, 32))
    ]
    current = pinned_node
    while current is not memory.npu_prefix_cache.root_node:
        current.lock_ref += 1
        current = current.parent

    adapter = LLMServingSimAdapter(catalog, budget)
    before = adapter.snapshot(memory)
    assert set(before.pinned_block_ids) == {shared, pinned_leaf}
    outcome = adapter.apply_target(memory, first_path)

    assert set(outcome.after.pinned_block_ids) == {shared, pinned_leaf}
    assert not (set(outcome.evicted_block_ids) & set(before.pinned_block_ids))
    assert outcome.after.reusable_occupancy_blocks == 0
    assert memory.events_applied == 1


def test_adapter_enforces_prefix_closed_target_and_budget(tmp_path: Path) -> None:
    catalog, budget, memory = _fixture(tmp_path, reusable_blocks=1)
    adapter = LLMServingSimAdapter(catalog, budget)
    child = catalog.requests[0].ordered_block_ids[-1]
    with pytest.raises(AdapterError, match="prefix-closed"):
        adapter.apply_target(memory, [child])

    roots = [request.ordered_block_ids[0] for request in catalog.requests]
    with pytest.raises(AdapterError, match="exceeds"):
        adapter.apply_target(memory, roots)


def test_adapter_snapshot_and_pruning_are_deterministic(tmp_path: Path) -> None:
    catalog, budget, first_memory = _fixture(tmp_path / "first")
    _, _, second_memory = _fixture(tmp_path / "second")
    adapter = LLMServingSimAdapter(catalog, budget)
    target = catalog.requests[1].ordered_block_ids

    first = adapter.apply_target(first_memory, target)
    second = adapter.apply_target(second_memory, target)
    assert first == second
    assert first.after.reusable_occupancy_blocks <= budget.reusable_gpu_blocks
