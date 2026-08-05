"""Narrow adapter over LLMServingSim's native radix-cache lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Iterable

from .catalog import (
    BlockCatalog,
    CatalogError,
    validate_prefix_closed,
)
from .config import ResolvedBudget


class AdapterError(RuntimeError):
    """Raised when native cache state violates the Prism boundary."""


@dataclass(frozen=True)
class CacheSnapshot:
    resident_block_ids: tuple[str, ...]
    pinned_block_ids: tuple[str, ...]
    host_block_ids: tuple[str, ...]
    reusable_occupancy_blocks: int
    protected_native_tokens: int


@dataclass(frozen=True)
class PlacementOutcome:
    before: CacheSnapshot
    after: CacheSnapshot
    evicted_block_ids: tuple[str, ...]
    structurally_evicted_native_tokens: int


@dataclass(frozen=True)
class _NativePage:
    block_id: str
    node: Any
    pinned: bool


class LLMServingSimAdapter:
    """Apply desired reusable targets without managing active request memory.

    The adapter intentionally uses the pinned simulator's radix tree and event
    accounting. Its only private calls are leaf deletion/event emission because
    upstream exposes LRU eviction but no exact-target eviction hook.
    """

    def __init__(self, catalog: BlockCatalog, budget: ResolvedBudget):
        self.catalog = catalog
        self.budget = budget
        self._known = set(catalog.by_id)
        self._block_for_edge: dict[tuple[str | None, int], str] = {}
        for block in catalog.blocks:
            edge = (block.parent_block_id, block.native_block_hash)
            previous = self._block_for_edge.setdefault(edge, block.block_id)
            if previous != block.block_id:
                raise AdapterError("catalog has an ambiguous native prefix edge")

    def snapshot(self, memory: Any) -> CacheSnapshot:
        npu = self._known_pages(memory.npu_prefix_cache)
        host_cache = getattr(memory, "second_tier_prefix_cache", None)
        host = self._known_pages(host_cache) if host_cache is not None else {}
        resident = set(npu)
        pinned = {block_id for block_id, page in npu.items() if page.pinned}
        reusable = resident - pinned
        return CacheSnapshot(
            resident_block_ids=tuple(sorted(resident)),
            pinned_block_ids=tuple(sorted(pinned)),
            host_block_ids=tuple(sorted(host)),
            reusable_occupancy_blocks=len(reusable),
            protected_native_tokens=int(memory.npu_prefix_cache.protected_size()),
        )

    def apply_target(
        self,
        memory: Any,
        target_block_ids: Iterable[str],
    ) -> PlacementOutcome:
        """Prune reusable pages while leaving native demand loads untouched.

        A target block absent from GPU remains eligible for upstream's normal
        CPU prefix match. The unchanged scheduler charges and executes that
        host-to-GPU load, and its completion inserts the page into the NPU
        radix cache. A later lifecycle callback retains it iff it is targeted.
        """
        try:
            target = set(validate_prefix_closed(target_block_ids, self.catalog))
        except CatalogError as exc:
            raise AdapterError(str(exc)) from exc
        before = self.snapshot(memory)
        pinned = set(before.pinned_block_ids)
        if len(target - pinned) > self.budget.reusable_gpu_blocks:
            raise AdapterError("desired reusable target exceeds the reserved GPU budget")
        eligible = set(before.resident_block_ids) | set(before.host_block_ids)
        if target - eligible - pinned:
            raise AdapterError("desired target contains a block absent from both native tiers")

        unwanted = set(before.resident_block_ids) - pinned - target
        if not unwanted:
            return PlacementOutcome(
                before=before,
                after=before,
                evicted_block_ids=(),
                structurally_evicted_native_tokens=0,
            )

        evicted, structural_tokens = self._prune(memory, target | pinned)
        memory.apply_kv_cache_events()
        after = self.snapshot(memory)
        if set(after.pinned_block_ids) != pinned:
            raise AdapterError("placement changed the simulator's pinned block set")
        if after.reusable_occupancy_blocks > self.budget.reusable_gpu_blocks:
            raise AdapterError(
                "Prism-controlled reusable occupancy exceeds the reserved GPU budget"
            )
        return PlacementOutcome(
            before=before,
            after=after,
            evicted_block_ids=tuple(sorted(evicted)),
            structurally_evicted_native_tokens=structural_tokens,
        )

    def _prune(self, memory: Any, keep: set[str]) -> tuple[set[str], int]:
        cache = memory.npu_prefix_cache
        before = set(self._known_pages(cache))
        structural_tokens = 0
        heap: list[tuple[tuple[int, ...], int, Any]] = []
        push_order = 0
        while True:
            split_paths: list[tuple[int, ...]] = []
            heap.clear()
            for leaf in cache._collect_leaves():
                candidate = self._leaf_candidate(cache, leaf, keep)
                if candidate is None:
                    continue
                token_path, split_at = candidate
                if split_at is not None:
                    split_paths.append(token_path[:split_at])
                else:
                    heapq.heappush(heap, (token_path, push_order, leaf))
                    push_order += 1
            if split_paths:
                cache.match_prefix(list(min(split_paths)))
                continue
            if not heap:
                break
            while heap:
                _, _, victim = heapq.heappop(heap)
                if victim.parent is None or victim.children:
                    continue
                if victim.lock_ref > 0:
                    raise AdapterError("attempted to evict a pinned radix leaf")
                parent = victim.parent
                structural_tokens += len(victim.key)
                cache._delete_leaf(victim)
                cache._record_remove_event(victim)
                if parent is not cache.root_node and not parent.children:
                    candidate = self._leaf_candidate(cache, parent, keep)
                    if candidate is not None:
                        token_path, split_at = candidate
                        if split_at is not None:
                            raise AdapterError(
                                "retained prefix boundary was not split before pruning"
                            )
                        heapq.heappush(heap, (token_path, push_order, parent))
                        push_order += 1
            break
        after = set(self._known_pages(cache))
        if after - keep:
            raise AdapterError("native radix pruning left unwanted known blocks resident")
        return before - after, structural_tokens

    def _leaf_candidate(
        self, cache: Any, leaf: Any, keep: set[str]
    ) -> tuple[tuple[int, ...], int | None] | None:
        if leaf is cache.root_node or leaf.lock_ref > 0:
            return None
        token_path = self._node_path(leaf)
        page_ids = self._known_ids_along_path(token_path)
        if not page_ids or all(block_id in keep for block_id in page_ids):
            return None
        retained_pages = 0
        for block_id in page_ids:
            if block_id not in keep:
                break
            retained_pages += 1
        retain_tokens = retained_pages * self.catalog.block_size_tokens
        parent_tokens = len(token_path) - len(leaf.key)
        split_at = retain_tokens if retain_tokens > parent_tokens else None
        return token_path, split_at

    def _known_pages(self, cache: Any | None) -> dict[str, _NativePage]:
        if cache is None:
            return {}
        result: dict[str, _NativePage] = {}
        stack: list[tuple[Any, str | None, tuple[int, ...]]] = [
            (cache.root_node, None, ())
        ]
        while stack:
            node, parent_block_id, pending_tokens = stack.pop()
            unprocessed = pending_tokens + tuple(node.key or ())
            pinned = node is not cache.root_node and node.lock_ref > 0
            consumed = 0
            block_size = self.catalog.block_size_tokens
            path_is_known = True
            while consumed + block_size <= len(unprocessed):
                native_hash = hash(tuple(unprocessed[consumed : consumed + block_size]))
                block_id = self._block_for_edge.get((parent_block_id, native_hash))
                if block_id is None:
                    path_is_known = False
                    break
                previous = result.get(block_id)
                page = _NativePage(block_id, node, pinned)
                if previous is None or (page.pinned and not previous.pinned):
                    result[block_id] = page
                parent_block_id = block_id
                consumed += block_size
            if not path_is_known:
                continue
            next_pending = unprocessed[consumed:]
            for child in sorted(
                node.children.values(), key=lambda value: tuple(value.key or ()), reverse=True
            ):
                stack.append((child, parent_block_id, next_pending))
        return result

    def _known_ids_along_path(self, token_path: tuple[int, ...]) -> tuple[str, ...]:
        result: list[str] = []
        block = self.catalog.block_size_tokens
        parent_block_id: str | None = None
        for end in range(block, len(token_path) + 1, block):
            native_hash = hash(tuple(token_path[end - block : end]))
            block_id = self._block_for_edge.get(
                (parent_block_id, native_hash)
            )
            if block_id is None:
                break
            result.append(block_id)
            parent_block_id = block_id
        return tuple(result)

    @staticmethod
    def _node_path(node: Any) -> tuple[int, ...]:
        parts: list[tuple[int, ...]] = []
        current = node
        while current is not None:
            parts.append(tuple(current.key or ()))
            current = current.parent
        return tuple(token for part in reversed(parts) for token in part)
