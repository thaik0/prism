"""Narrow adapter over LLMServingSim's native radix-cache lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .catalog import (
    BlockCatalog,
    CatalogError,
    block_id_for_token_path,
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
    token_path: tuple[int, ...]
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
        while True:
            victim = self._next_victim_leaf(cache, keep)
            if victim is None:
                break
            if victim.lock_ref > 0:
                raise AdapterError("attempted to evict a pinned radix leaf")
            structural_tokens += len(victim.key)
            cache._delete_leaf(victim)
            cache._record_remove_event(victim)
        after = set(self._known_pages(cache))
        return before - after, structural_tokens

    def _next_victim_leaf(self, cache: Any, keep: set[str]) -> Any | None:
        candidates: list[tuple[tuple[int, ...], Any]] = []
        for leaf in cache._collect_leaves():
            if leaf is cache.root_node or leaf.lock_ref > 0:
                continue
            token_path = self._node_path(leaf)
            page_ids = self._known_ids_along_path(token_path)
            if not page_ids or all(block_id in keep for block_id in page_ids):
                continue
            # Prefix-closed targets mean every page following the first dropped
            # page may be removed. Split at the exact retained page boundary so
            # native leaf deletion never removes a retained ancestor.
            retained_pages = 0
            for block_id in page_ids:
                if block_id not in keep:
                    break
                retained_pages += 1
            retain_tokens = retained_pages * self.catalog.block_size_tokens
            parent_tokens = len(token_path) - len(leaf.key)
            if retain_tokens > parent_tokens:
                cache.match_prefix(list(token_path[:retain_tokens]))
                return self._next_victim_leaf(cache, keep)
            candidates.append((token_path, leaf))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _known_pages(self, cache: Any | None) -> dict[str, _NativePage]:
        if cache is None:
            return {}
        result: dict[str, _NativePage] = {}
        stack: list[tuple[Any, tuple[int, ...]]] = [(cache.root_node, ())]
        while stack:
            node, parent_path = stack.pop()
            node_path = parent_path + tuple(node.key or ())
            pinned = node is not cache.root_node and node.lock_ref > 0
            complete = len(node_path) // self.catalog.block_size_tokens
            for page_count in range(1, complete + 1):
                path = node_path[: page_count * self.catalog.block_size_tokens]
                block_id = block_id_for_token_path(
                    self.catalog.namespace, path, self.catalog.block_size_tokens
                )
                if block_id in self._known:
                    previous = result.get(block_id)
                    page = _NativePage(block_id, path, node, pinned)
                    if previous is None or (page.pinned and not previous.pinned):
                        result[block_id] = page
            for child in sorted(
                node.children.values(), key=lambda value: tuple(value.key or ()), reverse=True
            ):
                stack.append((child, node_path))
        return result

    def _known_ids_along_path(self, token_path: tuple[int, ...]) -> tuple[str, ...]:
        result: list[str] = []
        block = self.catalog.block_size_tokens
        for end in range(block, len(token_path) + 1, block):
            block_id = block_id_for_token_path(
                self.catalog.namespace, token_path[:end], block
            )
            if block_id in self._known:
                result.append(block_id)
        return tuple(result)

    @staticmethod
    def _node_path(node: Any) -> tuple[int, ...]:
        parts: list[tuple[int, ...]] = []
        current = node
        while current is not None:
            parts.append(tuple(current.key or ()))
            current = current.parent
        return tuple(token for part in reversed(parts) for token in part)
