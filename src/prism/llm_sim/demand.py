"""Policy-independent logical prefix-block demand construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .catalog import BlockCatalog
from .config import RequestSplit


class LogicalDemandError(ValueError):
    """Raised when logical demand would leak policy outcomes or future data."""


@dataclass(frozen=True)
class LogicalDemand:
    block_ids: tuple[str, ...]
    request_ids: np.ndarray
    demand_matrix: np.ndarray
    ordered_block_indices: tuple[tuple[int, ...], ...]
    logical_stream_sha256: str
    training_seen: np.ndarray
    cold_start: np.ndarray

    def __post_init__(self) -> None:
        for name in ("request_ids", "demand_matrix", "training_seen", "cold_start"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def summary(self, split: RequestSplit) -> dict[str, int | float | str]:
        test = self.demand_matrix[split.validation_end : split.request_count]
        seen_refs = int(test[:, self.training_seen].sum())
        cold_refs = int(test[:, self.cold_start].sum())
        total = seen_refs + cold_refs
        return {
            "request_count": int(len(self.request_ids)),
            "block_count": len(self.block_ids),
            "training_seen_blocks": int(self.training_seen.sum()),
            "cold_start_blocks": int(self.cold_start.sum()),
            "test_training_seen_references": seen_refs,
            "test_cold_start_references": cold_refs,
            "test_training_seen_share": seen_refs / total if total else 0.0,
            "test_cold_start_share": cold_refs / total if total else 0.0,
            "logical_stream_sha256": self.logical_stream_sha256,
        }


def build_logical_demand(catalog: BlockCatalog, split: RequestSplit) -> LogicalDemand:
    if len(catalog.requests) != split.request_count:
        raise LogicalDemandError(
            f"trace has {len(catalog.requests)} requests, expected {split.request_count}"
        )
    block_ids = tuple(block.block_id for block in catalog.blocks)
    block_index = {block_id: index for index, block_id in enumerate(block_ids)}
    matrix = np.zeros((split.request_count, len(block_ids)), dtype=np.int64)
    ordered_indices: list[tuple[int, ...]] = []
    digest = hashlib.sha256()
    for request in catalog.requests:
        indices = tuple(block_index[block_id] for block_id in request.ordered_block_ids)
        ordered_indices.append(indices)
        for index in indices:
            matrix[request.request_id, index] += 1
        canonical = {
            "arrival_time_ns": request.arrival_time_ns,
            "ordered_block_ids": list(request.ordered_block_ids),
            "request_id": request.request_id,
        }
        digest.update(
            (json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    training_seen = matrix[: split.training_end].sum(axis=0) > 0
    cold_start = ~training_seen
    return LogicalDemand(
        block_ids=block_ids,
        request_ids=np.arange(split.request_count, dtype=np.int64),
        demand_matrix=matrix,
        ordered_block_indices=tuple(ordered_indices),
        logical_stream_sha256=digest.hexdigest(),
        training_seen=training_seen,
        cold_start=cold_start,
    )
