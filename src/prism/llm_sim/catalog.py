"""Metadata-only mapping from prompt token pages to Prism block records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


class CatalogError(ValueError):
    """Raised when an upstream trace cannot define a stable block catalog."""


@dataclass(frozen=True)
class PrefixBlock:
    block_id: str
    native_block_hash: int
    parent_block_id: str | None
    position: int
    token_count: int
    byte_size: int
    token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "native_block_hash": self.native_block_hash,
            "parent_block_id": self.parent_block_id,
            "position": self.position,
            "token_count": self.token_count,
            "byte_size": self.byte_size,
            "token_ids": list(self.token_ids),
        }


@dataclass(frozen=True)
class TraceRequest:
    request_id: int
    arrival_time_ns: int
    input_tokens: int
    output_tokens: int
    ordered_block_ids: tuple[str, ...]


@dataclass(frozen=True)
class BlockCatalog:
    namespace: str
    block_size_tokens: int
    block_bytes: int
    blocks: tuple[PrefixBlock, ...]
    requests: tuple[TraceRequest, ...]

    @property
    def by_id(self) -> dict[str, PrefixBlock]:
        return {block.block_id: block for block in self.blocks}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "namespace": self.namespace,
            "block_size_tokens": self.block_size_tokens,
            "block_bytes": self.block_bytes,
            "block_count": len(self.blocks),
            "request_count": len(self.requests),
            "blocks": [block.to_dict() for block in self.blocks],
        }


def build_block_catalog(
    trace_path: str | Path,
    *,
    namespace: str,
    block_size_tokens: int,
    block_bytes: int,
) -> BlockCatalog:
    """Scan only identities and structural metadata, never future frequency."""
    if not namespace or block_size_tokens <= 0 or block_bytes <= 0:
        raise CatalogError("catalog namespace and block sizes must be positive")
    path = Path(trace_path)
    blocks: dict[str, PrefixBlock] = {}
    requests: list[TraceRequest] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CatalogError(f"cannot read trace: {exc}") from exc
    for request_id, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"invalid JSON on trace line {request_id + 1}") from exc
        if not isinstance(row, dict) or "sub_requests" in row:
            raise CatalogError("Milestone 8 primary/tiny traces must use flat requests")
        input_count = _positive_int(row, "input_toks", request_id)
        output_count = _positive_int(row, "output_toks", request_id)
        arrival = _nonnegative_int(row, "arrival_time_ns", request_id)
        tokens = row.get("input_tok_ids")
        if not isinstance(tokens, list) or len(tokens) != input_count:
            raise CatalogError(
                f"trace request {request_id} must contain input_tok_ids matching input_toks"
            )
        if any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
            raise CatalogError(f"trace request {request_id} has invalid token IDs")
        ordered: list[str] = []
        parent: str | None = None
        native_chain: list[int] = []
        full_length = len(tokens) // block_size_tokens * block_size_tokens
        for start in range(0, full_length, block_size_tokens):
            page = tuple(tokens[start : start + block_size_tokens])
            native_hash = hash(page)
            native_chain.append(native_hash)
            block_id = _path_block_id(namespace, native_chain)
            block = PrefixBlock(
                block_id=block_id,
                native_block_hash=native_hash,
                parent_block_id=parent,
                position=start // block_size_tokens,
                token_count=block_size_tokens,
                byte_size=block_bytes,
                token_ids=page,
            )
            existing = blocks.get(block_id)
            if existing is not None and existing != block:
                raise CatalogError(f"stable block identity collision for {block_id}")
            blocks[block_id] = block
            ordered.append(block_id)
            parent = block_id
        requests.append(
            TraceRequest(
                request_id=request_id,
                arrival_time_ns=arrival,
                input_tokens=input_count,
                output_tokens=output_count,
                ordered_block_ids=tuple(ordered),
            )
        )
    ordered_blocks = tuple(blocks[key] for key in sorted(blocks))
    return BlockCatalog(
        namespace=namespace,
        block_size_tokens=block_size_tokens,
        block_bytes=block_bytes,
        blocks=ordered_blocks,
        requests=tuple(requests),
    )


def ancestors(block_id: str, catalog: BlockCatalog) -> tuple[str, ...]:
    by_id = catalog.by_id
    if block_id not in by_id:
        raise CatalogError(f"unknown block {block_id}")
    result: list[str] = []
    current = by_id[block_id].parent_block_id
    while current is not None:
        result.append(current)
        current = by_id[current].parent_block_id
    result.reverse()
    return tuple(result)


def validate_prefix_closed(target: Iterable[str], catalog: BlockCatalog) -> tuple[str, ...]:
    normalized = tuple(sorted(set(target)))
    selected = set(normalized)
    by_id = catalog.by_id
    unknown = selected - set(by_id)
    if unknown:
        raise CatalogError(f"target contains unknown blocks: {sorted(unknown)}")
    for block_id in normalized:
        missing: set[str] = set()
        parent = by_id[block_id].parent_block_id
        while parent is not None:
            if parent not in selected:
                missing.add(parent)
            parent = by_id[parent].parent_block_id
        if missing:
            raise CatalogError(
                f"target is not prefix-closed for {block_id}: missing {sorted(missing)}"
            )
    return normalized


def block_id_for_token_path(namespace: str, token_ids: Iterable[int], block_size: int) -> str:
    """Return the catalog identity for one page-aligned native token path."""
    tokens = tuple(token_ids)
    if block_size <= 0 or not tokens or len(tokens) % block_size:
        raise CatalogError("native token path must contain complete positive-size pages")
    native_chain = (
        hash(tokens[start : start + block_size])
        for start in range(0, len(tokens), block_size)
    )
    return _path_block_id(namespace, native_chain)


def _path_block_id(namespace: str, native_chain: Iterable[int]) -> str:
    material = namespace + "\n" + "/".join(str(value) for value in native_chain)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _positive_int(row: dict[str, Any], name: str, request_id: int) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatalogError(f"trace request {request_id} field {name} must be positive")
    return value


def _nonnegative_int(row: dict[str, Any], name: str, request_id: int) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogError(f"trace request {request_id} field {name} must be nonnegative")
    return value
