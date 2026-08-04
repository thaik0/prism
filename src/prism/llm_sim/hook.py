"""Runtime callbacks loaded by the isolated LLMServingSim integration patch."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .adapter import LLMServingSimAdapter, PlacementOutcome
from .catalog import build_block_catalog
from .config import IntegrationConfig
from .demand import build_logical_demand
from .model import fit_demand_forecasts
from .policies import PlacementDecision, PolicyRuntime


class HookError(RuntimeError):
    """Raised when the simulator violates the callback protocol."""


class SimulatorHook:
    """One clean policy runtime attached to one simulator process."""

    def __init__(self, payload_path: str | Path):
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
        expected = {"config", "output_dir", "policy_id", "tiny"}
        if set(payload) != expected:
            raise HookError("hook payload keys differ from the fixed protocol")
        self.config = IntegrationConfig.from_json(payload["config"])
        self.policy_id = str(payload["policy_id"])
        self.tiny = bool(payload["tiny"])
        self.output_dir = Path(payload["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = self.config.split(tiny=self.tiny)
        self.budget = self.config.resolve_budget()
        self.catalog = build_block_catalog(
            self.config.trace_path(tiny=self.tiny),
            namespace=(
                f"{self.config.model}|{self.config.dtype}|"
                f"b{self.config.block_size_tokens}"
            ),
            block_size_tokens=self.config.block_size_tokens,
            block_bytes=self.budget.block_bytes,
        )
        self.demand = build_logical_demand(self.catalog, self.split)
        self.forecasts = fit_demand_forecasts(
            self.demand,
            self.split,
            self.config,
            allow_degenerate_smoke=self.tiny,
        )
        self.runtime = PolicyRuntime(
            self.policy_id,
            self.catalog,
            self.demand,
            self.forecasts,
            self.split,
            self.budget,
            self.config,
        )
        self.adapter = LLMServingSimAdapter(self.catalog, self.budget)
        self.requires_completion_first = self.policy_id != "llmservingsim_lru"
        self._desired_target: tuple[str, ...] | None = None
        self._decisions: list[dict[str, Any]] = []
        self._lifecycle: list[dict[str, Any]] = []
        self._request_metrics: dict[int, dict[str, Any]] = {}
        self._batch_metrics: list[dict[str, Any]] = []
        self._last_batch_id = -1
        self._placement_changes = 0
        self._promoted_blocks = 0
        self._evicted_blocks = 0
        self._max_reusable_occupancy = 0

    def before_request(self, request_data: dict[str, Any], scheduler: Any) -> None:
        request_id = int(request_data["index"])
        if request_id >= self.split.request_count:
            raise HookError("simulator routed more requests than the frozen split")
        snapshot = self.adapter.snapshot(scheduler.memory)
        eligible = set(snapshot.host_block_ids) | set(snapshot.resident_block_ids)
        decision = self.runtime.before_request(
            request_id,
            eligible_block_ids=eligible,
            resident_block_ids=snapshot.resident_block_ids,
            pinned_block_ids=snapshot.pinned_block_ids,
        )
        prior = self._desired_target
        self._desired_target = decision.target_block_ids
        outcome = None
        if self._desired_target is not None:
            outcome = self.adapter.apply_target(scheduler.memory, self._desired_target)
            self._record_placement("before_request", request_id, outcome)
        if prior is not None and self._desired_target is not None and prior != self._desired_target:
            self._placement_changes += len(set(prior) ^ set(self._desired_target))
        logical = tuple(
            self.demand.block_ids[index]
            for index in self.demand.ordered_block_indices[request_id]
        )
        target = set(self._desired_target or ())
        self._decisions.append(
            {
                **_decision_dict(decision),
                "eligible_blocks": len(eligible),
                "resident_blocks_before": len(snapshot.resident_block_ids),
                "pinned_blocks_before": len(snapshot.pinned_block_ids),
                "target_demand_blocks": len(target & set(logical)),
                "logical_demand_blocks": len(logical),
            }
        )

    def after_request_revealed(
        self, request_data: dict[str, Any], scheduler: Any
    ) -> None:
        del scheduler
        self.runtime.after_request(int(request_data["index"]))

    def after_lifecycle(self, cycle: int, schedulers: list[Any]) -> None:
        if self._desired_target is None:
            for scheduler in schedulers:
                occupancy = self.adapter.snapshot(
                    scheduler.memory
                ).reusable_occupancy_blocks
                self._max_reusable_occupancy = max(
                    self._max_reusable_occupancy, occupancy
                )
            return
        for scheduler in schedulers:
            outcome = self.adapter.apply_target(scheduler.memory, self._desired_target)
            self._record_placement("after_lifecycle", int(cycle), outcome)

    def on_batch_scheduled(self, batch: Any | None) -> None:
        if batch is None or int(batch.batch_id) <= self._last_batch_id:
            return
        self._last_batch_id = int(batch.batch_id)
        self._batch_metrics.append(
            {
                "batch_id": int(batch.batch_id),
                "batch_time_ns": int(batch.batch_time),
                "request_ids": sorted(int(req.id) for req in batch.requests),
                "native_kv_load_bytes": int(batch.load),
                "native_kv_evict_bytes": int(batch.evict),
            }
        )

    def on_request_completed(self, request: Any) -> None:
        request_id = int(request.id)
        logical_indices = self.demand.ordered_block_indices[request_id]
        logical_count = len(logical_indices)
        block_tokens = self.config.block_size_tokens
        gpu_blocks = min(logical_count, int(request.npu_cache_hit) // block_tokens)
        total_cached = min(logical_count, int(request.storage_cache_hit) // block_tokens)
        host_blocks = max(0, total_cached - gpu_blocks)
        recomputed = logical_count - gpu_blocks - host_blocks
        outcomes = (
            ("gpu", logical_indices[:gpu_blocks]),
            ("host", logical_indices[gpu_blocks : gpu_blocks + host_blocks]),
            ("recompute", logical_indices[gpu_blocks + host_blocks :]),
        )
        cold = self.demand.cold_start
        classification: dict[str, dict[str, int]] = {}
        for name, indices in outcomes:
            cold_count = sum(bool(cold[index]) for index in indices)
            classification[name] = {
                "training_seen_blocks": len(indices) - cold_count,
                "cold_start_blocks": cold_count,
            }
        promoted = 0
        self._promoted_blocks += host_blocks
        if self._desired_target is not None:
            logical_ids = [self.demand.block_ids[index] for index in logical_indices]
            host_ids = set(logical_ids[gpu_blocks : gpu_blocks + host_blocks])
            promoted = len(host_ids & set(self._desired_target))
        self._request_metrics[request_id] = {
            "policy_id": self.policy_id,
            "request_id": request_id,
            "split": _split_name(request_id, self.split),
            "input_tokens": int(request.original_input),
            "output_tokens": int(request.output - request.original_input),
            "arrival_ns": int(request.arrival),
            "end_time_ns": int(request.end_time),
            "latency_ns": int(request.latency),
            "ttft_ns": int(request.ttft),
            "tpot_ns": int(request.tpot),
            "logical_blocks": logical_count,
            "gpu_hit_blocks": gpu_blocks,
            "gpu_hit_tokens": gpu_blocks * block_tokens,
            "host_hit_blocks": host_blocks,
            "host_hit_tokens": host_blocks * block_tokens,
            "recomputed_blocks": recomputed,
            "recomputed_tokens": recomputed * block_tokens,
            "host_to_gpu_bytes": host_blocks * self.budget.block_bytes,
            "targeted_promoted_blocks": promoted,
            "cold_start_outcomes": classification,
        }

    def finalize(self, cycle: int, schedulers: list[Any]) -> None:
        if len(self._request_metrics) != self.split.request_count:
            raise HookError(
                f"completed {len(self._request_metrics)} requests, expected "
                f"{self.split.request_count}"
            )
        final_snapshots = [self.adapter.snapshot(item.memory) for item in schedulers]
        result = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "logical_demand_sha256": self.demand.logical_stream_sha256,
            "request_count": self.split.request_count,
            "total_cycles_ns": int(cycle),
            "placement_changes": self._placement_changes,
            "blocks_promoted": self._promoted_blocks,
            "blocks_evicted": self._evicted_blocks,
            "max_reusable_occupancy_blocks": self._max_reusable_occupancy,
            "final_reusable_occupancy_blocks": sum(
                item.reusable_occupancy_blocks for item in final_snapshots
            ),
            "reusable_budget_blocks": self.budget.reusable_gpu_blocks,
            "model_fit_status": self.forecasts.fit_status,
            "native_batch_load_bytes": sum(
                item["native_kv_load_bytes"] for item in self._batch_metrics
            ),
            "native_batch_evict_bytes": sum(
                item["native_kv_evict_bytes"] for item in self._batch_metrics
            ),
        }
        _write_json(self.output_dir / "hook_results.json", result)
        _write_jsonl(self.output_dir / "placement_decisions.jsonl", self._decisions)
        _write_jsonl(self.output_dir / "lifecycle_events.jsonl", self._lifecycle)
        _write_jsonl(
            self.output_dir / "request_cache_metrics.jsonl",
            [self._request_metrics[index] for index in sorted(self._request_metrics)],
        )
        _write_jsonl(self.output_dir / "batch_metrics.jsonl", self._batch_metrics)

    def _record_placement(
        self, phase: str, sequence: int, outcome: PlacementOutcome
    ) -> None:
        evicted = len(outcome.evicted_block_ids)
        self._evicted_blocks += evicted
        occupancy = outcome.after.reusable_occupancy_blocks
        self._max_reusable_occupancy = max(self._max_reusable_occupancy, occupancy)
        self._lifecycle.append(
            {
                "phase": phase,
                "sequence": sequence,
                "evicted_block_ids": list(outcome.evicted_block_ids),
                "structurally_evicted_native_tokens": (
                    outcome.structurally_evicted_native_tokens
                ),
                "resident_blocks_before": len(outcome.before.resident_block_ids),
                "resident_blocks_after": len(outcome.after.resident_block_ids),
                "pinned_blocks": len(outcome.after.pinned_block_ids),
                "reusable_occupancy_blocks": occupancy,
            }
        )


def load_hook(payload_path: str | Path) -> SimulatorHook:
    """Stable entry point imported by the upstream patch."""
    return SimulatorHook(payload_path)


def _decision_dict(decision: PlacementDecision) -> dict[str, Any]:
    value = asdict(decision)
    target = value["target_block_ids"]
    value["target_block_ids"] = list(target) if target is not None else None
    return value


def _split_name(request_id: int, split: Any) -> str:
    if request_id < split.training_end:
        return "training"
    if request_id < split.validation_end:
        return "validation"
    return "test"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
