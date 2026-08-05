"""Exactly six deterministic reusable-prefix placement policy paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from prism.simulation.config import SimulationConfig
from prism.simulation.controllers import greedy_placement, record_benefits

from .catalog import BlockCatalog, CatalogError, validate_prefix_closed
from .config import IntegrationConfig, RequestSplit, ResolvedBudget
from .demand import LogicalDemand
from .model import DemandForecasts


class PolicyError(ValueError):
    """Raised when a policy violates chronology, eligibility, or capacity."""


POLICY_DISPLAY_NAMES = {
    "llmservingsim_lru": "LLMServingSim LRU",
    "lfu": "LFU",
    "training_popularity_static_prism": "Training-Popularity Static (Prism)",
    "validation_final_frozen_prism": "Validation-Final Frozen (Prism)",
    "predictive_greedy_prism": "Predictive Greedy (Prism)",
    "oracle_greedy": "Oracle Greedy",
}


@dataclass(frozen=True)
class PlacementDecision:
    policy_id: str
    request_id: int
    target_block_ids: tuple[str, ...] | None
    revealed_current_request: bool
    frozen: bool


class PolicyRuntime:
    """Per-run state; construct a fresh instance for every simulator run."""

    def __init__(
        self,
        policy_id: str,
        catalog: BlockCatalog,
        demand: LogicalDemand,
        forecasts: DemandForecasts,
        split: RequestSplit,
        budget: ResolvedBudget,
        config: IntegrationConfig,
    ) -> None:
        if policy_id not in POLICY_DISPLAY_NAMES:
            raise PolicyError(f"unknown policy {policy_id}")
        self.policy_id = policy_id
        self.catalog = catalog
        self.demand = demand
        self.forecasts = forecasts
        self.split = split
        self.budget = budget
        self.config = config
        self._frequency = np.zeros(len(demand.block_ids), dtype=np.int64)
        self._last_access = np.full(len(demand.block_ids), -1, dtype=np.int64)
        self._seen = np.zeros(len(demand.block_ids), dtype=np.bool_)
        self._last_request = -1
        self._frozen_target: tuple[str, ...] | None = None
        self._id_to_index = {block_id: index for index, block_id in enumerate(demand.block_ids)}
        self._topological_ids = tuple(
            sorted(
                demand.block_ids,
                key=lambda block_id: (catalog.by_id[block_id].position, block_id),
            )
        )
        self._controller_ids = {
            block_id: index for index, block_id in enumerate(self._topological_ids)
        }
        self._controller_block_ids = {
            index: block_id for block_id, index in self._controller_ids.items()
        }
        self._record_sizes = {
            index: budget.block_bytes for index in range(len(self._topological_ids))
        }
        self._simulation_config = SimulationConfig(
            fast_capacity_bytes=budget.reusable_gpu_blocks * budget.block_bytes,
            fast_read_cost=config.fast_read_cost,
            slow_read_cost=config.slow_read_cost,
            promotion_cost_per_byte=config.promotion_cost_per_byte,
        )
        popularity = np.mean(
            demand.demand_matrix[: split.training_end], axis=0, dtype=np.float64
        )
        self._static_forecast = popularity

    def before_request(
        self,
        request_id: int,
        *,
        eligible_block_ids: Iterable[str],
        resident_block_ids: Iterable[str],
        pinned_block_ids: Iterable[str] = (),
    ) -> PlacementDecision:
        if request_id != self._last_request + 1:
            raise PolicyError("requests must be decided in strict chronological order")
        eligible = set(eligible_block_ids)
        resident = set(resident_block_ids)
        pinned = set(pinned_block_ids)
        known = set(self.demand.block_ids)
        if not eligible <= known or not resident <= known or not pinned <= known:
            raise PolicyError("runtime state contains unknown blocks")
        if resident - eligible - pinned:
            raise PolicyError("resident reusable state is not eligible or pinned")
        if len(resident - pinned) > self.budget.reusable_gpu_blocks:
            raise PolicyError("runtime reusable occupancy exceeds the fixed budget")

        target: tuple[str, ...] | None
        revealed = False
        frozen = False
        if self.policy_id == "llmservingsim_lru":
            target = None
        elif self.policy_id == "lfu":
            target = self._lfu_target(eligible)
        elif request_id < self.split.training_end:
            target = None
        elif self.policy_id == "training_popularity_static_prism":
            target = self._select_greedy(
                self._static_forecast, eligible & set(self._training_seen_ids()), resident
            )
            if self._frozen_target is None:
                self._frozen_target = target
            target = self._frozen_target
            frozen = True
        elif self.policy_id == "validation_final_frozen_prism":
            if request_id < self.split.validation_end:
                target = self._predictive_target(request_id, eligible, resident)
                if request_id == self.split.validation_end - 1:
                    self._frozen_target = target
            else:
                if self._frozen_target is None:
                    raise PolicyError("validation-final target was not captured")
                target = self._frozen_target
                frozen = True
        elif self.policy_id == "predictive_greedy_prism":
            target = self._predictive_target(request_id, eligible, resident)
        elif self.policy_id == "oracle_greedy":
            target = self._select_greedy(
                self.demand.demand_matrix[request_id].astype(np.float64),
                eligible,
                resident,
            )
            revealed = True
        else:
            raise AssertionError("unreachable policy")

        self._last_request = request_id
        if target is not None:
            target = self._validate_target(target, eligible, pinned)
        return PlacementDecision(
            policy_id=self.policy_id,
            request_id=request_id,
            target_block_ids=target,
            revealed_current_request=revealed,
            frozen=frozen,
        )

    def after_request(self, request_id: int) -> None:
        if request_id != self._last_request:
            raise PolicyError("request outcome does not match the most recent decision")
        row = self.demand.demand_matrix[request_id]
        touched = np.flatnonzero(row)
        self._frequency[touched] += row[touched]
        self._last_access[touched] = request_id
        self._seen[touched] = True

    def _training_seen_ids(self) -> tuple[str, ...]:
        return tuple(
            block_id
            for block_id, seen in zip(
                self.demand.block_ids, self.demand.training_seen, strict=True
            )
            if seen
        )

    def _predictive_target(
        self, request_id: int, eligible: set[str], resident: set[str]
    ) -> tuple[str, ...]:
        if not self.forecasts.prediction_available[request_id]:
            if self.forecasts.fit_status == "degenerate_tiny_smoke_no_predictive_target":
                return ()
            raise PolicyError(f"prediction is unavailable before request {request_id}")
        allowed = eligible & set(self._training_seen_ids())
        return self._select_greedy(
            self.forecasts.predicted_record_demand[request_id], allowed, resident
        )

    def _lfu_target(self, eligible: set[str]) -> tuple[str, ...]:
        selected = {
            block_id
            for block_id in eligible
            if self._seen[self._id_to_index[block_id]]
        }
        # Native radix eviction can remove only leaves. Repeatedly evict the
        # LFU-minimum eligible leaf so the target remains physically usable.
        by_id = self.catalog.by_id
        children: dict[str, set[str]] = {block_id: set() for block_id in selected}
        for block_id in selected:
            parent = by_id[block_id].parent_block_id
            if parent in selected:
                children[parent].add(block_id)
        while len(selected) > self.budget.reusable_gpu_blocks:
            leaves = [block_id for block_id in selected if not children[block_id]]
            victim = min(
                leaves,
                key=lambda block_id: (
                    self._frequency[self._id_to_index[block_id]],
                    self._last_access[self._id_to_index[block_id]],
                    block_id,
                ),
            )
            selected.remove(victim)
            parent = by_id[victim].parent_block_id
            if parent in selected:
                children[parent].remove(victim)
        return validate_prefix_closed(selected, self.catalog)

    def _select_greedy(
        self, forecast: np.ndarray, allowed: set[str], resident: set[str]
    ) -> tuple[str, ...]:
        values = np.zeros(len(self._topological_ids), dtype=np.float64)
        catalog_by_id = self.catalog.by_id
        for block_id in self._topological_ids:
            source_index = self._id_to_index[block_id]
            value = float(forecast[source_index]) if block_id in allowed else 0.0
            parent = catalog_by_id[block_id].parent_block_id
            if parent is not None:
                value = min(value, values[self._controller_ids[parent]])
            values[self._controller_ids[block_id]] = max(0.0, value)
        resident_ids = {
            self._controller_ids[block_id] for block_id in resident if block_id in allowed
        }
        benefits = record_benefits(
            values,
            tuple(range(len(values))),
            self._record_sizes,
            resident_ids,
            self._simulation_config.fast_read_cost,
            self._simulation_config.slow_read_cost,
            self._simulation_config.promotion_cost_per_byte,
        )
        selection = greedy_placement(
            benefits,
            self._record_sizes,
            self._simulation_config.fast_capacity_bytes,
        )
        target = tuple(
            sorted(self._controller_block_ids[index] for index in selection.target_record_ids)
        )
        try:
            return validate_prefix_closed(target, self.catalog)
        except CatalogError as exc:
            raise PolicyError(f"controller produced a non-prefix-closed target: {exc}") from exc

    def _validate_target(
        self, target: tuple[str, ...], eligible: set[str], pinned: set[str]
    ) -> tuple[str, ...]:
        normalized = validate_prefix_closed(target, self.catalog)
        if set(normalized) - eligible - pinned:
            raise PolicyError("target includes a currently ineligible reusable block")
        if len(set(normalized) - pinned) > self.budget.reusable_gpu_blocks:
            raise PolicyError("target exceeds the fixed reusable GPU budget")
        return normalized
