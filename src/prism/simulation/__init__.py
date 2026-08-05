"""Deterministic simulated placement for Prism Milestone 4."""

from prism.simulation.config import SimulationConfig, SimulationConfigError
from prism.simulation.controllers import (
    PlacementSelection,
    exact_placement,
    greedy_placement,
    record_benefits,
)
from prism.simulation.diagnostics import evaluate_causal_diagnostics
from prism.simulation.projection import (
    ProjectionModel,
    ProjectionResult,
    build_record_demand_variants,
    fit_record_demand_projection,
    project_record_demand,
)
from prism.simulation.persistence import (
    PolicyInputBundle,
    SimulationInputError,
    SimulationOutputDirectoryError,
    SimulationRun,
    load_policy_inputs,
    run_simulated_evaluation,
)
from prism.simulation.replay import (
    POLICY_DISPLAY_NAMES,
    POLICY_NAMES,
    PolicyReplayResult,
    ReplayError,
    greedy_policy_target,
    run_policy_replay,
    select_lfu_victim,
    select_lru_victim,
    training_popularity_forecast,
)
from prism.simulation.storage import Migration, PromotionEpisode, StorageState

__all__ = [
    "Migration",
    "PlacementSelection",
    "PolicyReplayResult",
    "PolicyInputBundle",
    "ProjectionModel",
    "ProjectionResult",
    "PromotionEpisode",
    "SimulationConfig",
    "SimulationConfigError",
    "SimulationInputError",
    "SimulationOutputDirectoryError",
    "SimulationRun",
    "StorageState",
    "build_record_demand_variants",
    "exact_placement",
    "evaluate_causal_diagnostics",
    "fit_record_demand_projection",
    "greedy_placement",
    "greedy_policy_target",
    "load_policy_inputs",
    "project_record_demand",
    "record_benefits",
    "ReplayError",
    "POLICY_NAMES",
    "POLICY_DISPLAY_NAMES",
    "run_policy_replay",
    "select_lfu_victim",
    "select_lru_victim",
    "training_popularity_forecast",
    "run_simulated_evaluation",
]
