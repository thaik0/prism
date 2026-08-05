from __future__ import annotations

import pytest

from prism.simulation import (
    StorageState,
    exact_placement,
    greedy_placement,
    record_benefits,
)
from prism.simulation.storage import StorageStateError


def test_storage_capacity_promotions_evictions_and_episodes() -> None:
    state = StorageState({0: 4, 1: 6, 2: 11}, capacity_bytes=10)

    state.promote(0, began_in_test=False)
    state.promote(1, began_in_test=True)
    assert state.resident == {0, 1}
    assert state.resident_bytes == 10
    assert state.note_access(1)
    assert not state.note_access(2)
    eviction = state.evict(1)
    assert eviction.episode is not None
    assert eviction.episode.began_in_test
    assert eviction.episode.resident_accesses == 1
    assert state.resident_bytes == 4
    assert not state.fits(2)
    with pytest.raises(StorageStateError, match="oversized"):
        state.promote(2, began_in_test=True)
    with pytest.raises(StorageStateError, match="already resident"):
        state.promote(0, began_in_test=True)


def test_shared_benefit_charges_only_nonresident_promotions() -> None:
    benefits = record_benefits(
        [2.0, 2.0],
        [0, 1],
        {0: 10, 1: 10},
        {0},
        fast_read_cost=1.0,
        slow_read_cost=10.0,
        promotion_cost_per_byte=0.5,
    )

    assert benefits == {0: 18.0, 1: 13.0}


def test_greedy_density_ties_capacity_nonpositive_and_exact_replacement() -> None:
    sizes = {0: 4, 1: 4, 2: 2, 3: 20}
    benefits = {0: 8.0, 1: 8.0, 2: 0.0, 3: 100.0}

    selection = greedy_placement(benefits, sizes, capacity_bytes=4)

    assert selection.target_record_ids == (0,)
    assert selection.objective_value == 8.0


def test_exact_controller_proves_a_greedy_suboptimal_case() -> None:
    sizes = {0: 10, 1: 20, 2: 30}
    benefits = {0: 60.0, 1: 100.0, 2: 120.0}

    greedy = greedy_placement(benefits, sizes, capacity_bytes=50)
    exact_a = exact_placement(benefits, sizes, capacity_bytes=50)
    exact_b = exact_placement(benefits, sizes, capacity_bytes=50)

    assert greedy.target_record_ids == (0, 1)
    assert exact_a.target_record_ids == (1, 2)
    assert exact_a.objective_value == 220.0 > greedy.objective_value
    assert exact_a.solver_status.startswith("optimal")
    assert exact_a == exact_b


def test_controllers_leave_capacity_unused_without_positive_candidates() -> None:
    sizes = {0: 5, 1: 6}
    benefits = {0: 0.0, 1: -1.0}
    assert greedy_placement(benefits, sizes, 10).target_record_ids == ()
    assert exact_placement(benefits, sizes, 10).target_record_ids == ()
