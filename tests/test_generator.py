from __future__ import annotations

from collections import defaultdict
import math
import random

import pytest

from prism.workload import generate_workload
from prism.workload.generator import (
    _sample_access_source_and_record,
    _weighted_index,
    calculate_activation_probabilities,
)
from prism.workload.models import Burst, OBSERVABLE_EVENT_FIELDS


def test_in_memory_determinism_and_distinct_fixed_seeds(make_config) -> None:
    config = make_config(seed=19)

    assert generate_workload(config) == generate_workload(config)
    assert generate_workload(config) != generate_workload(make_config(seed=20))


def test_observable_schema_ordering_ids_and_record_contract(make_config) -> None:
    config = make_config(seed=7, num_windows=6)
    result = generate_workload(config)
    events = result.observable_events

    assert [event.event_index for event in events] == list(range(len(events)))
    assert all(set(event.to_dict()) == OBSERVABLE_EVENT_FIELDS for event in events)
    assert all(0 <= event.window_id < config.num_windows for event in events)
    assert [event.window_id for event in events] == sorted(
        event.window_id for event in events
    )
    assert all(0 <= event.record_id < config.num_records for event in events)
    assert all(0 <= event.user_id < config.num_users for event in events)
    assert all(event.request_type in config.request_types for event in events)
    assert all(
        event.operation_type in config.operation_type_probabilities
        for event in events
    )

    record_sizes = result.hidden_ground_truth.record_sizes_bytes
    assert all(
        config.record_size_min_bytes <= size <= config.record_size_max_bytes
        for size in record_sizes
    )
    assert all(
        event.record_size_bytes == record_sizes[event.record_id] for event in events
    )
    minimum_events = (
        config.num_windows
        * config.min_sessions_per_window
        * config.min_requests_per_session
        * config.min_accesses_per_request
    )
    maximum_events = (
        config.num_windows
        * config.max_sessions_per_window
        * config.max_requests_per_session
        * config.max_accesses_per_request
    )
    assert minimum_events <= len(events) <= maximum_events


def test_sessions_and_requests_have_one_parent_context(make_config) -> None:
    result = generate_workload(make_config(num_windows=7))
    session_contexts: dict[int, set[tuple[int, int]]] = defaultdict(set)
    request_contexts: dict[int, set[tuple[int, int, int, str]]] = defaultdict(set)

    for event in result.observable_events:
        session_contexts[event.session_id].add((event.window_id, event.user_id))
        request_contexts[event.request_id].add(
            (
                event.session_id,
                event.window_id,
                event.user_id,
                event.request_type,
            )
        )

    assert all(len(contexts) == 1 for contexts in session_contexts.values())
    assert all(len(contexts) == 1 for contexts in request_contexts.values())
    assert sorted(session_contexts) == list(range(result.summary.total_sessions))
    assert sorted(request_contexts) == list(range(result.summary.total_requests))


def test_static_hidden_vectors_and_memberships_are_valid(make_config) -> None:
    config = make_config(
        num_records=12,
        num_working_sets=3,
        working_set_support_min=3,
        working_set_support_max=5,
    )
    hidden = generate_workload(config).hidden_ground_truth

    for membership in hidden.working_set_memberships:
        record_ids = [member.record_id for member in membership.members]
        assert 3 <= len(record_ids) <= 5
        assert record_ids == sorted(set(record_ids))
        assert all(member.weight > 0.0 for member in membership.members)
        assert math.isclose(
            math.fsum(member.weight for member in membership.members),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert len(record_ids) < config.num_records

    vectors = (
        hidden.user_working_set_affinities
        + hidden.request_type_working_set_affinities
        + hidden.user_request_type_preferences
    )
    for vector in vectors:
        assert all(weight > 0.0 for weight in vector.weights)
        assert math.isclose(
            math.fsum(vector.weights), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    assert all(weight > 0.0 for weight in hidden.baseline_record_popularity)
    assert math.isclose(
        math.fsum(hidden.baseline_record_popularity),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_independent_full_supports_guarantee_overlap(make_config) -> None:
    config = make_config(
        num_records=5,
        num_working_sets=3,
        working_set_support_min=5,
        working_set_support_max=5,
    )
    result = generate_workload(config)

    assert result.summary.records_in_multiple_working_sets == 5
    assert all(
        {member.record_id for member in membership.members} == set(range(5))
        for membership in result.hidden_ground_truth.working_set_memberships
    )


@pytest.mark.parametrize(
    ("spontaneous", "scale", "score", "expected_context", "expected_combined"),
    [
        (0.2, 0.0, 0.0, 0.0, 0.2),
        (0.2, 0.5, 0.0, 0.0, 0.2),
        (0.2, 0.5, 0.6, 0.3, 0.44),
        (0.0, 1.0, 1.0, 1.0, 1.0),
    ],
)
def test_activation_probability_formula(
    spontaneous: float,
    scale: float,
    score: float,
    expected_context: float,
    expected_combined: float,
) -> None:
    contextual, combined = calculate_activation_probabilities(
        spontaneous, scale, score
    )

    assert contextual == pytest.approx(expected_context)
    assert combined == pytest.approx(expected_combined)


def test_nonzero_precursor_can_remain_probabilistic() -> None:
    contextual, combined = calculate_activation_probabilities(0.1, 0.5, 0.5)

    assert contextual > 0.0
    assert 0.1 < combined < 1.0


def test_one_window_delay_and_guaranteed_true_precursor(make_config) -> None:
    config = make_config(
        num_windows=3,
        num_working_sets=1,
        spontaneous_activation_probability=0.0,
        precursor_probability_scale=1.0,
        burst_duration_min_windows=1,
        burst_duration_max_windows=1,
    )
    hidden = generate_workload(config).hidden_ground_truth
    trials = {(trial.window_id, trial.working_set_id): trial for trial in hidden.activation_trials}

    assert trials[(0, 0)].previous_window_precursor_score == 0.0
    assert not trials[(0, 0)].activated
    assert hidden.precursor_scores_by_window[0].scores == (1.0,)
    assert trials[(1, 0)].previous_window_precursor_score == 1.0
    assert trials[(1, 0)].activation_probability == 1.0
    assert trials[(1, 0)].activated

    for trial in hidden.activation_trials:
        if trial.window_id == 0:
            assert trial.previous_window_precursor_score == 0.0
        else:
            expected = hidden.precursor_scores_by_window[trial.window_id - 1].scores[
                trial.working_set_id
            ]
            assert trial.previous_window_precursor_score == expected


def test_unannounced_bursts_expire_and_retrigger_without_overlap(make_config) -> None:
    config = make_config(
        num_windows=5,
        num_working_sets=1,
        spontaneous_activation_probability=1.0,
        precursor_probability_scale=0.0,
        burst_duration_min_windows=2,
        burst_duration_max_windows=2,
        burst_intensity_min=2.5,
        burst_intensity_max=2.5,
    )
    hidden = generate_workload(config).hidden_ground_truth

    assert [burst.start_window for burst in hidden.bursts] == [0, 2, 4]
    assert all(burst.sampled_duration_windows == 2 for burst in hidden.bursts)
    assert all(burst.intensity == 2.5 for burst in hidden.bursts)
    assert hidden.activation_trials[0].previous_window_precursor_score == 0.0
    assert hidden.activation_trials[0].activated
    for previous, current in zip(hidden.bursts, hidden.bursts[1:], strict=False):
        assert previous.end_window_exclusive <= current.start_window
    assert {trial.window_id for trial in hidden.activation_trials} == {0, 2, 4}


def test_source_counts_match_events_and_summary(make_config) -> None:
    result = generate_workload(make_config(num_windows=8))
    hidden = result.hidden_ground_truth

    for counts in hidden.access_source_counts_by_window:
        event_count = sum(
            event.window_id == counts.window_id for event in result.observable_events
        )
        assert counts.total_access_count == event_count
    assert sum(c.total_access_count for c in hidden.access_source_counts_by_window) == (
        result.summary.total_events
    )
    assert (
        result.summary.baseline_access_count
        + result.summary.noise_access_count
        + result.summary.working_set_access_count
        == result.summary.total_events
    )


@pytest.mark.parametrize(
    ("baseline_weight", "noise_weight", "expected_source"),
    [(1.0, 0.0, "baseline"), (0.0, 1.0, "noise")],
)
def test_nonburst_sources_work_without_active_bursts(
    make_config,
    baseline_weight: float,
    noise_weight: float,
    expected_source: str,
) -> None:
    result = generate_workload(
        make_config(
            spontaneous_activation_probability=0.0,
            precursor_probability_scale=0.0,
            baseline_access_weight=baseline_weight,
            noise_access_weight=noise_weight,
        )
    )

    assert result.summary.total_bursts == 0
    assert getattr(result.summary, f"{expected_source}_access_count") == (
        result.summary.total_events
    )


def test_burst_source_never_samples_outside_membership(make_config) -> None:
    config = make_config(
        num_records=10,
        num_working_sets=1,
        working_set_support_min=2,
        working_set_support_max=2,
        burst_access_weight=1.0,
        baseline_access_weight=0.0,
        noise_access_weight=1e-300,
    )
    result = generate_workload(config)
    membership = result.hidden_ground_truth.working_set_memberships[0]
    support = {member.record_id for member in membership.members}
    active = {
        0: Burst(
            burst_id=0,
            working_set_id=0,
            start_window=0,
            sampled_duration_windows=1,
            end_window_exclusive=1,
            intensity=1.0,
        )
    }
    rng = random.Random(5)

    for _ in range(100):
        source, record_id = _sample_access_source_and_record(
            config,
            rng,
            active,
            result.hidden_ground_truth.working_set_memberships,
            result.hidden_ground_truth.baseline_record_popularity,
        )
        assert source == 0
        assert record_id in support


def test_weighted_selection_uses_supplied_distribution() -> None:
    rng = random.Random(3)

    assert [_weighted_index((0.0, 1.0, 0.0), rng) for _ in range(20)] == [
        1
    ] * 20


def test_operation_type_is_label_only(make_config) -> None:
    read_result = generate_workload(
        make_config(operation_type_probabilities={"read": 1.0})
    )
    write_result = generate_workload(
        make_config(operation_type_probabilities={"write": 1.0})
    )

    assert read_result.hidden_ground_truth == write_result.hidden_ground_truth
    for read_event, write_event in zip(
        read_result.observable_events, write_result.observable_events, strict=True
    ):
        read_fields = read_event.to_dict()
        write_fields = write_event.to_dict()
        assert read_fields.pop("operation_type") == "read"
        assert write_fields.pop("operation_type") == "write"
        assert read_fields == write_fields
