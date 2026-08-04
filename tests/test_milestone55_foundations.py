from __future__ import annotations

from pathlib import Path

import numpy as np

from prism.experiments.actionability import (
    CommonWindowProtocol,
    build_horizon_targets,
    cumulative_future,
)
from prism.experiments.config import ActionabilityManifest, load_manifest
from prism.experiments.materialize import resolve_actionability_workload_config
from prism.predictor.features import PredictorDataset, WindowContext
from prism.structure import build_demand_matrix
from prism.workload import WorkloadConfig, generate_workload, persist_workload


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_actionability_manifest_exact_values_order_and_ids() -> None:
    manifest = load_manifest(ROOT / "configs/milestone55_actionability.json")
    assert isinstance(manifest, ActionabilityManifest)
    assert [regime.id for regime in manifest.regimes] == [
        "baseline", "sparse", "very_sparse"
    ]
    assert manifest.horizons == (1, 2, 4)
    assert manifest.seeds == (1729, 2718, 31415)
    assert len(manifest.experiment_ids) == 27
    assert manifest.experiment_ids[0] == "baseline__h1__seed_1729"
    assert manifest.experiment_ids[-1] == "very_sparse__h4__seed_31415"
    assert manifest.regime("sparse").workload_overrides == {
        "precursor_probability_scale": 0.275,
        "spontaneous_activation_probability": 0.04,
        "post_burst_cooldown_windows": 5,
    }
    assert manifest.regime("very_sparse").workload_overrides[
        "post_burst_cooldown_windows"
    ] == 10


def test_regime_materialization_changes_only_frozen_fields_and_seed() -> None:
    manifest = load_manifest(ROOT / "configs/milestone55_actionability.json")
    assert isinstance(manifest, ActionabilityManifest)
    base = WorkloadConfig.from_json(manifest.base_workload_config).to_resolved_dict()
    for regime in manifest.regimes:
        resolved = resolve_actionability_workload_config(
            manifest, regime, 2718
        ).to_resolved_dict()
        changed = {key for key in base if base[key] != resolved[key]}
        expected = {
            key for key, value in regime.workload_overrides.items()
            if base[key] != value
        } | {"seed"}
        assert changed == expected


def test_cooldown_exact_eligibility_simultaneous_sets_and_legacy_bytes(
    tmp_path, make_config
) -> None:
    raw = make_config(
        num_windows=8,
        num_working_sets=2,
        working_set_support_min=2,
        working_set_support_max=2,
        spontaneous_activation_probability=1.0,
        precursor_probability_scale=0.0,
        burst_duration_min_windows=1,
        burst_duration_max_windows=1,
        post_burst_cooldown_windows=2,
    ).to_resolved_dict()
    result = generate_workload(WorkloadConfig.from_dict(raw))
    starts = {
        factor: [
            burst.start_window
            for burst in result.hidden_ground_truth.bursts
            if burst.working_set_id == factor
        ]
        for factor in range(2)
    }
    assert starts == {0: [0, 3, 6], 1: [0, 3, 6]}
    assert [event.event_index for event in result.observable_events] == list(
        range(len(result.observable_events))
    )

    legacy = make_config()
    explicit = WorkloadConfig.from_dict(
        {**legacy.to_dict(), "post_burst_cooldown_windows": 0}
    )
    for name, config in (("legacy", legacy), ("explicit", explicit)):
        persist_workload(generate_workload(config), tmp_path / name)
    for filename in (
        "config.json", "observable_events.jsonl", "hidden_ground_truth.json", "summary.json"
    ):
        assert (tmp_path / "legacy" / filename).read_bytes() == (
            tmp_path / "explicit" / filename
        ).read_bytes()


def test_common_windows_and_cumulative_targets_are_exact() -> None:
    protocol = CommonWindowProtocol(600, 600, 800, 800, 1000)
    assert protocol.to_dict()["validation_evaluation_windows"] == [600, 797]
    assert protocol.to_dict()["validation_carry_only_windows"] == [797, 800]
    assert protocol.to_dict()["test_evaluation_windows"] == [800, 997]
    assert protocol.to_dict()["final_excluded_tail_windows"] == [997, 1000]
    assert protocol.target_is_common(596)
    assert not protocol.target_is_common(597)
    assert protocol.target_is_common(996)
    assert not protocol.target_is_common(997)
    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    assert np.array_equal(cumulative_future(values, 2)[1], values[1] + values[2])
    assert np.all(np.isnan(cumulative_future(values, 4)[-3:]))


def test_horizon_labels_binary_multiple_starts_and_summed_intensity(
    tmp_path, make_config
) -> None:
    config = make_config(
        num_windows=10,
        num_records=4,
        num_working_sets=1,
        working_set_support_min=4,
        working_set_support_max=4,
        spontaneous_activation_probability=1.0,
        precursor_probability_scale=0.0,
        burst_duration_min_windows=1,
        burst_duration_max_windows=1,
    )
    result = generate_workload(config)
    persist_workload(result, tmp_path / "run")
    demand = build_demand_matrix(tmp_path / "run")
    membership = np.asarray(
        [[member.weight for member in result.hidden_ground_truth.working_set_memberships[0].members]],
        dtype=np.float64,
    )
    contexts = tuple(
        WindowContext(1, 1, int(demand.X[w].sum()), np.ones(1), np.ones(1))
        for w in range(config.num_windows)
    )
    dataset = PredictorDataset(
        projected_factor_demand=np.asarray(demand.X, dtype=np.float64) @ membership.T,
        recent_features=np.zeros((1, 1)),
        context_features=np.zeros((1, 1)),
        recent_feature_names=("x",),
        context_feature_names=("x",),
        recent_continuous_indices=np.asarray([0]),
        context_continuous_indices=np.asarray([0]),
        feature_window_ids=np.asarray([2]),
        target_window_ids=np.asarray([3]),
        factor_ids=np.asarray([0]),
        split_codes=np.asarray([0], dtype=np.int8),
        user_ids=np.asarray([0]),
        request_types=("interactive",),
        window_contexts=contexts,
        warnings=(),
    )
    targets = build_horizon_targets(
        tmp_path / "run", membership, demand.record_ids, dataset, 4
    )
    expected = sum(
        burst.intensity
        for burst in result.hidden_ground_truth.bursts
        if 3 <= burst.start_window < 7
    )
    assert targets.activation.tolist() == [1]
    assert targets.intensity.tolist() == [expected]
    assert targets.realized_next_window_accesses[0] == sum(
        row.working_set_access_counts[0]
        for row in result.hidden_ground_truth.access_source_counts_by_window[3:7]
    )
