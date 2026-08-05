from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prism.predictor.cli import main as predictor_main
from prism.predictor.persistence import (
    PREDICTOR_ARTIFACT_FILENAMES,
    PredictorOutputDirectoryError,
    run_predictor_experiment,
)
from prism.workload import WorkloadConfig, generate_workload, persist_workload
from prism.workload.generator import ARTIFACT_FILENAMES as SOURCE_ARTIFACT_FILENAMES
from prism.workload.validate import validate_workload_run
from tests.conftest import (
    MILESTONE3_PREDICTOR_CONFIG_PATH,
    MILESTONE3_WORKLOAD_CONFIG_PATH,
    REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
)


BUNDLE_ARRAYS = {
    "membership_matrix",
    "record_ids",
    "factor_ids",
    "recent_feature_names",
    "recent_scaler_means",
    "recent_scaler_scales",
    "recent_continuous_column_indices",
    "recent_logistic_coefficients",
    "recent_logistic_intercept",
    "context_feature_names",
    "context_scaler_means",
    "context_scaler_scales",
    "context_continuous_column_indices",
    "context_logistic_coefficients",
    "context_logistic_intercept",
    "intensity_scaler_means",
    "intensity_scaler_scales",
    "intensity_continuous_column_indices",
    "intensity_ridge_coefficients",
    "intensity_ridge_intercept",
    "activation_base_rates",
    "intensity_factor_means",
    "global_intensity_mean",
    "user_ids",
    "request_types",
}
PREDICTION_ARRAYS = {
    "feature_window_id",
    "target_window_id",
    "learned_factor_id",
    "split_code",
    "base_rate_activation_probability",
    "recent_demand_activation_probability",
    "context_plus_state_activation_probability",
    "per_factor_mean_intensity_prediction",
    "context_plus_state_intensity_prediction",
}


def _bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scores))


def _reconstructed_linear(
    features: np.ndarray,
    artifact,
    prefix: str,
    coefficient_name: str,
    intercept_name: str,
) -> np.ndarray:
    transformed = features.copy()
    indices = artifact[f"{prefix}_continuous_column_indices"]
    transformed[:, indices] = (
        transformed[:, indices] - artifact[f"{prefix}_scaler_means"]
    ) / artifact[f"{prefix}_scaler_scales"]
    return (
        transformed @ artifact[coefficient_name].reshape(-1)
        + artifact[intercept_name][0]
    )


def test_representative_predictor_gates_artifacts_leakage_and_determinism(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source_config = WorkloadConfig.from_json(MILESTONE3_WORKLOAD_CONFIG_PATH)
    persist_workload(generate_workload(source_config), source)
    validation = validate_workload_run(source)
    assert validation.demonstrations_passed
    assert validation.intensity_signal_passed
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}

    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    first = run_predictor_experiment(
        source,
        REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
        MILESTONE3_PREDICTOR_CONFIG_PATH,
        output_a,
    )
    second = run_predictor_experiment(
        source,
        REPRESENTATIVE_STRUCTURE_CONFIG_PATH,
        MILESTONE3_PREDICTOR_CONFIG_PATH,
        output_b,
    )

    assert set(_bytes(output_a)) == set(PREDICTOR_ARTIFACT_FILENAMES)
    assert _bytes(output_a) == _bytes(output_b)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == source_before
    assert first.training_structure.converged
    assert first.predictor.converged
    assert first.evaluation.all_gates_passed
    assert second.evaluation.all_gates_passed
    report = first.evaluation.report
    assert report["split_boundaries"]["train_end"] == 600
    assert report["split_boundaries"]["validation_end"] == 800
    assert report["split_counts"]["training"]["example_count"] == 2388
    assert report["split_counts"]["validation"]["example_count"] == 800
    assert report["split_counts"]["test"]["example_count"] == 800
    assert all(
        row["positive_activation_count"] > 0
        for row in report["split_counts"]["training"]["by_factor"]
    )
    assert first.training_structure.activation_matrix.shape == (600, 4)
    assert report["training_only_structure"]["matching_and_recovery"][
        "support_recovery"
    ]["mean_recall"] > report["training_only_structure"]["matching_and_recovery"][
        "support_recovery"
    ]["mean_analytic_random_support_expectation"]

    hidden = json.loads((source / "hidden_ground_truth.json").read_text())
    burst_starts = {
        (burst["start_window"], burst["working_set_id"]): burst["intensity"]
        for burst in hidden["bursts"]
    }
    for row in range(first.dataset.example_count):
        key = (
            int(first.dataset.target_window_ids[row]),
            int(first.targets.learned_to_planted[first.dataset.factor_ids[row]]),
        )
        assert first.targets.activation[row] == int(key in burst_starts)
        if key in burst_starts:
            assert first.targets.intensity[row] == burst_starts[key]
        else:
            assert np.isnan(first.targets.intensity[row])

    with np.load(output_a / "predictor_bundle.npz", allow_pickle=False) as bundle:
        assert set(bundle.files) == BUNDLE_ARRAYS
        assert not any(
            token in name
            for name in bundle.files
            for token in ("hidden", "planted", "label", "eligible", "precursor", "burst")
        )
        with np.load(output_a / "predictions.npz", allow_pickle=False) as predictions:
            assert set(predictions.files) == PREDICTION_ARRAYS
            assert not any(
                token in name
                for name in predictions.files
                for token in ("label", "eligible", "planted", "hidden")
            )
            ordering = list(
                zip(
                    predictions["feature_window_id"].tolist(),
                    predictions["learned_factor_id"].tolist(),
                    strict=True,
                )
            )
            assert ordering == sorted(ordering)
            recent_scores = _reconstructed_linear(
                first.dataset.recent_features,
                bundle,
                "recent",
                "recent_logistic_coefficients",
                "recent_logistic_intercept",
            )
            context_scores = _reconstructed_linear(
                first.dataset.context_features,
                bundle,
                "context",
                "context_logistic_coefficients",
                "context_logistic_intercept",
            )
            intensity = _reconstructed_linear(
                first.dataset.context_features,
                bundle,
                "intensity",
                "intensity_ridge_coefficients",
                "intensity_ridge_intercept",
            )
            np.testing.assert_allclose(
                _sigmoid(recent_scores),
                predictions["recent_demand_activation_probability"],
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                _sigmoid(context_scores),
                predictions["context_plus_state_activation_probability"],
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                intensity,
                predictions["context_plus_state_intensity_prediction"],
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_array_equal(
                bundle["activation_base_rates"][predictions["learned_factor_id"]],
                predictions["base_rate_activation_probability"],
            )

    config_artifact = json.loads((output_a / "predictor_config.json").read_text())
    for filename in SOURCE_ARTIFACT_FILENAMES:
        content = (source / filename).read_bytes()
        assert config_artifact["source_artifact_sha256"][filename] == hashlib.sha256(
            content
        ).hexdigest()
    assert str(tmp_path).encode() not in (output_a / "predictor_config.json").read_bytes()
    assert str(tmp_path).encode() not in (output_a / "evaluation_report.json").read_bytes()


def test_nonempty_output_is_rejected_before_source_work(tmp_path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    keep = occupied / "keep.txt"
    keep.write_text("keep\n", encoding="utf-8")

    with pytest.raises(PredictorOutputDirectoryError, match="must be empty"):
        run_predictor_experiment("missing", "missing", "missing", occupied)

    assert keep.read_text(encoding="utf-8") == "keep\n"


def test_cli_writes_diagnostics_then_exits_nonzero_on_logistic_nonconvergence(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    persist_workload(
        generate_workload(WorkloadConfig.from_json(MILESTONE3_WORKLOAD_CONFIG_PATH)),
        source,
    )
    raw = json.loads(MILESTONE3_PREDICTOR_CONFIG_PATH.read_text())
    raw["activation_max_iter"] = 1
    config = tmp_path / "predictor.json"
    config.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "output"

    exit_code = predictor_main(
        [
            "--run-dir",
            str(source),
            "--structure-config",
            str(REPRESENTATIVE_STRUCTURE_CONFIG_PATH),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 1
    assert set(_bytes(output)) == set(PREDICTOR_ARTIFACT_FILENAMES)
    report = json.loads((output / "evaluation_report.json").read_text())
    assert not report["model_convergence"]["recent_activation"]["converged"]
    assert not report["model_convergence"]["context_activation"]["converged"]
