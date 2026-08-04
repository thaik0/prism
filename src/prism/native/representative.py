"""Accepted frozen-artifact execution through the native storage engine."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from prism.native.parity import (
    ParityExecution,
    PolicyParityInputs,
    run_four_policy_parity,
    write_parity_artifacts,
)
from prism.native.payloads import (
    NativeStoreArtifacts,
    PAYLOAD_SCHEMA_VERSION,
    build_verified_native_store,
)
from prism.simulation import load_policy_inputs


class NativeParityOutputError(ValueError):
    """Raised when a native parity output root cannot be used safely."""


@dataclass(frozen=True, slots=True)
class RepresentativeParityResult:
    store_artifacts: NativeStoreArtifacts
    parity: ParityExecution


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_output_root(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise NativeParityOutputError(
                "native parity output directory must be empty"
            )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def run_representative_parity(
    run_dir: str | Path,
    predictor_run_dir: str | Path,
    simulation_config: str | Path,
    output_dir: str | Path,
) -> RepresentativeParityResult:
    """Execute exactly four accepted policies against validated frozen inputs."""

    destination = _prepare_output_root(output_dir)
    policy = load_policy_inputs(
        run_dir,
        predictor_run_dir,
        simulation_config,
        require_scientific_gates=True,
    )
    store_artifacts = build_verified_native_store(
        run_dir,
        destination / "native_store",
        policy.config.fast_capacity_bytes,
        manifest_path=destination / "native_store_manifest.json",
    )
    if (
        tuple(record.record_id for record in store_artifacts.payloads)
        != policy.record_ids
    ):
        raise ValueError("native payload IDs do not match policy inputs")
    if (
        tuple(record.byte_size for record in store_artifacts.payloads)
        != policy.record_sizes
    ):
        raise ValueError("native payload sizes do not match policy inputs")

    inputs = PolicyParityInputs(
        observable_events=policy.observable_events,
        observable_demand=policy.observable_demand,
        record_ids=policy.record_ids,
        record_sizes=policy.record_sizes,
        predicted_record_demand=policy.predicted_record_demand,
        prediction_available=policy.prediction_available,
        config=policy.config,
        validation_start=policy.train_end,
        test_start=policy.validation_end,
        evaluation_end=policy.evaluation_end,
    )
    simulation_path = Path(simulation_config)
    parity = run_four_policy_parity(
        inputs,
        store_artifacts.payloads,
        destination / "native_store",
        source_information={
            "input_kind": "accepted_representative",
            "source_workload_hashes": dict(policy.source_hashes),
            "policy_input_hashes": dict(policy.predictor_hashes),
            "simulation_configuration_sha256": _sha256(simulation_path),
            "resolved_simulation_configuration": policy.config.to_dict(),
            "workload_seed": policy.workload_seed,
            "capacity_bytes": policy.config.fast_capacity_bytes,
            "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "native_store_verification_passed": True,
            "payload_verification_passed": (
                store_artifacts.manifest.payload_verification_passed
            ),
            "validation_start": policy.train_end,
            "test_start": policy.validation_end,
            "evaluation_end": policy.evaluation_end,
        },
    )
    write_parity_artifacts(parity, destination)
    expected = {
        "native_store",
        "native_store_manifest.json",
        "parity_operations.jsonl",
        "parity_report.json",
    }
    if {path.name for path in destination.iterdir()} != expected:
        raise RuntimeError("native parity output violated the four-entry contract")
    return RepresentativeParityResult(store_artifacts, parity)
