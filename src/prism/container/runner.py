"""Strict versioned container boundary for one accepted Prism experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn

from prism.experiments.config import ExperimentManifest, load_manifest
from prism.experiments.runner import ExperimentRunError, run_experiments
from prism.native.representative import (
    NativeParityOutputError,
    run_representative_parity,
)


SPEC_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
SPEC_KIND = "accepted_milestone5_native_parity"
SPEC_FIELDS = {
    "schema_version",
    "kind",
    "experiment_manifest",
    "experiment_id",
}


class ContainerSpecError(ValueError):
    """Raised when the container boundary or its versioned spec is unsafe."""


@dataclass(frozen=True, slots=True)
class ContainerRunResult:
    output_dir: Path
    experiment_id: str
    artifact_count: int
    manifest_path: Path


def run_container_experiment(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> ContainerRunResult:
    """Run an accepted frozen pipeline and existing native parity harness."""

    spec_file = Path(spec_path)
    input_boundary = _boundary(
        input_root,
        os.environ.get("PRISM_INPUT_ROOT"),
        spec_file.parent,
        "input",
    )
    output_boundary = _optional_boundary(
        output_root, os.environ.get("PRISM_OUTPUT_ROOT"), "output"
    )
    spec_file = _contained_file(spec_file, input_boundary, "experiment spec")
    destination = _prepare_output(Path(output_dir), output_boundary)
    spec, spec_bytes = _load_spec(spec_file)
    manifest_path = _contained_file(
        input_boundary / spec["experiment_manifest"],
        input_boundary,
        "experiment manifest",
    )
    loaded = load_manifest(manifest_path)
    if not isinstance(loaded, ExperimentManifest):
        raise ContainerSpecError(
            "Phase 1 accepts the frozen Milestone 5 manifest, not actionability specs"
        )
    experiment_id = spec["experiment_id"]
    if experiment_id not in loaded.experiment_ids:
        raise ContainerSpecError(
            f"experiment_id is not present in the accepted manifest: {experiment_id}"
        )
    source_paths = {
        "experiment_spec": spec_file,
        "experiment_manifest": manifest_path,
        "workload_config": _contained_file(
            loaded.base_workload_config, input_boundary, "workload config"
        ),
        "structure_config": _contained_file(
            loaded.structure_config, input_boundary, "structure config"
        ),
        "predictor_config": _contained_file(
            loaded.predictor_config, input_boundary, "predictor config"
        ),
    }

    experiment_root = destination / "experiment"
    execution = run_experiments(
        manifest_path,
        experiment_root,
        experiment_id=experiment_id,
    )
    if execution.failed_count:
        raise ExperimentRunError(
            f"accepted experiment failed: {experiment_id}"
        )
    run_root = experiment_root / "runs" / experiment_id
    status = _read_object(run_root / "run_status.json", "run status")
    if status.get("status") != "completed":
        raise ExperimentRunError(
            f"accepted experiment did not complete: {experiment_id}"
        )

    native_root = destination / "native"
    native = run_representative_parity(
        run_root / "workload",
        run_root / "predictor",
        run_root / "resolved_simulation_config.json",
        native_root,
    )
    gates = native.parity.report["overall_gates"]
    if not gates["overall_parity_passed"]:
        raise RuntimeError("existing Python/native parity gates failed")

    artifacts = _tree_hashes(destination)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "prism": {
            "package_version": _distribution_version(
                "prism-storage", "unknown"
            ),
            "git_revision": os.environ.get("PRISM_GIT_REVISION", "unknown"),
        },
        "experiment": {
            "kind": SPEC_KIND,
            "experiment_id": experiment_id,
            "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "source_sha256": {
                name: _sha256(path) for name, path in sorted(source_paths.items())
            },
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "os": _os_identification(),
            "machine": platform.machine(),
            "dependency_versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit-learn": _distribution_version(
                    "scikit-learn", sklearn.__version__
                ),
            },
        },
        "native_build": {
            "build_type": os.environ.get("PRISM_NATIVE_BUILD_TYPE", "unknown"),
            "compiler_id": os.environ.get("PRISM_NATIVE_COMPILER_ID", "unknown"),
            "compiler_version": os.environ.get(
                "PRISM_NATIVE_COMPILER_VERSION", "unknown"
            ),
            "extension_imported": True,
            "parity_passed": True,
        },
        "artifacts": artifacts,
    }
    manifest_path_out = destination / "run_manifest.json"
    _write_json(manifest_path_out, manifest)
    return ContainerRunResult(
        destination,
        experiment_id,
        len(artifacts),
        manifest_path_out,
    )


def _load_spec(path: Path) -> tuple[dict[str, Any], bytes]:
    raw_bytes = path.read_bytes()
    try:
        value = json.loads(raw_bytes, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ContainerSpecError(f"invalid experiment spec JSON: {error.msg}") from error
    if not isinstance(value, dict) or set(value) != SPEC_FIELDS:
        raise ContainerSpecError("experiment spec fields do not match schema version 1")
    if value["schema_version"] != SPEC_SCHEMA_VERSION:
        raise ContainerSpecError("experiment spec schema_version must equal 1")
    if value["kind"] != SPEC_KIND:
        raise ContainerSpecError(f"experiment spec kind must equal {SPEC_KIND}")
    for name in ("experiment_manifest", "experiment_id"):
        if not isinstance(value[name], str) or not value[name]:
            raise ContainerSpecError(f"experiment spec {name} must be a nonempty string")
    manifest = Path(value["experiment_manifest"])
    if manifest.is_absolute() or ".." in manifest.parts:
        raise ContainerSpecError(
            "experiment_manifest must be a contained relative input path"
        )
    return value, raw_bytes


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContainerSpecError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _boundary(
    explicit: str | Path | None,
    environment: str | None,
    fallback: Path,
    name: str,
) -> Path:
    candidate = Path(explicit) if explicit is not None else Path(environment or fallback)
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise ContainerSpecError(f"{name} root must be an existing directory")
    return resolved


def _optional_boundary(
    explicit: str | Path | None, environment: str | None, name: str
) -> Path | None:
    if explicit is None and environment is None:
        return None
    candidate = Path(explicit) if explicit is not None else Path(environment or "")
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise ContainerSpecError(f"{name} root must be an existing directory")
    return resolved


def _contained_file(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ContainerSpecError(f"{description} escapes the read-only input root")
    if not resolved.is_file():
        raise ContainerSpecError(f"{description} is missing or is not a regular file")
    return resolved


def _prepare_output(path: Path, root: Path | None) -> Path:
    resolved = path.resolve()
    if root is not None and not resolved.is_relative_to(root):
        raise ContainerSpecError("output directory escapes the configured output root")
    if resolved.exists():
        if not resolved.is_dir():
            raise ContainerSpecError("output path exists and is not a directory")
        if any(resolved.iterdir()):
            raise ContainerSpecError("output directory must be empty")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentRunError(f"invalid {description}") from error
    if not isinstance(value, dict):
        raise ExperimentRunError(f"{description} must be a JSON object")
    return value


def _os_identification() -> dict[str, str]:
    values: dict[str, str] = {}
    release = Path("/etc/os-release")
    if release.is_file():
        for line in release.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, raw = line.split("=", 1)
            if key in {"ID", "VERSION_ID"}:
                values[key.lower()] = raw.strip().strip('"')
    if not values:
        values["id"] = platform.system().lower()
        values["version_id"] = platform.release()
    return values


def _distribution_version(name: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one versioned accepted Prism experiment as a batch workload."
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_container_experiment(args.spec, args.output_dir)
    except (
        ContainerSpecError,
        ExperimentRunError,
        NativeParityOutputError,
        OSError,
        ValueError,
    ) as error:
        print(f"prism-container-run: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_count": result.artifact_count,
                "experiment_id": result.experiment_id,
                "manifest": result.manifest_path.name,
                "status": "completed",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
