"""Determinism and native/Linux semantic parity verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


NUMERICAL_ABSOLUTE_TOLERANCE = 1e-9
EXACT_SUFFIXES = (
    "/resolved_workload_config.json",
    "/resolved_simulation_config.json",
    "/workload/config.json",
    "/workload/observable_events.jsonl",
    "/workload/hidden_ground_truth.json",
    "/workload/summary.json",
    "/native_store/store.data",
    "/native_store/store.index",
    "/parity_operations.jsonl",
)
SEMANTIC_REPORT_SUFFIXES = (
    "/recovery_report.json",
    "/predictor/evaluation_report.json",
    "/simulation/evaluation_report.json",
    "/parity_report.json",
    "/run_status.json",
)
HASH_KEYS = {
    "artifact_sha256",
    "predictor_artifact_sha256",
    "source_artifact_sha256",
}


class VerificationError(ValueError):
    """Raised when outputs violate exact or semantic parity."""


def verify_outputs(
    left_root: str | Path,
    right_root: str | Path,
    *,
    mode: str,
) -> dict[str, Any]:
    left = Path(left_root).resolve()
    right = Path(right_root).resolve()
    if mode not in {"repeat", "cross-platform"}:
        raise VerificationError("mode must be repeat or cross-platform")
    left_manifest = _validated_manifest(left)
    right_manifest = _validated_manifest(right)
    if mode == "repeat":
        left_hashes = _tree_hashes(left)
        right_hashes = _tree_hashes(right)
        if left_hashes != right_hashes:
            differing = sorted(
                path
                for path in set(left_hashes) | set(right_hashes)
                if left_hashes.get(path) != right_hashes.get(path)
            )
            raise VerificationError(
                "repeat outputs are not byte-identical: " + ", ".join(differing[:10])
            )
        return {
            "mode": mode,
            "passed": True,
            "byte_identical_file_count": len(left_hashes),
            "numerical_file_count": 0,
            "semantic_report_count": 0,
            "absolute_tolerance": 0.0,
        }

    _same_identity(left_manifest, right_manifest)
    common_paths = set(left_manifest["artifacts"]) & set(right_manifest["artifacts"])
    if set(left_manifest["artifacts"]) != set(right_manifest["artifacts"]):
        raise VerificationError("cross-platform artifact path sets differ")
    exact = sorted(path for path in common_paths if path.endswith(EXACT_SUFFIXES))
    numerical = sorted(path for path in common_paths if path.endswith(".npz"))
    semantic = sorted(
        path for path in common_paths if path.endswith(SEMANTIC_REPORT_SUFFIXES)
    )
    if not exact or not numerical or not semantic:
        raise VerificationError("cross-platform artifact classification is incomplete")
    for relative in exact:
        if _sha256(left / relative) != _sha256(right / relative):
            raise VerificationError(f"byte-identical artifact differs: {relative}")
    for relative in numerical:
        _compare_npz(left / relative, right / relative, relative)
    for relative in semantic:
        _compare_json(
            _read_json(left / relative),
            _read_json(right / relative),
            relative,
        )
    return {
        "mode": mode,
        "passed": True,
        "byte_identical_file_count": len(exact),
        "numerical_file_count": len(numerical),
        "semantic_report_count": len(semantic),
        "absolute_tolerance": NUMERICAL_ABSOLUTE_TOLERANCE,
    }


def _validated_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "run_manifest.json"
    value = _read_json(manifest_path)
    if value.get("schema_version") != 1 or not isinstance(value.get("artifacts"), dict):
        raise VerificationError("run manifest is missing or invalid")
    actual = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    if actual != value["artifacts"]:
        raise VerificationError("run manifest artifact hashes do not match output tree")
    return value


def _same_identity(left: dict[str, Any], right: dict[str, Any]) -> None:
    for keys in (
        ("prism", "package_version"),
        ("prism", "git_revision"),
        ("experiment", "kind"),
        ("experiment", "experiment_id"),
        ("experiment", "spec_sha256"),
    ):
        first: Any = left
        second: Any = right
        for key in keys:
            first = first[key]
            second = second[key]
        if first != second:
            raise VerificationError(f"run identity differs at {'.'.join(keys)}")
    if left["experiment"]["source_sha256"] != right["experiment"]["source_sha256"]:
        raise VerificationError("source configuration hashes differ")


def _compare_npz(left: Path, right: Path, relative: str) -> None:
    with np.load(left, allow_pickle=False) as first, np.load(
        right, allow_pickle=False
    ) as second:
        if set(first.files) != set(second.files):
            raise VerificationError(f"NPZ array names differ: {relative}")
        for name in first.files:
            a = first[name]
            b = second[name]
            if a.shape != b.shape:
                raise VerificationError(f"NPZ shape differs: {relative}:{name}")
            if a.dtype.kind in "fc" or b.dtype.kind in "fc":
                if not np.allclose(
                    a,
                    b,
                    rtol=0.0,
                    atol=NUMERICAL_ABSOLUTE_TOLERANCE,
                    equal_nan=True,
                ):
                    raise VerificationError(
                        f"numerical array exceeds tolerance: {relative}:{name}"
                    )
            elif not np.array_equal(a, b):
                raise VerificationError(f"discrete array differs: {relative}:{name}")


def _compare_json(left: Any, right: Any, path: str) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left) - HASH_KEYS
        right_keys = set(right) - HASH_KEYS
        if left_keys != right_keys:
            raise VerificationError(f"JSON keys differ: {path}")
        for key in sorted(left_keys):
            _compare_json(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise VerificationError(f"JSON list length differs: {path}")
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            _compare_json(first, second, f"{path}[{index}]")
        return
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if abs(float(left) - float(right)) > NUMERICAL_ABSOLUTE_TOLERANCE:
            raise VerificationError(f"JSON number exceeds tolerance: {path}")
        return
    if left != right:
        raise VerificationError(f"JSON value differs: {path}")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON artifact: {path.name}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Prism repeat determinism or native/container parity."
    )
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("repeat", "cross-platform")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_outputs(args.left, args.right, mode=args.mode)
    except (OSError, VerificationError, ValueError) as error:
        print(f"prism-container-verify: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
