"""Determinism and native/Linux semantic parity verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


NUMERICAL_ABSOLUTE_TOLERANCE = 1e-9
SEMANTIC_REPORT_SUFFIXES = (
    "/aggregate_report.json",
    "/experiment_index.json",
    "/workload/hidden_ground_truth.json",
    "/workload/workload_validation.json",
    "/recovery_report.json",
    "/predictor/evaluation_report.json",
    "/simulation/simulation_config.json",
    "/simulation/evaluation_report.json",
    "/parity_report.json",
    "/run_status.json",
)


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
    if mode not in {"repeat", "cross-host", "cross-platform"}:
        raise VerificationError(
            "mode must be repeat, cross-host, or cross-platform"
        )
    left_manifest = validate_output_root(left)
    right_manifest = validate_output_root(right)
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
            "verified_artifact_count": len(left_hashes),
            "absolute_tolerance": 0.0,
            "maximum_absolute_difference": 0.0,
        }

    _same_identity(left_manifest, right_manifest)
    if mode == "cross-host":
        _same_declared_runtime(left_manifest, right_manifest)
    common_paths = set(left_manifest["artifacts"]) & set(right_manifest["artifacts"])
    if set(left_manifest["artifacts"]) != set(right_manifest["artifacts"]):
        raise VerificationError("semantic parity artifact path sets differ")
    numerical = sorted(path for path in common_paths if path.endswith(".npz"))
    semantic = sorted(
        path for path in common_paths if path.endswith(SEMANTIC_REPORT_SUFFIXES)
    )
    exact = sorted(common_paths - set(numerical) - set(semantic))
    if not exact or not numerical or not semantic:
        raise VerificationError("semantic parity artifact classification is incomplete")
    for relative in exact:
        if _sha256(left / relative) != _sha256(right / relative):
            raise VerificationError(f"byte-identical artifact differs: {relative}")
    maximum_difference = 0.0
    for relative in numerical:
        maximum_difference = max(
            maximum_difference,
            _compare_npz(left / relative, right / relative, relative),
        )
    for relative in semantic:
        maximum_difference = max(
            maximum_difference,
            _compare_json(
                _read_json(left / relative),
                _read_json(right / relative),
                relative,
            ),
        )
    return {
        "mode": mode,
        "passed": True,
        "byte_identical_file_count": len(exact),
        "numerical_file_count": len(numerical),
        "semantic_report_count": len(semantic),
        "verified_artifact_count": len(common_paths),
        "absolute_tolerance": NUMERICAL_ABSOLUTE_TOLERANCE,
        "maximum_absolute_difference": maximum_difference,
    }


def validate_output_root(root: str | Path) -> dict[str, Any]:
    """Validate one complete Phase 1 output and return its run manifest."""

    root = Path(root).resolve()
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


def _same_declared_runtime(
    left: dict[str, Any], right: dict[str, Any]
) -> None:
    for field in ("runtime", "native_build"):
        if left.get(field) != right.get(field):
            raise VerificationError(f"declared runtime differs at {field}")


def _compare_npz(left: Path, right: Path, relative: str) -> float:
    maximum_difference = 0.0
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
            if a.dtype != b.dtype:
                raise VerificationError(f"NPZ dtype differs: {relative}:{name}")
            if a.dtype.kind in "fc":
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
                finite = np.isfinite(a) & np.isfinite(b)
                if np.any(finite):
                    maximum_difference = max(
                        maximum_difference,
                        float(np.max(np.abs(a[finite] - b[finite]))),
                    )
            elif not np.array_equal(a, b):
                raise VerificationError(f"discrete array differs: {relative}:{name}")
    return maximum_difference


def _compare_json(left: Any, right: Any, path: str) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            raise VerificationError(f"JSON keys differ: {path}")
        maximum_difference = 0.0
        for key in sorted(left_keys):
            child_path = f"{path}.{key}"
            if _is_hash_key(key):
                _compare_hash_structure(left[key], right[key], child_path)
            else:
                maximum_difference = max(
                    maximum_difference,
                    _compare_json(left[key], right[key], child_path),
                )
        return maximum_difference
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise VerificationError(f"JSON list length differs: {path}")
        maximum_difference = 0.0
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            maximum_difference = max(
                maximum_difference,
                _compare_json(first, second, f"{path}[{index}]"),
            )
        return maximum_difference
    if (
        isinstance(left, int)
        and not isinstance(left, bool)
        and isinstance(right, int)
        and not isinstance(right, bool)
    ):
        if left != right:
            raise VerificationError(f"JSON integer differs: {path}")
        return 0.0
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        first = float(left)
        second = float(right)
        if not math.isfinite(first) or not math.isfinite(second):
            raise VerificationError(f"JSON number is not finite: {path}")
        difference = abs(first - second)
        if difference > NUMERICAL_ABSOLUTE_TOLERANCE:
            raise VerificationError(f"JSON number exceeds tolerance: {path}")
        return difference
    if left != right:
        raise VerificationError(f"JSON value differs: {path}")
    return 0.0


def _compare_hash_structure(left: Any, right: Any, path: str) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise VerificationError(f"JSON hash keys differ: {path}")
        for key in sorted(left):
            _compare_hash_structure(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise VerificationError(f"JSON hash list length differs: {path}")
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            _compare_hash_structure(first, second, f"{path}[{index}]")
        return
    if not isinstance(left, str) or not isinstance(right, str):
        raise VerificationError(f"JSON hash value type differs: {path}")


def _is_hash_key(key: str) -> bool:
    return key == "sha256" or key.endswith("_sha256") or key.endswith("_hashes")


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
        description="Verify Prism repeat determinism or cross-runtime parity."
    )
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("repeat", "cross-host", "cross-platform"),
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
