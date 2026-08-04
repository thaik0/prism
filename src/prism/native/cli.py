"""Strict command-line entry point for native execution parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from prism.native.fixtures import run_forced_fixture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Prism's synchronous Python/native parity harness."
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="run the built-in deterministic forced fixture",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="empty destination for exactly four deterministic artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.fixture:
        raise SystemExit("--fixture is required until representative inputs are supplied")
    result = run_forced_fixture(args.output_dir)
    gates = result.parity.report["overall_gates"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "policy_order": result.parity.report["policy_order"],
                "total_mismatch_count": gates["total_mismatch_count"],
                "overall_parity_passed": gates["overall_parity_passed"],
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if gates["overall_parity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
