"""Milestone 8 six-policy LLMServingSim experiment CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .catalog import build_block_catalog
from .config import IntegrationConfig
from .demand import build_logical_demand
from .evaluate import (
    aggregate_experiment,
    file_sha256,
    source_hashes,
    write_static_artifacts,
)


IMAGE = "prism-llmservingsim-m8:2c2042ce"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--runtime", choices=("docker", "prepared"), default="docker",
        help="docker builds/runs the pinned environment; prepared uses the current Python",
    )
    parser.add_argument("--skip-container-build", action="store_true")
    parser.add_argument("--skip-upstream-build", action="store_true")
    args = parser.parse_args(argv)

    config = IntegrationConfig.from_json(args.config)
    _verify_pins(config)
    budget = config.resolve_budget()
    split = config.split(tiny=args.tiny)
    catalog = build_block_catalog(
        config.trace_path(tiny=args.tiny),
        namespace=f"{config.model}|{config.dtype}|b{config.block_size_tokens}",
        block_size_tokens=config.block_size_tokens,
        block_bytes=budget.block_bytes,
    )
    demand = build_logical_demand(catalog, split)
    output_dir = Path(args.output_dir).resolve()
    patch_path = config.project_root / "integration/llmservingsim/prism_hook.patch"
    write_static_artifacts(
        output_dir,
        config,
        budget,
        catalog,
        demand,
        tiny=args.tiny,
        patch_sha256=file_sha256(patch_path),
        source_hashes=source_hashes(config),
    )
    (output_dir / "runs").mkdir()

    with tempfile.TemporaryDirectory(prefix="prism-m8-serving-") as temp:
        patched_serving = Path(temp) / "serving"
        shutil.copytree(config.upstream_root / "serving", patched_serving)
        _apply_patch(Path(temp), patch_path)
        if args.runtime == "docker":
            if not args.skip_container_build:
                _ensure_container_image(config)
            if not args.skip_upstream_build:
                _ensure_upstream_binary(config)
        for policy_id in config.policy_ids:
            _run_policy(
                config,
                output_dir,
                patched_serving,
                policy_id,
                tiny=args.tiny,
                runtime=args.runtime,
            )
    aggregate_experiment(output_dir, config, tiny=args.tiny)
    print(f"completed deterministic six-policy experiment: {output_dir}")
    return 0


def _verify_pins(config: IntegrationConfig) -> None:
    upstream = _capture(["git", "-C", str(config.upstream_root), "rev-parse", "HEAD"])
    if upstream != config.upstream_commit:
        raise RuntimeError(f"upstream checkout {upstream} differs from frozen pin")
    astra = _capture(
        ["git", "-C", str(config.upstream_root), "rev-parse", "HEAD:astra-sim"]
    )
    if astra != config.astra_sim_commit:
        raise RuntimeError(f"ASTRA-Sim gitlink {astra} differs from frozen pin")


def _apply_patch(temp_root: Path, patch_path: Path) -> None:
    completed = subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(patch_path)],
        cwd=temp_root,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"cannot apply isolated upstream patch:\n{completed.stdout}{completed.stderr}"
        )


def _ensure_container_image(config: IntegrationConfig) -> None:
    exists = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True
    ).returncode == 0
    if exists:
        return
    dockerfile_dir = config.project_root / "integration/llmservingsim"
    _run_checked(
        ["docker", "build", "-t", IMAGE, "-f", str(dockerfile_dir / "Dockerfile"), "."],
        cwd=dockerfile_dir,
    )


def _ensure_upstream_binary(config: IntegrationConfig) -> None:
    binary = (
        config.upstream_root
        / "astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
    )
    if binary.is_file():
        return
    _run_checked(
        [
            "docker", "run", "--rm",
            "-v", f"{config.upstream_root}:/app/LLMServingSim",
            "-w", "/app/LLMServingSim",
            IMAGE,
            "bash", "scripts/compile.sh",
        ]
    )
    if not binary.is_file():
        raise RuntimeError("upstream build completed without the analytical binary")


def _run_policy(
    config: IntegrationConfig,
    output_dir: Path,
    patched_serving: Path,
    policy_id: str,
    *,
    tiny: bool,
    runtime: str,
) -> None:
    run_dir = output_dir / "runs" / policy_id
    run_dir.mkdir()
    if runtime == "docker":
        payload = {
            "config": "/app/prism/configs/milestone8_llmservingsim.json",
            "output_dir": f"/app/output/runs/{policy_id}",
            "policy_id": policy_id,
            "tiny": tiny,
        }
    else:
        payload = {
            "config": str(config.config_path),
            "output_dir": str(run_dir),
            "policy_id": policy_id,
            "tiny": tiny,
        }
    payload_path = run_dir / "hook_payload.json"
    payload_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    split = config.split(tiny=tiny)
    trace = config.tiny_trace if tiny else config.full_trace
    sim_args = [
        "python3", "-m", "serving",
        "--cluster-config", config.cluster_config,
        "--dtype", config.dtype,
        "--block-size", str(config.block_size_tokens),
        "--enable-prefix-caching",
        "--prefix-storage", "CPU",
        "--dataset", trace,
        "--num-reqs", str(split.request_count),
        "--output", (
            f"/app/output/runs/{policy_id}/simulator_request_metrics.csv"
            if runtime == "docker"
            else str(run_dir / "simulator_request_metrics.csv")
        ),
        "--run-id", f"m8-{policy_id}",
        "--inputs-root", (
            f"/app/output/runs/{policy_id}/simulator_inputs"
            if runtime == "docker"
            else str(run_dir / "simulator_inputs")
        ),
        "--cleanup-inputs",
        "--log-level", "WARNING",
        "--prism-hook-config", (
            f"/app/output/runs/{policy_id}/hook_payload.json"
            if runtime == "docker"
            else str(payload_path)
        ),
    ]
    if runtime == "docker":
        command = [
            "docker", "run", "--rm",
            "-e", "PYTHONHASHSEED=0",
            "-e", "PYTHONPATH=/app/prism/src:/app/LLMServingSim",
            "-v", f"{config.project_root}:/app/prism",
            "-v", f"{config.upstream_root}:/app/LLMServingSim",
            "-v", f"{patched_serving}:/app/LLMServingSim/serving",
            "-v", f"{output_dir}:/app/output",
            "-w", "/app/LLMServingSim",
            IMAGE,
            *sim_args,
        ]
        cwd = None
    else:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(config.project_root / "src"), str(Path(patched_serving).parent))
        )
        command = sim_args
        cwd = Path(patched_serving).parent
    log_path = run_dir / "simulator_stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=None if runtime == "docker" else environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(f"policy {policy_id} failed with {completed.returncode}:\n{tail}")


def _capture(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, capture_output=True
    ).stdout.strip()


def _run_checked(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
