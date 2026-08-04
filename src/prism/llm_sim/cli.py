"""Milestone 8 six-policy LLMServingSim experiment CLI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
RUNTIME_IMAGE = "prism-llmservingsim-m8-runtime:2c2042ce"


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent policy processes to run concurrently (default: 1)",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.workers <= 6:
        parser.error("--workers must be between 1 and 6")

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

    with tempfile.TemporaryDirectory(prefix="prism-m8-runtime-") as temp:
        runtime_root = Path(temp)
        runtime_project = runtime_root / "project"
        runtime_upstream = runtime_project / "third_party/LLMServingSim"
        runtime_output = runtime_root / "output"
        (runtime_project / "third_party").mkdir(parents=True)
        runtime_output.mkdir()
        (runtime_output / "runs").mkdir()
        shutil.copytree(config.project_root / "src", runtime_project / "src")
        shutil.copytree(config.project_root / "configs", runtime_project / "configs")
        shutil.copytree(config.upstream_root, runtime_upstream, symlinks=False)
        _apply_patch(runtime_upstream, patch_path)
        if args.runtime == "docker":
            if not args.skip_container_build:
                _ensure_container_image(config)
            if not args.skip_upstream_build:
                _build_runtime_image(
                    runtime_project,
                    config.project_root
                    / "integration/llmservingsim/Runtime.Dockerfile",
                )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                policy_id: executor.submit(
                    _run_policy,
                    config,
                    runtime_output,
                    runtime_project,
                    policy_id,
                    tiny=args.tiny,
                    runtime=args.runtime,
                )
                for policy_id in config.policy_ids
            }
            for policy_id in config.policy_ids:
                futures[policy_id].result()
        for policy_id in config.policy_ids:
            shutil.copytree(
                runtime_output / "runs" / policy_id,
                output_dir / "runs" / policy_id,
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
        [
            "docker", "build", "--platform", "linux/amd64", "-t", IMAGE,
            "-f", str(dockerfile_dir / "Dockerfile"), ".",
        ],
        cwd=dockerfile_dir,
    )


def _build_runtime_image(runtime_project: Path, dockerfile: Path) -> None:
    (runtime_project / ".dockerignore").write_text(
        "\n".join(
            (
                "**/.git",
                "third_party/LLMServingSim/docs",
                "third_party/LLMServingSim/bench",
                "third_party/LLMServingSim/outputs",
                "",
            )
        ),
        encoding="utf-8",
    )
    _run_checked(
        [
            "docker", "build", "--platform", "linux/amd64",
            "-t", RUNTIME_IMAGE, "-f", str(dockerfile), ".",
        ],
        cwd=runtime_project,
    )


def _run_policy(
    config: IntegrationConfig,
    runtime_output: Path,
    runtime_project: Path,
    policy_id: str,
    *,
    tiny: bool,
    runtime: str,
) -> None:
    run_dir = runtime_output / "runs" / policy_id
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
            "config": str(runtime_project / "configs/milestone8_llmservingsim.json"),
            "output_dir": str(run_dir),
            "policy_id": policy_id,
            "tiny": tiny,
        }
    payload_path = run_dir / "hook_payload.json"
    payload_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    payload_argument = json.dumps(payload, sort_keys=True, separators=(",", ":"))
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
            payload_argument
            if runtime == "docker"
            else str(payload_path)
        ),
    ]
    if runtime == "docker":
        command = []
        cwd = None
    else:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        runtime_upstream = runtime_project / "third_party/LLMServingSim"
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(runtime_project / "src"), str(runtime_upstream))
        )
        command = sim_args
        cwd = runtime_upstream
    log_path = run_dir / "simulator_stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        if runtime == "docker":
            build_root = runtime_output / "policy_build" / policy_id
            build_root.mkdir(parents=True)
            dockerfile = build_root / "Dockerfile"
            dockerfile.write_text(
                "\n".join(
                    (
                        f"FROM {RUNTIME_IMAGE}",
                        "ENV PYTHONHASHSEED=0",
                        (
                            "ENV PYTHONPATH=/app/prism/src:"
                            "/app/prism/third_party/LLMServingSim"
                        ),
                        "WORKDIR /app/prism/third_party/LLMServingSim",
                        f"RUN {json.dumps(sim_args)}",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            policy_image = f"prism-llmservingsim-m8-policy:{policy_id}"
            completed = subprocess.run(
                [
                    "docker", "build", "--platform", "linux/amd64", "--no-cache",
                    "-t", policy_image, "-f", str(dockerfile), ".",
                ],
                cwd=build_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            container_id = None
            try:
                if completed.returncode == 0:
                    container_id = _create_container([policy_image, "true"])
                    _run_checked(
                        [
                            "docker", "cp",
                            f"{container_id}:/app/output/runs/{policy_id}/.",
                            str(run_dir),
                        ]
                    )
            finally:
                if container_id is not None:
                    _remove_container(container_id)
                subprocess.run(
                    ["docker", "image", "rm", "-f", policy_image],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        else:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
    if completed.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(f"policy {policy_id} failed with {completed.returncode}:\n{tail}")
    generated_inputs = run_dir / "simulator_inputs"
    if generated_inputs.exists():
        shutil.rmtree(generated_inputs)


def _capture(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, capture_output=True
    ).stdout.strip()


def _run_checked(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _create_container(command: list[str]) -> str:
    created = subprocess.run(
        ["docker", "create", "--platform", "linux/amd64", *command],
        check=True,
        text=True,
        capture_output=True,
    )
    container_id = created.stdout.strip()
    if not container_id:
        raise RuntimeError("docker create returned no container ID")
    return container_id


def _remove_container(container_id: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
