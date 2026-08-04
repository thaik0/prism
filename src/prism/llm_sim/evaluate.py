"""Deterministic canonical artifacts and six-policy result aggregation."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

import numpy as np

from .catalog import BlockCatalog
from .config import IntegrationConfig, ResolvedBudget
from .demand import LogicalDemand
from .policies import POLICY_DISPLAY_NAMES


def write_static_artifacts(
    output_dir: Path,
    config: IntegrationConfig,
    budget: ResolvedBudget,
    catalog: BlockCatalog,
    demand: LogicalDemand,
    *,
    tiny: bool,
    patch_sha256: str,
    source_hashes: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    split = config.split(tiny=tiny)
    _write_json(output_dir / "block_catalog.json", catalog.to_dict())
    _write_deterministic_npz(
        output_dir / "logical_demand.npz",
        {
            "block_ids": np.asarray(demand.block_ids, dtype="U64"),
            "request_ids": demand.request_ids,
            "demand_matrix": demand.demand_matrix,
            "training_seen": demand.training_seen,
            "cold_start": demand.cold_start,
        },
    )
    manifest = {
        "schema_version": 1,
        "upstream": {
            "repository": config.upstream_repository,
            "commit": config.upstream_commit,
            "release": config.upstream_release,
            "license": "MIT",
            "submodules": {
                "astra-sim": config.astra_sim_commit,
                "astra-sim/extern/graph_frontend/chakra": (
                    "bf11884f4a8b042219837f63be44e1425d4a35ef"
                ),
                "astra-sim/extern/helper/fmt": (
                    "7e73566ce7bd3617cd4b14a2e3a19a2a30b524c7"
                ),
                "astra-sim/extern/helper/spdlog": (
                    "63f0875000bf90867aa7c76a831bd834cb7d4b5e"
                ),
                "astra-sim/extern/memory_backend/analytical": (
                    "62313ef09abaf6a9764d13e2684ca61247def14b"
                ),
                "astra-sim/extern/network_backend/analytical": (
                    "5dc02322f04ac42fcfbb5304d5f7f2fa20d25e3e"
                ),
                "astra-sim/extern/network_backend/analytical/extern/googletest": (
                    "1204d634444b0ba6da53201a8b6caf2a502d883c"
                ),
                "astra-sim/extern/network_backend/analytical/extern/yaml-cpp": (
                    "3d2888cc8a45da2f420454ad728cdfad01a3d54f"
                ),
                "astra-sim/extern/network_backend/csg-htsim": (
                    "67cbbbb1cccdbddc400803414127d044ecd55290"
                ),
                "astra-sim/extern/network_backend/ns-3": (
                    "3964b1d41cb5f7b51004c316b0eae1a3713c2498"
                ),
            },
            "container_base": (
                "astrasim/tutorial-micro2024@sha256:"
                "567a0f020f18de9b7391d620041d2fd10120d6e0f1485b5110e0db79c2849305"
            ),
            "container_recipe": "integration/llmservingsim/Dockerfile",
        },
        "dependencies": {
            "python": "3.10.12",
            "chakra": "0.0.4",
            "msgspec": "0.18.6",
            "numpy": "1.23.5",
            "pandas": "1.5.3",
            "protobuf_runtime": "7.35.1",
            "pyinstrument": "4.6.2",
            "pyyaml": "6.0.1",
            "rich": "13.7.1",
            "scikit-learn": "1.3.2",
            "scipy": "1.10.1",
        },
        "patch_sha256": patch_sha256,
        "configuration": {
            "model": config.model,
            "hardware": config.hardware,
            "dtype": config.dtype,
            "kv_cache_dtype": config.kv_cache_dtype,
            "block_size_tokens": config.block_size_tokens,
            "trace": config.tiny_trace if tiny else config.full_trace,
            "cluster": config.cluster_config,
            "seed": config.seed,
        },
        "resolved_budget": budget.to_dict(),
        "split": asdict(split),
        "policy_ids": list(config.policy_ids),
        "logical_demand_sha256": demand.logical_stream_sha256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "canonical_report_rules": {
            "timestamps": False,
            "absolute_machine_paths": False,
            "json_keys_sorted": True,
        },
    }
    _write_json(output_dir / "integration_manifest.json", manifest)
    _write_json(
        output_dir / "experiment_index.json",
        {
            "schema_version": 1,
            "mode": "tiny" if tiny else "full",
            "request_count": split.request_count,
            "block_count": len(catalog.blocks),
            "logical_reference_count": int(demand.demand_matrix.sum()),
            "logical_demand_sha256": demand.logical_stream_sha256,
            "demand_summary": demand.summary(split),
            "runs": [
                {"policy_id": policy_id, "directory": f"runs/{policy_id}"}
                for policy_id in config.policy_ids
            ],
        },
    )


def aggregate_experiment(output_dir: Path, config: IntegrationConfig, *, tiny: bool) -> None:
    split = config.split(tiny=tiny)
    all_rows: list[dict[str, Any]] = []
    policy_results: dict[str, Any] = {}
    decisions_by_policy: dict[str, list[dict[str, Any]]] = {}
    demand_hashes: set[str] = set()
    for policy_id in config.policy_ids:
        run_dir = output_dir / "runs" / policy_id
        hook = _read_json(run_dir / "hook_results.json")
        rows = _read_jsonl(run_dir / "request_cache_metrics.jsonl")
        decisions = _read_jsonl(run_dir / "placement_decisions.jsonl")
        if len(rows) != split.request_count:
            raise ValueError(f"{policy_id} produced an incomplete request metric stream")
        if len(decisions) != split.request_count:
            raise ValueError(f"{policy_id} produced an incomplete placement stream")
        demand_hashes.add(str(hook["logical_demand_sha256"]))
        all_rows.extend(rows)
        test = [row for row in rows if row["split"] == "test"]
        test_decisions = [
            row for row in decisions if int(row["request_id"]) >= split.validation_end
        ]
        decisions_by_policy[policy_id] = test_decisions
        policy_results[policy_id] = _policy_summary(
            policy_id, test, test_decisions, hook, config
        )
    if len(demand_hashes) != 1:
        raise ValueError(f"policy logical-demand hashes differ: {sorted(demand_hashes)}")
    _write_jsonl(output_dir / "request_metrics.jsonl", all_rows)
    result = {
        "schema_version": 1,
        "logical_demand_sha256": next(iter(demand_hashes)),
        "policies": policy_results,
    }
    _add_cross_policy_diagnostics(result, decisions_by_policy)
    _write_json(output_dir / "policy_results.json", result)
    (output_dir / "comparison_report.md").write_text(
        _comparison_report(result, tiny=tiny), encoding="utf-8"
    )


def source_hashes(config: IntegrationConfig) -> dict[str, str]:
    paths = {
        "integration_config": config.config_path,
        "container_recipe": config.project_root
        / "integration/llmservingsim/Dockerfile",
        "runtime_recipe": config.project_root
        / "integration/llmservingsim/Runtime.Dockerfile",
        "integration_patch": config.project_root
        / "integration/llmservingsim/prism_hook.patch",
        "cluster_config": config.upstream_root / config.cluster_config,
        "full_trace": config.upstream_root / config.full_trace,
        "tiny_trace": config.upstream_root / config.tiny_trace,
        "hardware_profile_meta": config.upstream_root
        / "profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/meta.yaml",
        "upstream_license": config.upstream_root / "LICENSE",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tree_hashes(root: Path) -> dict[str, str]:
    ignored = {"simulator_stdout.log", "hook_payload.json"}
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in ignored
    }


def _policy_summary(
    policy_id: str,
    test: list[dict[str, Any]],
    test_decisions: list[dict[str, Any]],
    hook: dict[str, Any],
    config: IntegrationConfig,
) -> dict[str, Any]:
    if not test:
        raise ValueError(f"{policy_id} has no test requests")
    ttft = np.asarray([row["ttft_ns"] for row in test], dtype=np.float64)
    latency = np.asarray([row["latency_ns"] for row in test], dtype=np.float64)
    tpot = np.asarray([row["tpot_ns"] for row in test], dtype=np.float64)
    first_arrival = min(row["arrival_ns"] for row in test)
    last_end = max(row["end_time_ns"] for row in test)
    elapsed_s = (last_end - first_arrival) / 1e9
    if elapsed_s <= 0:
        raise ValueError("test interval must be positive")
    counters = {
        key: sum(int(row[key]) for row in test)
        for key in (
            "gpu_hit_blocks", "gpu_hit_tokens", "host_hit_blocks", "host_hit_tokens",
            "recomputed_blocks", "recomputed_tokens", "host_to_gpu_bytes",
            "targeted_promoted_blocks",
        )
    }
    cold = {
        outcome: {
            classification: sum(
                row["cold_start_outcomes"][outcome][classification] for row in test
            )
            for classification in ("training_seen_blocks", "cold_start_blocks")
        }
        for outcome in ("gpu", "host", "recompute")
    }
    training_seen_demand = sum(
        cold[outcome]["training_seen_blocks"] for outcome in cold
    )
    cold_start_demand = sum(cold[outcome]["cold_start_blocks"] for outcome in cold)
    total_demand = training_seen_demand + cold_start_demand
    cold["demand"] = {
        "training_seen_blocks": training_seen_demand,
        "cold_start_blocks": cold_start_demand,
        "training_seen_share": training_seen_demand / total_demand,
        "cold_start_share": cold_start_demand / total_demand,
        "latency_contribution": None,
        "latency_contribution_note": (
            "the simulator reports request latency, not per-prefix-block latency"
        ),
    }
    cpu_bw_gib_s = 256
    counters["host_to_gpu_transfer_time_ns"] = int(
        round(counters["host_to_gpu_bytes"] / (cpu_bw_gib_s * 1024**3) * 1e9)
    )
    return {
        "display_name": POLICY_DISPLAY_NAMES[policy_id],
        "test_request_count": len(test),
        "latency_ns": {
            "ttft": _distribution(ttft, include_median=True),
            "end_to_end": _distribution(latency, include_median=True),
            "tpot": _distribution(tpot, include_median=False),
        },
        "throughput": {
            "requests_per_second": len(test) / elapsed_s,
            "prompt_tokens_per_second": sum(row["input_tokens"] for row in test) / elapsed_s,
            "output_tokens_per_second": sum(row["output_tokens"] for row in test) / elapsed_s,
        },
        "cache_execution": {
            **counters,
            "native_batch_load_bytes_all_splits": hook["native_batch_load_bytes"],
            "native_batch_evict_bytes_all_splits": hook["native_batch_evict_bytes"],
            "native_gpu_spill_blocks_all_splits": int(
                hook["native_batch_evict_bytes"] // config.resolve_budget().block_bytes
            ),
            "gpu_spills_evictions_all_splits": hook["blocks_evicted"],
            "max_reusable_gpu_occupancy_blocks": hook["max_reusable_occupancy_blocks"],
            "target_changes_all_splits": hook.get(
                "target_changes", hook["placement_changes"]
            ),
            "placement_changes_all_splits": hook["placement_changes"],
            "native_reusable_residency_changes_all_splits": hook.get(
                "native_reusable_residency_changes", hook["placement_changes"]
            ),
            "blocks_promoted_all_splits": hook["blocks_promoted"],
        },
        "cold_start": cold,
        "diagnostics": {
            "model_fit_status": hook["model_fit_status"],
            "reusable_budget_blocks": hook["reusable_budget_blocks"],
            "demand_covered_by_selected_targets": _target_demand_coverage(
                test_decisions
            ),
            "promotion_usefulness": (
                counters["targeted_promoted_blocks"] / counters["host_hit_blocks"]
                if counters["host_hit_blocks"]
                else None
            ),
            "transfer_time_basis": (
                "bytes emitted by native scheduler kv_load at configured 256 GiB/s host bandwidth; "
                "request latency includes ASTRA-Sim execution"
            ),
        },
    }


def _distribution(values: np.ndarray, *, include_median: bool) -> dict[str, float]:
    result = {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
    }
    if include_median:
        result["median"] = float(np.median(values))
        result["p99"] = float(np.percentile(values, 99))
    return result


def _add_cross_policy_diagnostics(
    result: dict[str, Any], decisions_by_policy: dict[str, list[dict[str, Any]]]
) -> None:
    policies = result["policies"]
    lru = policies["llmservingsim_lru"]["cache_execution"]
    for policy in policies.values():
        cache = policy["cache_execution"]
        policy["diagnostics"]["movement_avoided_vs_lru"] = (
            lru["native_reusable_residency_changes_all_splits"]
            - cache["native_reusable_residency_changes_all_splits"]
        )
    frozen = decisions_by_policy["validation_final_frozen_prism"]
    predictive = decisions_by_policy["predictive_greedy_prism"]
    disagreement = _target_comparison(predictive, frozen)
    policies["validation_final_frozen_prism"]["diagnostics"][
        "predictive_vs_frozen_target_disagreement"
    ] = disagreement
    policies["predictive_greedy_prism"]["diagnostics"][
        "predictive_vs_frozen_target_disagreement"
    ] = disagreement
    oracle = decisions_by_policy["oracle_greedy"]
    for policy_id in (
        "training_popularity_static_prism",
        "validation_final_frozen_prism",
        "predictive_greedy_prism",
    ):
        policies[policy_id]["diagnostics"]["prism_vs_oracle_target"] = (
            _target_comparison(decisions_by_policy[policy_id], oracle)
        )
    result["cross_policy_diagnostics"] = {
        "predictive_vs_frozen_target": disagreement,
        "prism_vs_oracle_target": {
            policy_id: policies[policy_id]["diagnostics"]["prism_vs_oracle_target"]
            for policy_id in (
                "training_popularity_static_prism",
                "validation_final_frozen_prism",
                "predictive_greedy_prism",
            )
        },
    }


def _target_demand_coverage(decisions: list[dict[str, Any]]) -> float | None:
    logical = sum(int(row["logical_demand_blocks"]) for row in decisions)
    if not logical or all(row["target_block_ids"] is None for row in decisions):
        return None
    return sum(int(row["target_demand_blocks"]) for row in decisions) / logical


def _target_comparison(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, float | int]:
    if [row["request_id"] for row in left] != [row["request_id"] for row in right]:
        raise ValueError("target comparison request IDs differ")
    jaccards: list[float] = []
    disagreeing = 0
    for left_row, right_row in zip(left, right, strict=True):
        left_target = set(left_row["target_block_ids"] or ())
        right_target = set(right_row["target_block_ids"] or ())
        union = left_target | right_target
        jaccards.append(len(left_target & right_target) / len(union) if union else 1.0)
        disagreeing += left_target != right_target
    return {
        "request_count": len(left),
        "disagreeing_request_count": disagreeing,
        "disagreement_rate": disagreeing / len(left) if left else 0.0,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
    }


def _comparison_report(result: dict[str, Any], *, tiny: bool) -> str:
    lines = [
        "# Milestone 8 KV-cache comparison",
        "",
        f"Mode: {'tiny smoke' if tiny else 'full 300-request experiment'}.",
        "",
        (
            "| Policy | Mean TTFT (ms) | P95 TTFT (ms) | Recomputed blocks | "
            "Host→GPU MiB | Target coverage | Native movement |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in result["policies"].values():
        ttft = policy["latency_ns"]["ttft"]
        cache = policy["cache_execution"]
        coverage = policy["diagnostics"]["demand_covered_by_selected_targets"]
        coverage_text = "n/a" if coverage is None else f"{coverage:.1%}"
        lines.append(
            f"| {policy['display_name']} | {ttft['mean']/1e6:.3f} | "
            f"{ttft['p95']/1e6:.3f} | {cache['recomputed_blocks']} | "
            f"{cache['host_to_gpu_bytes']/1024**2:.3f} | {coverage_text} | "
            f"{cache['native_reusable_residency_changes_all_splits']} |"
        )
    if not tiny:
        lines.extend(["", "## Scientific questions", "", *_scientific_lines(result)])
    lines.extend(
        [
            "",
            (
                "TTFT is the principal latency metric. These are outcomes from the pinned "
                "simulator configuration, not a completion gate. Prism is not required to "
                "beat native LRU. Cold-start blocks receive no Prism forecast or rescue policy."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _scientific_lines(result: dict[str, Any]) -> list[str]:
    policies = result["policies"]
    lru = policies["llmservingsim_lru"]
    static = policies["training_popularity_static_prism"]
    frozen = policies["validation_final_frozen_prism"]
    predictive = policies["predictive_greedy_prism"]
    lfu = policies["lfu"]
    oracle = policies["oracle_greedy"]

    def answer(number: int, name: str, candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
        candidate_cache = candidate["cache_execution"]
        baseline_cache = baseline["cache_execution"]
        ttft_delta = _relative_delta(
            candidate["latency_ns"]["ttft"]["mean"],
            baseline["latency_ns"]["ttft"]["mean"],
        )
        transfer_delta = _relative_delta(
            candidate_cache["host_to_gpu_bytes"], baseline_cache["host_to_gpu_bytes"]
        )
        return (
            f"{number}. **{name}:** mean TTFT {ttft_delta}; host→GPU bytes "
            f"{transfer_delta}; recomputed blocks {candidate_cache['recomputed_blocks']} "
            f"versus {baseline_cache['recomputed_blocks']}."
        )

    cold = lru["cold_start"]["demand"]
    oracle_jaccard = predictive["diagnostics"]["prism_vs_oracle_target"]
    predictive_coverage = predictive["diagnostics"][
        "demand_covered_by_selected_targets"
    ]
    return [
        answer(1, "Static Prism versus native LRU", static, lru),
        answer(2, "Validation-final frozen versus predictive", frozen, predictive),
        answer(3, "LFU versus native LRU", lfu, lru),
        (
            "4. **Cold start:** "
            f"{cold['cold_start_blocks']} of "
            f"{cold['cold_start_blocks'] + cold['training_seen_blocks']} test block "
            f"references ({cold['cold_start_share']:.1%}) were absent from training."
        ),
        answer(5, "Oracle opportunity versus predictive", oracle, predictive)
        + (
            f" Mean target Jaccard was {oracle_jaccard['mean_jaccard']:.3f}."
        ),
        (
            "6. **Coverage versus movement:** predictive selected targets covered "
            f"{predictive_coverage:.1%} of test logical demand; it avoided "
            f"{predictive['diagnostics']['movement_avoided_vs_lru']} native reusable "
            "residency changes versus LRU. These simultaneous diagnostics do not by "
            "themselves establish that either mechanism caused a latency difference."
        ),
    ]


def _relative_delta(value: float | int, baseline: float | int) -> str:
    if baseline == 0:
        return "unchanged at zero" if value == 0 else f"changed from zero to {value}"
    change = (value - baseline) / baseline
    direction = "higher" if change > 0 else "lower" if change < 0 else "unchanged"
    return f"{abs(change):.1%} {direction}"


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
