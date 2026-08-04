"""Validated frozen configuration and capacity resolution for Milestone 8."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class IntegrationConfigError(ValueError):
    """Raised when the integration boundary is not the frozen configuration."""


EXPECTED_POLICIES = (
    "llmservingsim_lru",
    "lfu",
    "training_popularity_static_prism",
    "validation_final_frozen_prism",
    "predictive_greedy_prism",
    "oracle_greedy",
)


@dataclass(frozen=True)
class RequestSplit:
    training_end: int
    validation_end: int
    request_count: int

    def __post_init__(self) -> None:
        if not (0 < self.training_end < self.validation_end < self.request_count):
            raise IntegrationConfigError("request split must be strictly chronological")


@dataclass(frozen=True)
class ResolvedBudget:
    total_gpu_bytes: int
    weight_bytes: int
    kv_bytes_per_token: int
    block_size_tokens: int
    block_bytes: int
    post_weight_blocks: int
    active_request_reservation_blocks: int
    remaining_blocks: int
    reusable_gpu_blocks: int
    host_capacity_bytes: int
    host_capacity_blocks: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.total_gpu_bytes,
            self.weight_bytes,
            self.kv_bytes_per_token,
            self.block_size_tokens,
            self.block_bytes,
            self.post_weight_blocks,
            self.active_request_reservation_blocks,
            self.remaining_blocks,
            self.reusable_gpu_blocks,
            self.host_capacity_bytes,
            self.host_capacity_blocks,
        )
        if any(isinstance(value, bool) or value <= 0 for value in integer_fields):
            raise IntegrationConfigError("resolved budget fields must be positive integers")
        if self.weight_bytes >= self.total_gpu_bytes:
            raise IntegrationConfigError("model weights must fit in modeled GPU memory")
        if self.block_bytes != self.kv_bytes_per_token * self.block_size_tokens:
            raise IntegrationConfigError("block byte calculation is inconsistent")
        if self.reusable_gpu_blocks > self.remaining_blocks:
            raise IntegrationConfigError("reusable GPU budget exceeds remaining capacity")

    def to_dict(self) -> dict[str, int]:
        return {
            "total_gpu_bytes": self.total_gpu_bytes,
            "weight_bytes": self.weight_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "block_size_tokens": self.block_size_tokens,
            "block_bytes": self.block_bytes,
            "post_weight_blocks": self.post_weight_blocks,
            "active_request_reservation_blocks": self.active_request_reservation_blocks,
            "remaining_blocks": self.remaining_blocks,
            "reusable_gpu_blocks": self.reusable_gpu_blocks,
            "host_capacity_bytes": self.host_capacity_bytes,
            "host_capacity_blocks": self.host_capacity_blocks,
        }


@dataclass(frozen=True)
class IntegrationConfig:
    schema_version: int
    config_path: Path
    project_root: Path
    upstream_root: Path
    upstream_repository: str
    upstream_commit: str
    upstream_release: str
    astra_sim_commit: str
    cluster_config: str
    model: str
    hardware: str
    dtype: str
    kv_cache_dtype: str
    block_size_tokens: int
    full_trace: str
    tiny_trace: str
    full_split: RequestSplit
    tiny_split: RequestSplit
    active_request_reservation_blocks: int
    reusable_gpu_fraction: float
    structure_components: int
    seed: int
    fast_read_cost: float
    slow_read_cost: float
    promotion_cost_per_byte: float
    anonymous_user: str
    request_type: str
    policy_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "IntegrationConfig":
        config_path = Path(path).resolve()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrationConfigError(f"cannot read integration config: {exc}") from exc
        if not isinstance(raw, dict):
            raise IntegrationConfigError("integration config must be a JSON object")
        expected = {
            "schema_version", "upstream_root", "upstream_repository",
            "upstream_commit", "upstream_release", "astra_sim_commit",
            "cluster_config", "model", "hardware", "dtype", "kv_cache_dtype",
            "block_size_tokens", "full_trace", "tiny_trace", "full_split",
            "tiny_split", "active_request_reservation_blocks",
            "reusable_gpu_fraction", "structure_components", "seed",
            "fast_read_cost", "slow_read_cost", "promotion_cost_per_byte",
            "anonymous_user", "request_type", "policy_ids",
        }
        unknown = set(raw) - expected
        missing = expected - set(raw)
        if unknown or missing:
            raise IntegrationConfigError(
                f"integration config keys differ: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        project_root = config_path.parent.parent
        split = lambda value: RequestSplit(
            training_end=_strict_int(value, "training_end"),
            validation_end=_strict_int(value, "validation_end"),
            request_count=_strict_int(value, "request_count"),
        )
        config = cls(
            schema_version=_strict_int(raw, "schema_version"),
            config_path=config_path,
            project_root=project_root,
            upstream_root=(project_root / _strict_str(raw, "upstream_root")).resolve(),
            upstream_repository=_strict_str(raw, "upstream_repository"),
            upstream_commit=_strict_str(raw, "upstream_commit"),
            upstream_release=_strict_str(raw, "upstream_release"),
            astra_sim_commit=_strict_str(raw, "astra_sim_commit"),
            cluster_config=_strict_str(raw, "cluster_config"),
            model=_strict_str(raw, "model"),
            hardware=_strict_str(raw, "hardware"),
            dtype=_strict_str(raw, "dtype"),
            kv_cache_dtype=_strict_str(raw, "kv_cache_dtype"),
            block_size_tokens=_strict_int(raw, "block_size_tokens"),
            full_trace=_strict_str(raw, "full_trace"),
            tiny_trace=_strict_str(raw, "tiny_trace"),
            full_split=split(_strict_mapping(raw, "full_split")),
            tiny_split=split(_strict_mapping(raw, "tiny_split")),
            active_request_reservation_blocks=_strict_int(
                raw, "active_request_reservation_blocks"
            ),
            reusable_gpu_fraction=_strict_float(raw, "reusable_gpu_fraction"),
            structure_components=_strict_int(raw, "structure_components"),
            seed=_strict_int(raw, "seed", allow_zero=True),
            fast_read_cost=_strict_float(raw, "fast_read_cost"),
            slow_read_cost=_strict_float(raw, "slow_read_cost"),
            promotion_cost_per_byte=_strict_float(
                raw, "promotion_cost_per_byte", allow_zero=True
            ),
            anonymous_user=_strict_str(raw, "anonymous_user"),
            request_type=_strict_str(raw, "request_type"),
            policy_ids=tuple(raw["policy_ids"]),
        )
        config.validate_frozen_boundary()
        return config

    def validate_frozen_boundary(self) -> None:
        if self.schema_version != 1:
            raise IntegrationConfigError("unsupported integration config schema")
        frozen = {
            "upstream_repository": "https://github.com/casys-kaist/LLMServingSim.git",
            "upstream_commit": "2c2042ce4bf1b0283ebeed1db95db6f25e3e7511",
            "model": "meta-llama/Llama-3.1-8B",
            "hardware": "RTXPRO6000",
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "block_size_tokens": 16,
            "full_trace": "workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl",
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise IntegrationConfigError(f"{name} must remain frozen as {expected!r}")
        if self.full_split != RequestSplit(180, 240, 300):
            raise IntegrationConfigError("full trace split must be 180/240/300")
        if self.policy_ids != EXPECTED_POLICIES:
            raise IntegrationConfigError("policy_ids must contain exactly the six fixed policies")
        if not (0.0 < self.reusable_gpu_fraction <= 1.0):
            raise IntegrationConfigError("reusable_gpu_fraction must be in (0, 1]")
        if self.structure_components != 4:
            raise IntegrationConfigError("Milestone 8 reuses the accepted four-factor learner")
        if self.slow_read_cost <= self.fast_read_cost:
            raise IntegrationConfigError("slow read cost must exceed fast read cost")
        if not self.upstream_root.is_dir():
            raise IntegrationConfigError(f"pinned upstream root is missing: {self.upstream_root}")
        required = (
            self.cluster_config,
            self.full_trace,
            self.tiny_trace,
            "profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/meta.yaml",
            "LICENSE",
        )
        missing = [item for item in required if not (self.upstream_root / item).is_file()]
        if missing:
            raise IntegrationConfigError(f"pinned upstream artifacts are missing: {missing}")

    def trace_path(self, *, tiny: bool) -> Path:
        return self.upstream_root / (self.tiny_trace if tiny else self.full_trace)

    def split(self, *, tiny: bool) -> RequestSplit:
        return self.tiny_split if tiny else self.full_split

    def resolve_budget(self) -> ResolvedBudget:
        """Mirror the pinned simulator's deterministic Llama memory formulas."""
        cluster = json.loads((self.upstream_root / self.cluster_config).read_text())
        nodes = cluster.get("nodes", [])
        if len(nodes) != 1 or len(nodes[0].get("instances", [])) != 1:
            raise IntegrationConfigError("primary topology must be one node and one instance")
        node = nodes[0]
        instance = node["instances"][0]
        if instance.get("model_name") != self.model or instance.get("hardware") != self.hardware:
            raise IntegrationConfigError("cluster model/hardware does not match the frozen config")
        model_path = self.upstream_root / "configs/model" / f"{self.model}.json"
        model = json.loads(model_path.read_text())
        total_gpu_bytes = _strict_int(instance["npu_mem"], "mem_size") * 1024**3
        host_capacity_bytes = _strict_int(node["cpu_mem"], "mem_size") * 1024**3
        tp = _strict_int(instance, "tp_size")
        num_npus = _strict_int(instance, "num_npus")
        weight_bytes = _llama_weight_bytes(model, bytes_per_element=2, tp=tp)
        head_dim = model.get("head_dim", model["hidden_size"] // model["num_attention_heads"])
        kv_dim = model.get("num_key_value_heads", model["num_attention_heads"]) * head_dim
        kv_bytes_per_token = (
            2 * kv_dim * model["num_hidden_layers"] * 2 // num_npus
        )
        block_bytes = kv_bytes_per_token * self.block_size_tokens
        post_weight_blocks = (total_gpu_bytes - weight_bytes) // block_bytes
        remaining_blocks = post_weight_blocks - self.active_request_reservation_blocks
        if remaining_blocks <= 0:
            raise IntegrationConfigError("active-request reservation exhausts KV capacity")
        reusable_gpu_blocks = int(remaining_blocks * self.reusable_gpu_fraction)
        return ResolvedBudget(
            total_gpu_bytes=total_gpu_bytes,
            weight_bytes=weight_bytes,
            kv_bytes_per_token=kv_bytes_per_token,
            block_size_tokens=self.block_size_tokens,
            block_bytes=block_bytes,
            post_weight_blocks=post_weight_blocks,
            active_request_reservation_blocks=self.active_request_reservation_blocks,
            remaining_blocks=remaining_blocks,
            reusable_gpu_blocks=reusable_gpu_blocks,
            host_capacity_bytes=host_capacity_bytes,
            host_capacity_blocks=host_capacity_bytes // block_bytes,
        )


def _llama_weight_bytes(model: dict[str, Any], *, bytes_per_element: int, tp: int) -> int:
    hidden = int(model["hidden_size"])
    vocab = int(model["vocab_size"])
    layers = int(model["num_hidden_layers"])
    heads = int(model["num_attention_heads"])
    head_dim = int(model.get("head_dim", hidden // heads))
    kv_heads = int(model.get("num_key_value_heads", heads))
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    ffn = int(model["intermediate_size"])
    embedding = (vocab // tp) * hidden * bytes_per_element
    norm = hidden * bytes_per_element
    qkv = hidden * ((q_dim + 2 * kv_dim) // tp) * bytes_per_element
    output = (q_dim // tp) * hidden * bytes_per_element
    gate_up = hidden * 2 * (ffn // tp) * bytes_per_element
    down = (ffn // tp) * hidden * bytes_per_element
    block = norm + qkv + output + norm + gate_up + down
    lm_head = hidden * (vocab // tp) * bytes_per_element
    return embedding + layers * block + norm + lm_head


def _strict_mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise IntegrationConfigError(f"{name} must be an object")
    return value


def _strict_str(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise IntegrationConfigError(f"{name} must be a nonempty string")
    return value


def _strict_int(raw: dict[str, Any], name: str, *, allow_zero: bool = False) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrationConfigError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise IntegrationConfigError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def _strict_float(
    raw: dict[str, Any], name: str, *, allow_zero: bool = False
) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegrationConfigError(f"{name} must be numeric")
    result = float(value)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        raise IntegrationConfigError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return result
