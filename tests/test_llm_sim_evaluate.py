from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from prism.llm_sim.catalog import build_block_catalog
from prism.llm_sim.config import IntegrationConfig
from prism.llm_sim.demand import build_logical_demand
from prism.llm_sim.evaluate import (
    _target_comparison,
    canonical_tree_hashes,
    write_static_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = IntegrationConfig.from_json(
    PROJECT_ROOT / "configs/milestone8_llmservingsim.json"
)


def test_canonical_static_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    budget = CONFIG.resolve_budget()
    catalog = build_block_catalog(
        CONFIG.trace_path(tiny=True),
        namespace=f"{CONFIG.model}|{CONFIG.dtype}|b{CONFIG.block_size_tokens}",
        block_size_tokens=CONFIG.block_size_tokens,
        block_bytes=budget.block_bytes,
    )
    demand = build_logical_demand(catalog, CONFIG.tiny_split)
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        write_static_artifacts(
            root,
            CONFIG,
            budget,
            catalog,
            demand,
            tiny=True,
            patch_sha256="0" * 64,
            source_hashes={"fixture": "1" * 64},
        )
    assert canonical_tree_hashes(roots[0]) == canonical_tree_hashes(roots[1])
    manifest = json.loads((roots[0] / "integration_manifest.json").read_text())
    assert manifest["upstream"]["commit"] == CONFIG.upstream_commit
    assert manifest["upstream"]["submodules"]["astra-sim"] == CONFIG.astra_sim_commit
    assert manifest["dependencies"]["protobuf_runtime"] == "7.35.1"


def test_target_diagnostics_are_request_aligned_and_deterministic() -> None:
    left = [
        {"request_id": 4, "target_block_ids": ["a", "b"]},
        {"request_id": 5, "target_block_ids": []},
    ]
    right = [
        {"request_id": 4, "target_block_ids": ["b", "c"]},
        {"request_id": 5, "target_block_ids": []},
    ]
    assert _target_comparison(left, right) == {
        "request_count": 2,
        "disagreeing_request_count": 1,
        "disagreement_rate": 0.5,
        "mean_jaccard": (1 / 3 + 1) / 2,
    }


class _Scheduler:
    pd_type = None
    enable_prefix_caching = False
    model = "fixture"
    instance_id = 0

    def __init__(self):
        self.added = []

    def add_request(self, request, is_init=True):
        self.added.append((request, is_init))


def _load_router(source: str):
    source = source.replace(
        "from .logger import get_logger",
        "def get_logger(*args, **kwargs): return None",
    )
    namespace = {}
    exec(compile(source, "router.py", "exec"), namespace)
    return namespace["Router"]


def test_disabled_hook_preserves_native_router_behavior(tmp_path: Path) -> None:
    upstream = CONFIG.upstream_root
    original_source = (upstream / "serving/core/router.py").read_text(encoding="utf-8")
    patched_root = tmp_path / "patched"
    (patched_root / "serving/core").mkdir(parents=True)
    shutil.copy(upstream / "serving/core/router.py", patched_root / "serving/core/router.py")
    shutil.copy(upstream / "serving/__main__.py", patched_root / "serving/__main__.py")
    subprocess.run(
        [
            "patch", "--batch", "-p1", "-i",
            str(PROJECT_ROOT / "integration/llmservingsim/prism_hook.patch"),
        ],
        cwd=patched_root,
        check=True,
        capture_output=True,
        text=True,
    )
    patched_source = (patched_root / "serving/core/router.py").read_text(encoding="utf-8")
    original_scheduler = _Scheduler()
    patched_scheduler = _Scheduler()
    original = _load_router(original_source)(1, [original_scheduler], 1)
    patched = _load_router(patched_source)(1, [patched_scheduler], 1)
    request = {
        "index": 0,
        "input_toks": 16,
        "output_toks": 17,
        "arrival_time_ns": 5,
    }
    original._pending_requests = [dict(request)]
    patched._pending_requests = [dict(request)]

    assert original.route_arrived_requests(5) == patched.route_arrived_requests(5) == 1
    assert original_scheduler.added == patched_scheduler.added
    assert patched.prism_hook is None
