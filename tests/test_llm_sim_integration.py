from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from prism.llm_sim.config import IntegrationConfig
from prism.llm_sim.evaluate import canonical_tree_hashes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/milestone8_llmservingsim.json"
CONFIG = IntegrationConfig.from_json(CONFIG_PATH)


@pytest.mark.skipif(
    os.environ.get("PRISM_RUN_LLM_SIM_INTEGRATION") != "1",
    reason="set PRISM_RUN_LLM_SIM_INTEGRATION=1 for the pinned Docker smoke",
)
def test_tiny_trace_runs_all_six_policies_from_clean_state(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "prism.llm_sim.cli",
                "--config",
                str(CONFIG_PATH),
                "--output-dir",
                str(root),
                "--tiny",
                "--skip-container-build",
                "--skip-upstream-build",
                "--workers",
                "6",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    result = json.loads((roots[0] / "policy_results.json").read_text())
    assert tuple(result["policies"]) == tuple(sorted(CONFIG.policy_ids))
    assert {
        json.loads(
            (root / "runs" / policy / "hook_results.json").read_text()
        )["logical_demand_sha256"]
        for root in roots
        for policy in CONFIG.policy_ids
    } == {result["logical_demand_sha256"]}
    assert canonical_tree_hashes(roots[0]) == canonical_tree_hashes(roots[1])
