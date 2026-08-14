from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prism.container.runner import (
    ContainerSpecError,
    _load_spec,
    main,
    run_container_experiment,
)
from prism.experiments.config import ExperimentManifest, VariantDefinition


def _write_spec(root: Path, **overrides: object) -> Path:
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "accepted_milestone5_native_parity",
        "experiment_manifest": "configs/manifest.json",
        "experiment_id": "baseline__seed_1729",
    }
    value.update(overrides)
    path = root / "spec.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_spec_is_strict_and_rejects_input_escape(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path)
    value, raw = _load_spec(spec)
    assert value["schema_version"] == 1
    assert hashlib.sha256(raw).hexdigest()

    escaped = _write_spec(tmp_path, experiment_manifest="../manifest.json")
    with pytest.raises(ContainerSpecError, match="contained relative"):
        _load_spec(escaped)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1,"kind":"x",'
        '"experiment_manifest":"x","experiment_id":"x"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ContainerSpecError, match="duplicate JSON key"):
        _load_spec(duplicate)


def test_runner_orchestrates_existing_pipeline_and_writes_stable_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    configs = input_root / "configs"
    configs.mkdir(parents=True)
    spec = _write_spec(input_root)
    paths = {
        name: configs / filename
        for name, filename in (
            ("manifest", "manifest.json"),
            ("workload", "workload.json"),
            ("structure", "structure.json"),
            ("predictor", "predictor.json"),
        )
    }
    for name, path in paths.items():
        path.write_text(f'{{"fixture":"{name}"}}\n', encoding="utf-8")
    manifest = ExperimentManifest(
        path=paths["manifest"],
        sha256="a" * 64,
        raw={},
        base_workload_config=paths["workload"],
        structure_config=paths["structure"],
        predictor_config=paths["predictor"],
        seeds=(1729,),
        variants=(VariantDefinition("baseline", "baseline", {}, 0.25, 2.0),),
        policies=(),
    )
    monkeypatch.setattr("prism.container.runner.load_manifest", lambda _: manifest)

    def fake_experiments(manifest_path, output, *, experiment_id):
        run = Path(output) / "runs" / experiment_id
        for child in ("workload", "predictor"):
            (run / child).mkdir(parents=True, exist_ok=True)
        (run / "resolved_simulation_config.json").write_text("{}\n")
        (run / "run_status.json").write_text('{"status":"completed"}\n')
        return SimpleNamespace(failed_count=0)

    def fake_native(run_dir, predictor_dir, simulation_config, output):
        destination = Path(output)
        destination.mkdir()
        (destination / "native.txt").write_text("native\n")
        return SimpleNamespace(
            parity=SimpleNamespace(
                report={"overall_gates": {"overall_parity_passed": True}}
            )
        )

    monkeypatch.setattr("prism.container.runner.run_experiments", fake_experiments)
    monkeypatch.setattr(
        "prism.container.runner.run_representative_parity", fake_native
    )
    monkeypatch.setenv("PRISM_GIT_REVISION", "0123456789abcdef")

    outputs = [tmp_path / "first", tmp_path / "second"]
    results = [
        run_container_experiment(
            spec,
            output,
            input_root=input_root,
            output_root=tmp_path,
        )
        for output in outputs
    ]
    assert results[0].experiment_id == "baseline__seed_1729"
    first = (outputs[0] / "run_manifest.json").read_bytes()
    assert first == (outputs[1] / "run_manifest.json").read_bytes()
    manifest_value = json.loads(first)
    assert manifest_value["native_build"]["parity_passed"] is True
    assert manifest_value["prism"]["git_revision"] == "0123456789abcdef"
    assert str(tmp_path).encode() not in first


def test_runner_rejects_output_escape_and_cli_reports_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    spec = _write_spec(input_root)
    output_root = tmp_path / "allowed"
    output_root.mkdir()
    with pytest.raises(ContainerSpecError, match="escapes"):
        run_container_experiment(
            spec,
            tmp_path / "outside",
            input_root=input_root,
            output_root=output_root,
        )

    monkeypatch.delenv("PRISM_INPUT_ROOT", raising=False)
    monkeypatch.delenv("PRISM_OUTPUT_ROOT", raising=False)
    assert main(["--spec", str(tmp_path / "missing.json"), "--output-dir", str(tmp_path / "out")]) == 2
