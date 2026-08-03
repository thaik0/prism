# Prism

Prism is a predictive storage-tiering research system. The current implementation
is limited to **Milestone 1**, a controlled synthetic workload generator. It
creates deterministic observable record-access traces and structurally separate
simulator-only ground truth for later research milestones.

No learning, prediction, cache, placement, storage-tier, or latency behavior is
implemented yet.

## Requirements

- Python 3.11 or newer
- pytest, for development tests

The generator has no third-party runtime dependencies.

## Generate the representative workload

From the repository root:

```bash
PYTHONPATH=src python3 -m prism.workload.cli \
  --config configs/milestone1_representative.json \
  --output-dir /tmp/prism_milestone1_run
```

The destination must not already contain files. The command writes exactly:

```text
config.json
observable_events.jsonl
hidden_ground_truth.json
summary.json
```

`observable_events.jsonl` is the model-plausible access history. Configuration is
experiment metadata, while `hidden_ground_truth.json` is strictly simulator-only
and must never be used as predictor input.

## Python API

```python
from prism.workload import WorkloadConfig, generate_workload, persist_workload

config = WorkloadConfig.from_json("configs/milestone1_representative.json")
result = generate_workload(config)
persist_workload(result, "/tmp/prism_milestone1_run")
```

`result.observable_events`, `result.hidden_ground_truth`, and `result.summary` are
separate in-memory structures.

## Validate scientific workload properties

Validation analyzes an existing run without regenerating it or modifying its four
source artifacts:

```bash
PYTHONPATH=src python3 -m prism.workload.validate \
  --run-dir /tmp/prism_milestone1_run \
  --require-demonstrations
```

This deterministically creates or replaces the separate derived artifact
`workload_validation.json`. The required-demonstrations flag checks for a clear
precursor followed by a burst, a clear precursor followed by no burst, and a burst
without a clear precursor.

## Test

```bash
python3 -m pytest -q
```

See [the Milestone 1 workload documentation](docs/workload_generator.md) for the
generation model, schemas, engineering and scientific validation rules, and
reproducibility contract.
