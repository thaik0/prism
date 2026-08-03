# Prism

Prism is a predictive storage-tiering research system. The current implementation
includes **Milestone 1**, a controlled synthetic workload generator, and
**Milestone 2**, a deterministic slow structural-recovery baseline. Milestone 2
constructs raw demand windows, fits one nonnegative matrix factorization, and
measures recovery of planted fuzzy working sets. The narrow Milestone 3
prerequisite also plants and validates stochastic context-informed burst
intensity; the predictor itself is not implemented.

No future activation prediction, cache, placement, storage-tier, or latency
behavior is implemented yet.

## Requirements

- Python 3.11 or newer
- NumPy
- SciPy
- scikit-learn
- pytest, for development tests

The Milestone 1 generator itself remains standard-library-only.

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

## Recover slow working-set structure

After generating and validating a source trace:

```bash
PYTHONPATH=src python3 -m prism.structure.cli \
  --run-dir /tmp/prism_milestone1_run \
  --config configs/milestone2_representative.json \
  --output-dir /tmp/prism_milestone2_run
```

The learner counts raw accesses in every configured window and record, fits one
fixed deterministic scikit-learn NMF with supplied factor count `K`, normalizes
learned fuzzy memberships, and evaluates them against hidden truth only after
fitting. It writes exactly `learner_config.json`, `demand_matrix.npz`,
`learned_structure.npz`, and `recovery_report.json`.

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
without a clear precursor. For the longer predictor-prerequisite trace, generate
`configs/milestone3_predictor_workload.json` and add
`--require-intensity-signal` to require useful but imperfect conditional-intensity
variation.

## Test

```bash
python3 -m pytest -q
```

See [the Milestone 1 workload documentation](docs/workload_generator.md) for the
generation model, schemas, engineering and scientific validation rules, and
reproducibility contract. See [the Milestone 2 structural-recovery
documentation](docs/structure_recovery.md) for demand construction, the fixed NMF
baseline, recovery metrics, artifacts, limitations, and reproducibility.
