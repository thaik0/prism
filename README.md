# Prism

Prism is a predictive storage-tiering research system. The current implementation
includes **Milestone 1**, a controlled synthetic workload generator;
**Milestone 2**, a deterministic slow structural-recovery baseline; and
**Milestone 3**, a deterministic next-window activation and conditional-intensity
predictor. Milestone 3 fits structure on training windows only, freezes learned
memberships, compares fixed recent-state and context-plus-state linear models,
and evaluates calibration and three untouched-test scientific gates.

No record-demand forecast, cache, placement, storage-tier, controller, or latency
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

## Fit and evaluate the fast predictor

After generating the dedicated source trace and passing its prerequisite gates:

```bash
PYTHONPATH=src python3 -m prism.predictor.cli \
  --run-dir /tmp/prism_m3_predictor_source \
  --structure-config configs/milestone2_representative.json \
  --config configs/milestone3_predictor.json \
  --output-dir /tmp/prism_milestone3_run
```

The command writes exactly `predictor_config.json`, `predictor_bundle.npz`,
`predictions.npz`, and `evaluation_report.json`. Deployable arrays contain no
hidden targets or planted identities.

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
See [the Milestone 3 predictor documentation](docs/fast_predictor.md) for splits,
features, fixed models, metrics, gates, artifacts, leakage protections, and
limitations.
