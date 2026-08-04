# Prism

Prism is a predictive storage-tiering research system. The current implementation
includes **Milestone 1**, a controlled synthetic workload generator;
**Milestone 2**, a deterministic slow structural-recovery baseline; and
**Milestone 3**, a deterministic next-window activation and conditional-intensity
predictor; and **Milestone 4**, a deterministic byte-constrained placement
simulator. **Milestone 5** adds a frozen 36-run, 12-variant evaluation with two
static Prism controls, three mechanical forecast ablations, dynamic-action and
pre-transition diagnostics, paired comparisons, and deterministic aggregation.
**Milestone 5.5** adds a frozen 27-run diagnosis across three activation regimes
and cumulative horizons `1`, `2`, and `4`, tracing forecast movement through
record ranking, controller rejection, oracle agreement, and promotion repayment.
Milestone 4 consumes frozen predictor artifacts, calibrates
record-demand projection on training windows only, warms six independent policies
on validation, and evaluates access plus promotion cost on identical test events.

**Milestone 6** adds a standalone C++17 storage library and tools: a deterministic
FlatBuffers-indexed immutable store, validated file-backed slow reads, explicitly
owned byte-constrained fast-tier buffers, atomic exact-target placement,
structured errors/counters, full inspection, and deterministic JSONL replay.

Milestones 1--5.5 continue to use simulated costs. Milestone 6 provides real
filesystem reads and process-owned memory, but it collects no canonical
wall-clock timing and makes no RAM/SSD performance-superiority claim. Python/C++
integration and asynchronous migration remain unimplemented.

## Requirements

- Python 3.11 or newer
- NumPy
- SciPy
- scikit-learn
- pytest, for development tests
- CMake 3.24+, a C++17 compiler, and zlib, for Milestone 6

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

## Evaluate simulated placement policies

After producing the dedicated source and a gate-passing frozen predictor run:

```bash
PYTHONPATH=src python3 -m prism.simulation.cli \
  --run-dir /tmp/prism_m4_source \
  --predictor-run-dir /tmp/prism_m4_predictor \
  --config configs/milestone4_simulation.json \
  --output-dir /tmp/prism_milestone4_run
```

The command verifies source hashes and dimensions, fits only the projection
calibration, replays LRU, LFU, recent-demand greedy, predictive greedy, one-window
oracle greedy, and one-window oracle exact, then writes exactly
`simulation_config.json`, `projection_model.npz`, `policy_traces.npz`, and
`evaluation_report.json`.

## Run the frozen Milestone 5 evaluation

```bash
PYTHONPATH=src python3 -m prism.experiments.cli \
  --manifest configs/milestone5_experiments.json \
  --output-dir /tmp/prism_milestone5
```

The sequential harness runs all 12 variants at seeds `1729`, `2718`, and
`31415`, preserves each Milestone 1--4 stage directory, and writes a complete
index plus deterministic JSON and Markdown aggregates. Add `--experiment-id
context_weak__seed_31415` for one run or `--resume` to hash-validate and reuse
completed runs. A non-resume run rejects a nonempty destination.

The completed sweep found that Predictive Greedy was identical to its frozen
validation-final placement in 35 of 36 runs and identical in cost to the
Recent-State-Only ablation in 34 of 36 runs. This qualifies the earlier result:
the current implementation usually benefits from a good validation-developed
placement, but the sweep does not support continued dynamic value from the fast
predictor. See [the compact Milestone 5 results](docs/milestone5_results.md).

## Run the frozen Milestone 5.5 diagnosis

```bash
PYTHONPATH=src python3 -m prism.experiments.cli \
  --manifest configs/milestone55_actionability.json \
  --output-dir /tmp/prism_milestone55
```

The 27-run sweep completed reproducibly and the sparse regimes separated as
intended, but all four predeclared candidate cells remained identical to frozen
placement and Recent-State-Only. The precommitted conclusion is
`stable_cost_aware_tiering_reframe`: the evidence supports learned latent-demand
structure for stable cost-aware placement, not useful dynamic predictive tiering
from the current fast forecast. See [the diagnosis](docs/milestone55_actionability.md)
and [compact results](docs/milestone55_results.md).

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

## Build and exercise the native storage engine

```bash
cmake -S cpp -B build/cpp-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPRISM_BUILD_TESTS=ON \
  -DPRISM_WARNINGS_AS_ERRORS=ON
cmake --build build/cpp-debug --parallel
ctest --test-dir build/cpp-debug --output-on-failure

build/cpp-debug/prism_store_build \
  --manifest cpp/tests/fixtures/store_manifest.json \
  --output-dir /tmp/prism_store
build/cpp-debug/prism_store_inspect \
  --store-dir /tmp/prism_store --verify-all
build/cpp-debug/prism_store_replay \
  --store-dir /tmp/prism_store \
  --capacity-bytes 40 \
  --trace cpp/tests/fixtures/replay.jsonl \
  --output /tmp/prism_replay.json
```

The first configure fetches pinned dependency releases into the ignored build
tree. See [the native engine documentation](docs/native_storage_engine.md) for
the format, APIs, error/counter contracts, atomicity, determinism, sanitizer
commands, and limitations.

See [the Milestone 1 workload documentation](docs/workload_generator.md) for the
generation model, schemas, engineering and scientific validation rules, and
reproducibility contract. See [the Milestone 2 structural-recovery
documentation](docs/structure_recovery.md) for demand construction, the fixed NMF
baseline, recovery metrics, artifacts, limitations, and reproducibility.
See [the Milestone 3 predictor documentation](docs/fast_predictor.md) for splits,
features, fixed models, metrics, gates, artifacts, leakage protections, and
limitations. See [the Milestone 4 simulated-placement
documentation](docs/simulated_placement.md) for projection calibration, policy
timing, byte capacity, cost accounting, transition metrics, scientific gates,
artifacts, representative results, and limitations.
See [the Milestone 5 results](docs/milestone5_results.md) for the frozen design,
causal controls, full-sweep results, gate outcomes, reproducibility evidence, and
small-sample limitations.
See [the Milestone 5.5 actionability diagnosis](docs/milestone55_actionability.md)
for cumulative horizons, common windows, signal-flow diagnostics, and the
precommitted thesis gate, and [its results](docs/milestone55_results.md) for the
frozen negative outcome and stable cost-aware tiering reframe.
