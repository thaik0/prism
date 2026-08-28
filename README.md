# Prism

Prism is a research prototype for cost-aware storage tiering. It learns
recurring access patterns, estimates near-future record demand, and passes those
estimates to a deterministic placement controller.

The project began with a predictive question: can data move into fast memory
before a demand burst arrives? The experiments found that better forecasts did
not reliably improve placement after accounting for memory limits and movement
cost. Prism therefore demonstrates stable, learned placement—not successful
predictive prefetching.

## Pipeline

```text
access history
    ↓
latent demand structure
    ↓
demand estimates
    ↓
record-level projection
    ↓
deterministic controller  ←  capacity, residency, movement cost
    ↓
RAM / file-backed storage
```

Machine learning estimates demand. The controller alone decides what moves.

## What the experiments showed

- The learned representation recovered overlapping working sets with `0.950`
  mean cosine similarity and `0.785` support recall (`0.25` analytic chance).
- In a frozen 36-run study, predictive placement matched the final stable
  placement in 35 runs. Better predictions rarely changed the chosen records.
- The Python model and C++17 storage engine matched across `130,349` operations
  with no state mismatches or capacity violations.
- In a pinned LLM-serving simulation, predictive and conventional policies had
  identical latency and recomputation results. Even the oracle was slightly
  slower after transfer cost.

The main takeaway is simple: prediction quality and decision quality are not
the same thing.

## Quick start

Prism supports Python 3.11+ on macOS and Linux. Building the native extension
also requires CMake 3.24+, a C++17 compiler, and zlib.

```bash
python3 -m pip install -e .
python3 -m pytest -q
```

Run one representative experiment (the output path must be absent or empty):

```bash
python3 -m prism.experiments.cli \
  --manifest configs/milestone5_experiments.json \
  --experiment-id baseline__seed_1729 \
  --output-dir /tmp/prism_run
```

## Repository guide

- `src/prism/workload`, `structure`, and `predictor` build the learned demand
  pipeline.
- `src/prism/simulation` and `experiments` evaluate placement on frozen traces.
- `cpp` and `src/prism/native` contain the two-tier store and parity checks.
- `src/prism/llm_sim` contains the pinned LLM-serving simulator integration.
- `src/prism/container`, `src/prism/cloud`, and `infra` provide reproducible
  local and AWS batch execution.

## Scope

Prism is research software, not a production storage service. Its conclusions
come from controlled traces and a pinned simulator, not real heterogeneous
hardware. Synthetic hidden truth is used only for evaluation, and compared
policies receive identical seeded traces.
