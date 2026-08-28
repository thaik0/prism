# Prism

**Stack:** Python, C++17, pybind11, AWS, Docker

Prism is an experimental storage-tiering system that studies whether learned access patterns can help decide which data should stay in fast memory and which data can remain on slower storage.

The original idea was predictive: learn upcoming demand changes well enough to move data into memory before a burst of accesses arrives. The experiments produced a more nuanced result. Forecasts became more accurate, but those improvements did not reliably translate into better placement decisions after accounting for limited memory and data-movement cost.

The final system uses learned demand estimates for stable, cost-aware storage tiering rather than claiming successful predictive prefetching.

## Architecture
```
       Access history
             |
             v
    Learn demand structure
             |
             v
      Estimate demand
             |
             v
     Project to records
             |
             v
   Cost-aware controller <---- sizes, capacity,
             |                 residency, move cost
             v
        RAM / disk
```
ML estimates demand by learning groups of records that become active together. The controller choose placement, combining the estimates with record sizes, residency, and factors like capacity and cost. ML never directly moves data.

## What I found
I evaluated Prism on controlled synthetic workloads where the underlying demand structure was known but hidden from the model. The learned representation recovered the planted overlapping working sets with approximately: 0.950 mean membership cosine similarity, 0.785 support recall, and 0.250 analytic-chance support recall.

But better forecasts did not reliably create better storage descriptions. The main conclusion was that prediction quality and decision quality are not the same thing. A forecast can only help if it changes which records should be stored in memory and the expected benefit exceeds the cost of moving them.

## Why didn't prediction help?
Prism predicts demand at the level of learned working sets and then projects that demand back to individual records.

Several factors can prevent a better forecast from changing the final placement:

- a predicted-hot record may already be in memory
- stable background demand may dominate the short-term forecast
- a large record may displace several other useful records
- multiple records may compete for limited capacity
- the expected access savings may be smaller than the migration cost.

## Native storage engine
To test placement against an actual storage implementation rather than only a Python model, I built a C++17 two-tier store with an in-memory tier (byte constrained), immutable records, CRC-verified reads, explicit promotion & eviction, atomic placement, capacity enforcement, and replay.

The C++ engine contains storage mechanics, not learned policy logic. Python computes the placement decisions and communicates with the engine through pybind11.

## Reproducible cloud experiments
I containerized Prism's experimental pipeline so the same frozen workloads could run locally or through AWS Batch.

The cloud workflow uses Docker and Terraform-backed AWS infrastructure for batch execution and artifact handling. Experiment configurations, random seeds, and inputs remain fixed across environments, so moving an experiment to AWS changes where the computation runs rather than changing the scientific workload.

This provided a reproducible Linux execution environment for larger experiment batches while keeping local and cloud runs consistent.

## LLM-serving experiment
As an additional application test, I integrated Prism with a pinned LLM-serving simulator and used Prism's decisions to control simulated KV-cache placement. The simulator remained responsible for scheduling, prefix matching, recomputation, transfers, and timing. Prism only supplied placement targets.

In the accepted experiment, predictive Prism, frozen Prism, static placement, LFU, and native LRU produced identical time-to-first-token and recomputation results. Even an oracle policy did not improve latency. It transferred an additional 56 MiB and increased mean time-to-first-token by approximately 1.401 ms. This further supported that knowing future demand more accurately is useful only when the resulting movement is worth its transfer cost.

## Run Prism
Reqs: Python 3.11+, C++17 compiler, CMake, zlib

Install and run Python tests:
```
python3 -m pip install -e .
python3 -m pytest -q
```

Run a representative controlled experiment:
```
python3 -m prism.experiments.cli \
  --manifest configs/milestone5_experiments.json \
  --experiment-id baseline__seed_1729 \
  --output-dir /tmp/prism_run
```

Build the native engine:
```
cmake -S cpp -B build/cpp-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPRISM_BUILD_TESTS=ON

cmake --build build/cpp-debug --parallel
ctest --test-dir build/cpp-debug --output-on-failure
```

Run the parity fixture:
```
python3 -m prism.native.cli \
  --fixture \
  --output-dir /tmp/prism_native_fixture
```

## Limitations
- main workloads are synthetic
- learned forecasts did not demonstrate reliable proactive placement gains
- native engine has no concurrency
- records are immutable after creation
- LLM-serving experiment uses a simulator rather than a real GPU
- no latency advantage




