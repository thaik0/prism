# Prism

Prism is an experimental storage-tiering system that explores whether learned
demand structure can help decide what data should remain in expensive fast
memory. It learns fuzzy latent working sets, estimates demand, projects that
demand to records or prefix blocks, and passes the result to a deterministic
cost-aware placement controller. The project spans controlled simulation, a
native C++ RAM/file-backed store, exact Python/C++ execution parity, a pinned
LLM-serving simulator integration, and a reproducible Linux/ARM64 batch path on
AWS.

## Question and conclusion

Prism began with a stronger question: could context-informed forecasts move data
before abrupt demand changes and outperform reactive caching after movement
cost? Forecast metrics improved, but that improvement did not reliably produce
useful placement actions. Stable residual demand, record projection, migration
economics, and capacity competition dominated the controller.

The final thesis is:

> **Prism learns latent-demand structure and uses it for stable, cost-aware
> storage tiering.**

Dynamic predictive actionability was not demonstrated. Milestone 9, the planned
real heterogeneous deployment, was intentionally canceled at project closeout.

## Architecture

```mermaid
flowchart LR
    A["Access history"] --> B["Latent structure"]
    B --> C["Demand estimation"]
    C --> D["Record / block projection"]
    D --> E["Deterministic cost-aware controller"]
    E --> F["Simulated or native tier execution"]
    G["Sizes, capacity, residency,<br/>and movement cost"] --> E

    subgraph ML["ML estimates demand"]
        B
        C
        D
    end

    subgraph SYS["Systems logic chooses placement"]
        E
        F
    end
```

Synthetic hidden truth is separate from model-visible events. Compared policies
receive identical frozen traces. ML never chooses storage actions directly.

## What the experiments found

- The representative NMF recovered fuzzy working-set structure with mean cosine
  similarity `0.949876` and support recall `0.785291` versus `0.25` analytic
  chance.
- On the dedicated controlled trace, activation Brier score improved from
  `0.172441` to `0.159576`, and conditional-intensity RMSE improved from
  `0.378863` to `0.365930`.
- In the frozen 36-run placement study, Predictive Greedy (Prism) matched
  Validation-Final Frozen (Prism) in 35 runs and Recent-State-Only (Prism
  ablation) in cost in 34 runs.
- All four precommitted sparse, multi-window Milestone 5.5 candidate cells made
  zero test target changes and failed every actionability gate.
- The native engine passed deterministic correctness, corruption, atomicity,
  and sanitizer tests. Four Python policy paths matched 130,349 native
  operations with zero mismatches or capacity violations.
- In the pinned LLMServingSim run, static, frozen, predictive, LFU, and native
  LRU had identical TTFT and recomputation. Oracle added 56 MiB of transfer and
  was `1.401 ms` slower in mean TTFT.

These are controlled and simulator-specific results, not production or
real-hardware performance claims.

## Implementation highlights

- seeded contextual workload generator with leakage-safe hidden truth;
- deterministic NMF structure recovery and fixed linear forecast baselines;
- training-only record-demand projection and deterministic byte-constrained
  placement;
- frozen multi-seed controls, ablations, oracle diagnostics, and actionability
  gates;
- C++17 immutable two-tier store with CRC-verified reads and atomic exact-target
  transitions;
- private pybind11 boundary with an independent Python semantic-parity ledger;
- narrow LLMServingSim hook that leaves scheduling, active KV, transfers,
  recomputation, and timing simulator-owned;
- digest-pinned Linux/ARM64 batch execution with deterministic manifests and
  local/cloud parity verification;
- AWS Batch on Fargate with ECR, S3, and CloudWatch Logs, managed through
  Terraform and GitHub Actions OIDC rather than long-lived AWS keys.

## How to run Prism

Supported development environments are POSIX macOS and Linux with Python 3.11+
and CMake 3.24+. A C++17 compiler and zlib are required for the native extension.
From the repository root, install Prism and run one accepted end-to-end
experiment. The output directory must be absent or empty.

```bash
python3 -m pip install -e .
python3 -m prism.experiments.cli \
  --manifest configs/milestone5_experiments.json \
  --experiment-id baseline__seed_1729 \
  --output-dir /tmp/prism_run
```

For the bounded Linux/ARM64 batch path, build the image and run the committed
single-experiment spec:

```bash
docker build --no-cache \
  --build-arg PRISM_GIT_REVISION="$(git rev-parse HEAD)" \
  -t prism:phase1 .

mkdir -p /tmp/prism_batch_output
chmod 0777 /tmp/prism_batch_output
docker run --rm \
  --mount type=bind,src="$PWD",dst=/input,readonly \
  --mount type=bind,src=/tmp/prism_batch_output,dst=/output \
  prism:phase1 \
  prism-container-run \
  --spec /input/container/phase1-experiment.json \
  --output-dir /output
```

After deploying the documented AWS foundation and setting the required
`PRISM_CLOUD_*` environment variables, submit the same accepted workload through
the narrow AWS Batch adapter:

```bash
prism-cloud submit --spec container/phase1-experiment.json
prism-cloud status RUN_ID
prism-cloud wait RUN_ID
prism-cloud logs RUN_ID
prism-cloud download RUN_ID --output-dir /tmp/prism-cloud-output
```

Run `python3 -m pytest -q` for the Python suite.

## Repository map

| Path | Purpose |
|---|---|
| `src/prism/workload` | controlled trace generation and scientific validation |
| `src/prism/structure` | demand matrices and latent working-set recovery |
| `src/prism/predictor` | leakage-safe activation and intensity prediction |
| `src/prism/simulation` | projection, policies, controller, and cost replay |
| `src/prism/experiments` | frozen Milestone 5 and 5.5 orchestration |
| `src/prism/native` | deterministic payloads and Python/C++ parity |
| `src/prism/container` | Linux batch runner, manifest, and parity verification |
| `src/prism/cloud` | deterministic S3 bundle, AWS Batch CLI/bootstrap, and verified download |
| `src/prism/llm_sim` | pinned LLMServingSim policy integration |
| `cpp` | native immutable two-tier storage engine and tools |
| `container` | accepted batch spec and pinned Python constraints |
| `infra/terraform` | AWS state bootstrap and main Batch/Fargate infrastructure |
| `.github/workflows` | cloud CI, image publication, Terraform promotion, and smoke verification |
| `configs` | committed experiment and controller configurations |
| `integration/llmservingsim` | isolated upstream hook and runtime definition |
| `tests` | deterministic, failure-path, leakage, and integration tests |

## Project status

Prism's scientific core is complete at `v1.0.0`: Milestones 1--8 are closed and
Milestone 9 is canceled. Cloud Phases 1 and 2 package and execute the accepted
workload without changing its scientific claims; Cloud Phase 3 Terraform/OIDC
acceptance is in progress. The repository is not production-ready: it has no
asynchronous migration, concurrent native request path, mutable data, or
demonstrated heterogeneous-hardware advantage.
