# Prism

Prism is an experimental storage-tiering system that explores whether learned
demand structure can help decide what data should remain in expensive fast
memory. It learns fuzzy latent working sets, estimates demand, projects that
demand to records or prefix blocks, and passes the result to a deterministic
cost-aware placement controller. The project spans controlled simulation, a
native C++ RAM/file-backed store, exact Python/C++ execution parity, a pinned
LLM-serving simulator integration, and a reproducible Linux/ARM64 batch path on
AWS.
**Stack:** Python, C++17, pybind11, NumPy, scikit-learn, AWS

Prism is an ongoing, experimental storage-tiering system that studies whether learned access patterns can help decide which data should stay in fast memory and which data can remain on slower storage.

The idea was predictive: learn demand patterns well enough to move data into memory before a burst of accesses arrives. The experiments produced a more nuanced result. Forecasts became more accurate, but those improvements did not reliably translate into better placement decisions after accounting for limited memory and data-movement cost.

The final system therefore uses learned demand estimates for stable, cost-aware placement rather than claiming successful predictive prefetching.

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
ML estimates demand, while the controller chooses placement. ML never will directly move the data.

## What I found

I evaluated Prism on controlled synthetic workloads where the hidden demand structure was known.

The learned representation recovered the underlying overlapping working sets with 0.950 mean cosine similarity and 0.785 support recall, compared with 0.25 analytic-chance recall.

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
But better forecasts did not reliably create better storage actions. In a frozen 36-run placement experiment, the predictive policy matched the final stable Prism policy in 35 of 36 runs. A simpler recent-state-only ablation also matched its cost in 34 runs. I then tested additional predictive settings designed to make future demand shifts more actionable. None produced placement changes that passed the pre-specified evaluation criteria.

My main conclusion is that prediction quality and decision quality are not the same thing.

## Native storage engine
To test the placement logic against an actual storage implementation rather than only a Python simulator, I built a C++17 two-tier store with a bounded in-memory tier, file-backed storage, CRC-verified reads, deterministic placement transitions, and capacity enforcement.

I also built a parity test that runs the same storage operations through the Python model and native engine. Across 130,349 operations from four policy paths, the implementations produced zero state mismatches and zero capacity violations. 

## LLM-serving experiment
As an additional application test, I integrated Prism with a pinned LLM-serving simulator and used its placement decisions for simulated KV-cache storage.

In that experiment, static placement, frozen learned placement, predictive placement, LFU, and native LRU produced identical time-to-first-token and recomputation results.

Even an oracle policy did not improve latency: it transferred an additional 56 MiB and increased mean time-to-first-token by 1.401 ms.

That result reinforced the earlier finding: knowing future demand more accurately is useful only when the resulting placement change is worth its movement cost.

