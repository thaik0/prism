# Milestone 8 Results

## Outcome

Milestone 8 is complete for the frozen LLMServingSim configuration. Two
independent 300-request, six-policy runs produced byte-identical canonical
artifacts. The result is a null/negative transfer result: Training-Popularity
Static (Prism), Validation-Final Frozen (Prism), Predictive Greedy (Prism), LFU,
and native LRU had identical test TTFT and recomputation. Oracle Greedy caused
56 MiB of host-to-GPU traffic without reducing recomputation and was 1.401 ms
slower in mean TTFT.

This does not reverse the Milestone 5.5 conclusion. The experiment supports no
claim that continued fast prediction is actionable for this trace and cost
configuration.

## Upstream and integration boundary

- Repository: `https://github.com/casys-kaist/LLMServingSim.git`
- Commit: `2c2042ce4bf1b0283ebeed1db95db6f25e3e7511`
- Description: `v1.1.0-53-g2c2042c`; nearest release: `v1.1.0`
- License: MIT
- ASTRA-Sim: `f82fb3d861614a2a61febfaf87a1360db05efd81`
- Container base: `astrasim/tutorial-micro2024@sha256:567a0f020f18de9b7391d620041d2fd10120d6e0f1485b5110e0db79c2849305`

The recursive Chakra, fmt, spdlog, memory-backend, network-backend,
GoogleTest, yaml-cpp, CSG-HTSim, and ns-3 SHAs are recorded in every
`integration_manifest.json`. The runtime uses Python 3.10.12 and the exact
dependency versions in that manifest, including protobuf 7.35.1 required by
the generated Chakra sources.

The isolated patch changes only `serving/core/router.py` and
`serving/__main__.py`. It adds an optional hook argument and request/lifecycle,
batch, completion, and finalization callbacks. No scheduler, memory model,
batching algorithm, routing algorithm, ASTRA-Sim, profile, or model-execution
source is changed. With the hook absent, the native route remains unchanged;
both a focused behavioral test and an unmodified-upstream smoke run passed.

Prism supplies only prefix-closed targets for eligible, lock-free reusable GPU
prefix pages. LLMServingSim remains authoritative for arrivals, active-request
KV, prefix matching, native LRU, transfers, recomputation, scheduling, and
latency. The Milestone 7 native store is not used.

## Frozen experiment

The run used Llama 3.1 8B, bfloat16 weights, automatic KV dtype, 16-token KV
blocks, the RTXPRO6000 profile, one node and one serving instance, and
`sharegpt-llama-3.1-8b-300-sps10.jsonl`. Requests were split chronologically as
training `[0, 180)`, validation `[180, 240)`, and test `[240, 300)`.

The budget was resolved before policy results were inspected:

| Quantity | Resolved value |
|---|---:|
| Total modeled GPU memory | 103,079,215,104 bytes |
| Model weights | 16,060,522,496 bytes |
| KV bytes per token | 131,072 bytes |
| Bytes per 16-token block | 2,097,152 bytes |
| Post-weight KV capacity | 41,493 blocks |
| Active-request reservation | 2,048 blocks |
| Remaining capacity | 39,445 blocks |
| Prism reusable target budget (25%) | 9,861 blocks |
| Host prefix capacity | 262,144 blocks / 549,755,813,888 bytes |

The complete metadata-only catalog has 14,722 stable native prefix blocks and
15,942 logical references. Training saw 8,764 blocks; 5,958 were post-training
cold-start blocks. Test contained 3,060 logical block references: 177 (5.78%)
to training-seen blocks and 2,883 (94.22%) to cold-start blocks. All six runs
reported the same logical-demand SHA-256:

`ab5a5f6f367de74c7f7e1e743af6bddbd73e349eeddc0137586803fb8a100ba2`

## Test results

All latency values are simulator-native. TTFT is the primary metric.

| Policy | TTFT mean / median / p95 / p99 (ms) | E2E mean / median / p95 / p99 (ms) | TPOT mean / p95 (ms) |
|---|---:|---:|---:|
| LLMServingSim LRU | 16225.878 / 16760.961 / 19878.147 / 20294.223 | 32900.938 / 32586.183 / 35253.869 / 36462.978 | 26.068 / 32.391 |
| LFU | 16225.878 / 16760.961 / 19878.147 / 20294.223 | 32900.938 / 32586.183 / 35253.869 / 36462.978 | 26.068 / 32.391 |
| Training-Popularity Static (Prism) | 16225.878 / 16760.961 / 19878.147 / 20294.223 | 32900.938 / 32586.183 / 35253.869 / 36462.978 | 26.068 / 32.391 |
| Validation-Final Frozen (Prism) | 16225.878 / 16760.961 / 19878.147 / 20294.223 | 32900.938 / 32586.183 / 35253.869 / 36462.978 | 26.068 / 32.391 |
| Predictive Greedy (Prism) | 16225.878 / 16760.961 / 19878.147 / 20294.223 | 32900.938 / 32586.183 / 35253.869 / 36462.978 | 26.068 / 32.391 |
| Oracle Greedy | 16227.279 / 16762.312 / 19879.728 / 20295.804 | 32902.519 / 32587.764 / 35255.450 / 36464.559 | 26.068 / 32.391 |

The first five policies each achieved 1.389666 requests/s, 1,143.347 prompt
tokens/s, and 902.819 output tokens/s. Oracle achieved 1.389615 requests/s,
1,143.305 prompt tokens/s, and 902.786 output tokens/s.

| Policy | GPU hit blocks / tokens | Host hit blocks / tokens | Recomputed blocks / tokens | Host→GPU bytes | Adapter evictions, all splits | Maximum reusable GPU blocks | Target coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLMServingSim LRU | 228 / 3,648 | 0 / 0 | 2,832 / 45,312 | 0 | 0 | 14,722 | n/a |
| LFU | 228 / 3,648 | 0 / 0 | 2,832 / 45,312 | 0 | 5,111 | 9,611 | 7.45% |
| Training-Popularity Static (Prism) | 228 / 3,648 | 0 / 0 | 2,832 / 45,312 | 0 | 8,282 | 6,440 | 0.92% |
| Validation-Final Frozen (Prism) | 228 / 3,648 | 0 / 0 | 2,832 / 45,312 | 0 | 7,603 | 7,119 | 0.92% |
| Predictive Greedy (Prism) | 228 / 3,648 | 0 / 0 | 2,832 / 45,312 | 0 | 5,958 | 8,764 | 5.78% |
| Oracle Greedy | 200 / 3,200 | 28 / 448 | 2,832 / 45,312 | 58,720,256 | 14,915 | 275 | 6.54% |

The 56 MiB Oracle transfer estimate is 213,623 ns at the configured 256 GiB/s
host bandwidth; request latency itself includes native ASTRA-Sim execution.
Oracle promoted 193 blocks over all splits, but none of its test host hits were
useful target promotions under the recorded diagnostic. All policies had 51
cold-start GPU hits and recomputed 2,832 cold-start blocks. The non-Oracle
policies had 177 training-seen GPU hits; Oracle instead had 149 GPU and 28 host
hits for that group. Per-prefix-block latency attribution is unavailable from
the simulator and is recorded as such rather than estimated.

The native LRU maximum is the simulator-owned terminal reusable occupancy after
active locks drain; it receives no Prism target intervention. Every
adapter-controlled target and every post-decision reusable occupancy remained
within the fixed 9,861-block reserved budget, and pinned pages were never
evicted.

## Scientific answers

1. Training-Popularity Static (Prism) did not improve TTFT, transfer traffic, or
   recomputation relative to native LRU; all three were identical.
2. Validation-Final Frozen (Prism) matched continued Predictive Greedy (Prism) on every reported
   latency, throughput, hit, transfer, and recomputation result. Their test
   targets nevertheless disagreed on all 60 requests (mean Jaccard 0.8683).
3. LFU was not a stronger performance baseline. It matched native LRU while
   adding 5,111 adapter evictions over the full run.
4. Cold start dominated test demand: 94.22% of logical references were to blocks
   absent from training. Prism assigned those blocks no forecast and used no
   rescue policy.
5. Oracle showed no beneficial unrealized opportunity under the frozen
   controller costs. It had the same recomputation, added 56 MiB of transfers,
   and increased mean TTFT by 1.401 ms (0.0086%). Predictive/Oracle mean target
   Jaccard was 0.00029.
6. No Prism gain can be attributed to coverage. Predictive covered 5.78% of
   test logical demand and produced 5,958 more native reusable-residency changes
   than LRU, yet its simulator outcomes were identical to LRU. Static and frozen
   coverage was only 0.92%.

## Determinism and verification

The primary runs were written to `/tmp/prism_m8_full_g` and
`/tmp/prism_m8_full_h`. They completed in 1,272.44 s and 1,297.24 s. The
canonical comparison covered 43 files per root and found no differing paths.
Raw `simulator_stdout.log` and `hook_payload.json` are intentionally excluded
from canonical comparison because they contain progress rendering and
machine-specific output paths. Repeated tiny six-policy roots were likewise
byte-identical across their 43 canonical files.

The pinned upstream was compiled with its `scripts/compile.sh` path in the exact
container environment. An unmodified-upstream one-request smoke completed with
TTFT 11.008270 ms. The verification commands and outcomes were:

| Command | Result |
|---|---|
| `python3 -m pytest -q` before implementation | 254 passed |
| pinned Docker build using `integration/llmservingsim/Dockerfile` and upstream `scripts/compile.sh` | passed |
| unmodified upstream one-request smoke | passed; TTFT 11.008270 ms |
| `DOCKER_CONFIG=/tmp/prism-docker-config DOCKER_HOST=unix:///Users/thaikoehnle/.docker/run/docker.sock PYTHONPATH=src python3 -m prism.llm_sim.cli --config configs/milestone8_llmservingsim.json --output-dir /tmp/prism_m8_full_g --workers 6 --skip-container-build --skip-upstream-build` | six policies passed in 1,272.44 s |
| same command with `--output-dir /tmp/prism_m8_full_h` | six policies passed in 1,297.24 s |
| `canonical_tree_hashes` comparison of the two full roots | 43 files per root, byte-identical, zero differing paths |
| `python3 -m pytest -q` | 278 passed, 1 opt-in test skipped in 14.61 s |
| `PRISM_RUN_LLM_SIM_INTEGRATION=1 DOCKER_CONFIG=/tmp/prism-docker-config DOCKER_HOST=unix:///Users/thaikoehnle/.docker/run/docker.sock python3 -m pytest -q tests/test_llm_sim_integration.py` | 1 passed in 89.35 s; two clean six-policy tiny runs |
| `python3 -m pytest -q tests/test_llm_sim_evaluate.py::test_disabled_hook_preserves_native_router_behavior` | 1 passed in 0.81 s |
| `python3 -m compileall -q src/prism/llm_sim` | passed |
| `git diff --check` | passed |
| `git submodule status --recursive third_party/LLMServingSim` | all recursive SHAs matched; upstream status clean |

The first sandboxed opt-in integration invocation could not access the local
Docker socket and failed before simulator execution. Re-running the identical
test with approved Docker-socket access produced the passing result above; no
product code was changed in response to the environment-only denial.

## Limitations

- Results apply only to this pinned simulator, trace, profile, split, budget,
  controller costs, and seed; they are not real-hardware measurements.
- The full catalog is much smaller than simulator post-weight capacity, so
  native LRU has no natural capacity eviction in this bounded trace. Its
  terminal occupancy therefore exceeds Prism's deliberately smaller target
  partition and should not be read as a capacity-matched LRU cache.
- Host promotion is demand-coupled through native prefix matching. Prism does
  not synthesize asynchronous prefetch, and most target policies generated no
  host hit to promote.
- The adapter depends on revision-specific private radix leaf deletion/event
  methods because the pin exposes no exact-target policy API.
- Neutral `anonymous` user and `llm_request` type constants satisfy the existing
  feature contract but provide no predictive context.
- There was no predictor tuning, test-set adaptation, online NMF, cold-start
  rescue, active-KV control, scheduler modification, learned placement, or
  native-store double modeling. No Milestone 9 behavior was implemented; that
  milestone was later intentionally canceled at project closeout.
