# Milestone 8 Execution Plan

**Status:** In progress

## Scope

Milestone 8 integrates Prism with one pinned LLMServingSim 2.0 revision to
evaluate reusable prefix-KV placement. LLMServingSim remains authoritative for
arrivals, scheduling, active-request KV, prefix matching, transfers,
recomputation, and latency/throughput. Prism supplies targets only for eligible,
unpinned reusable GPU prefix blocks inside a fixed reserved budget.

The milestone compares exactly native LRU, LFU, Training-Popularity Static
(Prism), Validation-Final Frozen (Prism), Predictive Greedy (Prism), and Oracle
Greedy. It does not tune the accepted learner, predictor, projection, or greedy
controller and does not use Prism's native C++ store.

## Upstream pin

- Repository: `https://github.com/casys-kaist/LLMServingSim.git`
- Commit: `2c2042ce4bf1b0283ebeed1db95db6f25e3e7511`
- Description: `v1.1.0-53-g2c2042c`
- Nearest release: `v1.1.0`
- License: MIT
- ASTRA-Sim gitlink: `f82fb3d861614a2a61febfaf87a1360db05efd81`
- Location: `third_party/LLMServingSim`

This post-release commit is intentionally pinned because it retains the frozen
RTXPRO6000/Llama-3.1-8B artifacts while including upstream fixes for prefix
cache hit, eviction, reload, multi-tier transfer, and memory accounting.
LLMServingSim's supported simulator environment is the
`astrasim/tutorial-micro2024` Docker image plus the repository's
`scripts/compile.sh`; the profiler/benchmark environment is separate and is not
needed because the required bf16 profile is bundled.

## Frozen configuration

- Model: `meta-llama/Llama-3.1-8B`
- Weight/KV datatype: `bfloat16` / `auto`
- KV block size: 16 tokens
- Hardware: `RTXPRO6000`
- Topology: `configs/cluster/single_node_single_instance.json`
- Full trace: `workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl`
- Tiny trace: `workloads/example_trace.jsonl`
- Split: `[0, 180)`, `[180, 240)`, `[240, 300)`
- Reusable GPU budget: floor of 25% of KV blocks remaining after weights and
  the frozen active-request reservation
- Host capacity: enough for all catalogued reusable blocks

All resolved byte/block values and source hashes are computed before policy
results and reused without tuning.

## Implementation sequence

### Checkpoint 1: pin, catalog, and logical demand

1. Add the exact upstream submodule and deterministic integration config.
2. Build a metadata-only catalog from full 16-token prompt pages. Use the
   simulator-native page hash plus namespace and parent chain for stable block
   identity; retain token IDs only where the simulator adapter requires them.
3. Persist the ordered logical-demand stream and complete request-by-block
   matrix once, with a canonical SHA-256 independent of cache outcomes.
4. Classify training-seen and post-training cold-start blocks without exposing
   future counts or recurrence to deployable policies.
5. Add boundary, identity, leakage, determinism, and malformed-trace tests.

### Checkpoint 2: model, policies, and adapter

1. Fit the existing four-factor deterministic NMF on training requests only.
2. Build the existing one-window predictor contract with one anonymous user,
   one `llm_request` type, one request/session per row, and logical block count
   as observable access count. These constants carry no predictive context.
3. Reuse the accepted training-only projection and greedy benefit controller.
4. Implement the six fixed policy state machines with deterministic ties,
   pre-reveal deployable timing, oracle isolation, static/frozen semantics, and
   no proactive cold-start placement.
5. Add a narrow upstream adapter hook for request boundaries, lifecycle state,
   eligible target application, and metrics. The disabled path is unchanged.
6. Enforce pinned-block protection, prefix closure, fixed reusable occupancy,
   clean per-run state, and identical demand hashes.

### Checkpoint 3: evaluation and verification

1. Run all policies from clean simulator states and preserve raw outputs.
2. Emit the canonical manifest, catalog, demand NPZ, index, per-request JSONL,
   policy results, comparison report, and per-policy run directories.
3. Add deterministic metric aggregation for native latency/throughput,
   cache/transfer/recompute outcomes, cold start, movement, and Prism target
   diagnostics.
4. Run the tiny six-policy integration twice, then the full 300-request
   experiment twice and compare directories.
5. Run upstream hook-disabled smoke coverage, all Prism tests, compile checks,
   `git diff --check`, scope review, documentation, and final clean status.

## Risks and controls

- Prefix blocks are radix-tree dependent. Targets must be prefix-closed; the
  adapter rejects invalid targets and verifies physical occupancy after every
  change.
- Active-request blocks are protected by upstream lock references and are never
  candidates. Only lock-free reusable pages count against Prism's budget.
- Upstream currently has no supported external cache-policy interface. Any
  required patch is kept minimal under `integration/llmservingsim`, hashed in
  the manifest, applied only to a disposable upstream worktree, and tested with
  the hook disabled.
- The local Docker client is installed but the daemon was not running at
  inspection time. Full upstream execution requires the pinned container or a
  compatible prebuilt ASTRA-Sim environment; any unavailable verification will
  be reported rather than replaced with mocked timing.
- Generated experiment roots remain outside Git. Canonical files contain no
  timestamps or absolute machine paths.

## Explicit non-goals

No predictor/model/feature search, online NMF, cold-start rescue, learned
controller, test-set adaptation, active-KV management, scheduler/routing/model
execution changes, ASTRA-Sim changes, native-store double modeling, C++/CUDA,
production serving, or Milestone 9 work is included.
