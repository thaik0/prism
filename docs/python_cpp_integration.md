# Python/C++ Integration and Native Execution Parity

## Status and thesis

Milestone 7 is complete. Prism's supported thesis remains:

> Learned latent-demand structure for stable, cost-aware storage tiering.

This milestone certifies synchronous execution semantics. It does not show that
the current predictor anticipates transitions usefully, and it does not measure
or claim a performance advantage over the Python simulator.

## Ownership boundary

Python owns workload metadata, learned structure, prediction, record-demand
projection, benefit calculation, greedy target selection, LRU/LFU state and
victim selection, orchestration, and parity analysis. C++ owns native store
construction, FlatBuffers metadata, CRC32 generation and verification,
authoritative file-backed bytes, resident byte buffers, reads, promotions,
evictions, exact-target transitions, snapshots, and logical counters.

The synchronous boundary is intentionally narrow:

```text
Python policy decision
-> one explicit native operation
-> native bytes/residency/counters
-> independent Python reference comparison
```

Native state is never overwritten from the reference state. Each policy opens
an independent native store instance and has an independent Python ledger.

## Build and module boundary

The package uses pinned `scikit-build-core==0.11.6` and `pybind11==3.0.4`.
`scikit-build-core` drives the existing CMake project, and the extension links
the existing `prism_storage` target rather than compiling a second storage
implementation. Editable and wheel builds use separate state-qualified CMake
directories so generator or build-mode caches cannot collide.

The extension is private as `prism._native`. Application code uses
`prism.native`, which provides immutable Python result values and the single
public `NativeStoreError`. Pure `prism` and `prism.simulation` imports still work
in a source-only tree. Importing `prism.native` without a built extension gives
an actionable installation message and retains the original import error as
context.

The existing `prism_store_build`, `prism_store_inspect`, and
`prism_store_replay` tools and the complete CTest suite remain independently
buildable. The GIL remains held for every native call.

## Bound operations

The private extension exposes only:

- `build_store(records, output_dir)`;
- `TieredStore.open(store_dir, fast_capacity_bytes)`;
- `read`, `promote`, `evict`, `apply_target_set`, `snapshot`, `stats`, and
  `record_metadata`.

Inputs reject booleans, overflows, malformed tuples, mutable byte buffers, and
non-finite target iterables at the binding boundary. Duplicate target IDs are
passed through for authoritative native rejection. Reads return a detached
Python `bytes` copy; no pointer, file descriptor, native container, or mutable
resident view is exposed.

The builder accepts immutable Python bytes but uses the same C++ core as the
manifest CLI. C++ sorts IDs, calculates CRC32, builds FlatBuffers, verifies the
complete store, and atomically publishes the directory. Python calculates only
the reporting SHA-256 hashes after native construction.

## Deterministic payload schema

Workloads specify record sizes but not application content. Integration payload
schema 1 derives repeated SHA-256 blocks from:

```text
ASCII "PRISM_RECORD_PAYLOAD"
uint32 little-endian schema version
uint64 little-endian workload seed
uint64 little-endian record ID
uint64 little-endian block counter
```

Blocks are concatenated and only the final block is truncated. No random-number
generator or object string representation is used. The resulting byte count is
exact. The manifest records per-record size, native offset, CRC32, SHA-256,
source hashes, store hashes, dependency versions, capacity, and verification
status, without payload bytes, timestamps, hidden truth, or absolute paths.

Setup verifies every record's metadata length and reads every authoritative
payload through C++, comparing its SHA-256 digest before parity begins.

## Errors and operation semantics

`NativeStoreError` preserves native `code`, `message`, `record_id`, `offset`,
and `path`, and the public wrapper adds `operation`. Its rendering is stable and
does not include a traceback, memory address, or unnecessary absolute path in a
canonical parity artifact.

Native semantics remain unchanged:

- a slow read does not promote;
- promotion never chooses an eviction victim;
- repeated promotion and known-absent eviction are idempotent;
- oversized and insufficient-capacity promotions preserve residency;
- exact-target calls reject duplicate, unknown, oversized, or over-capacity
  targets;
- staged target failures discard staged bytes and preserve prior residency;
- all authoritative slow-tier and promotion reads verify CRC32.

The independent `ReferenceLedger` implements these results and logical counters
without consulting native outcomes. Expected and native operation result,
error, payload digest, tier, byte count, residency, capacity, and every logical
counter are compared after each operation.

## Four certified policy paths

Exactly four accepted paths are certified, in fixed order:

1. **Training-Popularity Static (Prism)** computes the accepted training-only
   popularity forecast, applies one greedy exact target before validation, then
   never changes placement.
2. **Predictive Greedy (Prism)** uses the accepted projection, benefit formula,
   and greedy controller at every eligible validation and test boundary.
3. **LRU** performs the native read first, then Python selects individual
   victims by ascending last access and record ID before native promotion.
4. **LFU** updates cumulative frequency at the accepted point and selects by
   ascending frequency, last access, and record ID.

The fixture uses sizes `[2, 3, 4, 6, 11]`, capacity `9`, seed `7007`, explicit
events, and explicit forecasts processed by the production controller. It forces
fast and slow reads, exact-fill and unused-capacity targets, changed and
unchanged targets, multiple outgoing records, reactive promotions and evictions,
LRU/LFU ordering and ties, and structured failure/corruption paths.

## Parity artifacts and invalidation

Both fixture and representative commands reject a nonempty output root and
write exactly:

```text
native_store/store.data
native_store/store.index
native_store_manifest.json
parity_operations.jsonl
parity_report.json
```

The JSONL row contains operation order, policy, phase, window/event coordinates,
arguments, expected/native success or error, detached results, snapshots,
cumulative statistics, status, and classified mismatch details. Read rows store
SHA-256 only, never full payload bytes. JSON keys, policy order, operation order,
and newline termination are deterministic; timing and timestamps are excluded.

Mismatch categories are `operation_result`, `error_code`, `read_tier`,
`payload_size`, `payload_digest`, `residency`, `resident_bytes`, `counter`,
`capacity_invariant`, and `unexpected_exception`. The first mismatch is fully
recorded and conservatively invalidates that policy. Later expected operations
are marked `not_compared_due_to_prior_divergence` and no further native calls are
made for that policy. Other policy instances continue independently.

## Commands

Install or rebuild the editable package:

```bash
python3 -m pip install -e .
python3 -c "import prism; import prism._native; import prism.native; print('native import ok')"
```

Run the built-in forced fixture:

```bash
python3 -m prism.native.cli \
  --fixture \
  --output-dir /tmp/prism_m7_fixture
```

Run the accepted representative inputs after generating the seed-1729 workload
and predictor artifacts:

```bash
python3 -m prism.native.cli \
  --run-dir /tmp/prism_m7_source \
  --predictor-run-dir /tmp/prism_m7_predictor \
  --simulation-config configs/milestone4_simulation.json \
  --output-dir /tmp/prism_milestone7
```

Build a wheel with the pinned backend:

```bash
python3 -m pip wheel . --no-deps --wheel-dir /tmp/prism_m7_wheels
```

The first CMake configuration may fetch the pinned C++ dependencies. Existing
native Debug, ASan, and UBSan commands remain as documented in
`docs/native_storage_engine.md`.

## Verified representative result

The accepted seed-1729 source contains 1,000 windows, 64 records, and 57,164
events. Validation begins at window 600 and test at window 800. Each policy
replayed 22,809 validation/test reads. Across both phases, 130,349 operations
were compared: 66,113 in validation and 64,236 in test.

The 618,914-byte store used capacity 154,728. Every payload passed setup
verification. All four policy summaries passed with zero mismatches, zero
invalidations, zero unexpected exceptions, zero capacity violations, exact
final states, and exact final counters. Two complete output roots were
byte-identical under `diff -ru`.

The representative predictive path was stable: 400 target calls resulted in 15
initial promotions and no eviction. The forced fixture, rather than this stable
representative path, supplies the required dynamic target evidence. This is
execution-parity evidence, not predictive-actionability evidence.

## Limitations and explicit non-goals

Execution is synchronous and single-threaded. Reads copy payloads into Python.
The GIL is not released. There is no background migration, callback, batching,
threading, asynchronous I/O, model transfer into C++, native policy logic,
mutable record, dirty eviction, concurrency, memory mapping, direct I/O, timing
gate, new predictor search, or Milestone 8 integration.
