# Milestone 6 Native Storage Engine Execution Plan

**Status:** Complete

## Scope

Implement the standalone C++17 two-tier storage substrate authorized by the
Milestone 6 brief. The engine executes caller-supplied placement decisions over
immutable variable-sized records; it contains no ranking, admission, victim
selection, prediction, Python integration, or policy logic.

## Implementation

1. Add a repository-native `cpp/` CMake/CTest build with pinned, conventional
   dependencies for FlatBuffers, JSON, CLI parsing, expected results, and tests,
   plus platform zlib for CRC32.
2. Define FlatBuffers store format version 1 with file identifier `PRSM`, checked
   ordered metadata, deterministic payload concatenation, and validated loading.
3. Add a deterministic manifest builder that stages in one exact sibling
   temporary directory, verifies the completed store, and publishes it with one
   directory rename.
4. Implement a read-only `pread` slow tier and explicitly owned in-memory fast
   tier with caller-owned read destinations, individual promotion and eviction,
   and all-or-nothing exact-target transitions.
5. Return expected runtime failures through stable structured errors and expose
   deterministic logical statistics and sorted residency snapshots.
6. Add deterministic build, inspect, and JSONL replay tools with strict inputs,
   stable output ordering, expected-outcome checking, and no canonical timing.
7. Test format validation, deterministic builds, exact bytes, capacity and
   counter invariants, corruption, short reads, failure atomicity, and repeated
   replay artifacts.
8. Verify debug, AddressSanitizer, UndefinedBehaviorSanitizer, Python regression,
   compile, dependency, determinism, whitespace, scope, and Git state gates.

## Checkpoints

1. Build/schema/builder/validation and focused deterministic tests.
2. Tiered engine operations, errors, statistics, and focused tests.
3. Inspection/replay, deterministic reports, corruption, and integration tests.
4. Sanitizers, full regression verification, documentation, and clean pushed
   branch.

## Assumptions, risks, and explicit non-goals

- The milestone brief's C++17 and policy-agnostic requirements supersede older
  planning text that provisionally mentioned C++20 and native baseline policies.
- Supported systems are POSIX macOS and Linux. Offset-based reads use `pread`.
- Atomic directory rename provides final visibility on one supported filesystem;
  it is not a crash-durable transaction or recovery journal.
- Determinism is conditioned on identical inputs and pinned dependency versions.
- CMake is absent from the starting environment and must be installed before
  native verification. Configure also needs network access for pinned sources.
- No Python bindings, model or workload changes, policy implementation, record
  writes, dirty state, concurrency, asynchronous I/O, compression, `mmap`, direct
  I/O, GPU, network service, database, LLM work, or Milestone 7 capability is
  included.
