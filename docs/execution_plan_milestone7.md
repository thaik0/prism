# Milestone 7 Execution Plan

**Status:** Complete

## Scope

Milestone 7 integrates the Milestone 6 storage engine with Python and proves
operation-level semantic parity for four accepted Python policies:
Training-Popularity Static (Prism), Predictive Greedy (Prism), LRU, and LFU.
Python continues to own workload generation, prediction, benefit calculation,
placement, and replacement decisions. C++ continues to own store construction,
payload integrity, residency, reads, movement, atomic target replacement, and
logical storage counters.

This milestone is synchronous and single-threaded. It does not add asynchronous
I/O, GIL release, callbacks, native policy logic, learned placement, performance
timing, or any Milestone 8 capability.

## Starting point

- Branch: `milestone7` with no initial upstream.
- Commit: `6487941 Document and verify Milestone 6`.
- Worktree: clean.
- Python baseline: 230 tests passed.
- Native baseline: 28/28 tests passed in Debug, AddressSanitizer, and
  UndefinedBehaviorSanitizer builds.
- Package baseline: `python3 -m pip check` passed.

## Implementation sequence

### Checkpoint 1: packaging and private binding

1. Replace the pure-setuptools build backend with pinned scikit-build-core and
   pybind11 build requirements.
2. Extend the existing CMake project with an optional `prism_native` target
   installed as `prism._native`, while preserving standalone tools and CTest.
3. Refactor the native builder so manifest files and in-memory immutable bytes
   share sorting, CRC32, FlatBuffers, verification, and atomic publication.
4. Bind the existing store factory and narrow storage operations directly.
5. Preserve stable native error fields in a private binding exception.
6. Add conversion, builder, read, movement, target-set, metadata, statistics,
   error, and native-tool regression tests.
7. Verify editable import, focused Python tests, and existing native tools/tests;
   then commit and push.

### Checkpoint 2: deterministic payloads and public wrapper

1. Add `prism.native` with one public `NativeStoreError` and immutable result
   dataclasses.
2. Generate deterministic record bytes from record ID and declared size without
   using global random state or hidden model inputs.
3. Build stores only through the bound native builder and compute report SHA-256
   hashes in Python.
4. Write a canonical payload/store manifest and verify every record by reading
   exact bytes and auditing native metadata.
5. Test missing-extension guidance, public conversions, payload determinism,
   store construction, and installed-package behavior; then commit and push.

### Checkpoint 3: reference ledger and parity harness

1. Implement an independent Python reference ledger matching the Milestone 6
   storage contract and counters.
2. Record each expected operation before execution, execute exactly one native
   operation, and compare result, residency, capacity, and cumulative counters.
3. Reuse existing Python demand-benefit and greedy-placement APIs for the two
   Prism policies. Keep LRU/LFU victim selection entirely in Python with the
   accepted deterministic tie-break rules.
4. Add hand-checkable fixtures that force predictive target changes, reactive
   eviction, idempotence, capacity failures, checksum failures, and post-error
   usability.
5. Invalidate mismatched operations and dependent policy summaries explicitly;
   never overwrite native state with expected state.
6. Persist canonical JSONL operations and JSON reports with exactly the required
   top-level output entries; run the fixture twice and compare bytes; then commit
   and push.

### Checkpoint 4: accepted representative run and final verification

1. Generate the committed seed-1729 Milestone 3 source workload and accepted
   predictor artifacts without changing frozen configurations.
2. Run validation and test operations for the four required policies against a
   native store built from deterministic payloads.
3. Run the parity CLI twice and require byte-identical output roots, zero
   mismatches, zero invalidations, and zero capacity violations.
4. Build a wheel, install it into a fresh virtual environment, and run a neutral
   working-directory store lifecycle smoke test while inspecting `prism.__file__`.
5. Rerun the full Python suite, compileall, pip check, native Debug, ASan, and
   UBSan suites, artifact inspection, diff review, and scope review.
6. Document the API, build, semantics, artifacts, verification evidence, and
   limitations; then commit and push the clean branch.

## Risks and controls

- Native failure counters and atomic target staging are compared after every
  operation, including failures, rather than inferred only at the end.
- Direct bytes construction is implemented in the existing builder library so
  the CLI and binding cannot diverge in store-format logic.
- Python reference updates commit only after the corresponding native operation
  succeeds.
- Representative artifacts are written outside the repository and are not
  committed.
- No scientific gate is reinterpreted: this milestone certifies execution
  parity, not new predictive actionability or measured performance.

## Completion evidence

- Checkpoint 1: `d89422e Add pybind11 storage engine bindings`.
- Checkpoint 2: `850e35c Add deterministic native store integration`.
- Checkpoint 3: `655b695 Add simulator and native execution parity`.
- Python regression suite: 254 passed.
- Native Debug, AddressSanitizer, and UndefinedBehaviorSanitizer suites: 29/29
  passed in each configuration.
- Forced fixture: all four policies, zero mismatches, byte-identical repeated
  roots.
- Accepted representative: seed 1729, 130,349 compared validation/test
  operations, zero mismatches/invalidations/capacity violations, byte-identical
  repeated roots.
- Editable package import and an isolated wheel build/install/native-read smoke
  test passed.

Detailed contracts, commands, results, and limitations are in
`docs/python_cpp_integration.md`.
