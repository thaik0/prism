# Cloud Phase 2 Cross-Host Parity Execution Plan

## Status and boundary

**Status:** Complete

This follow-up corrects the verification contract exposed by the first real
AWS Batch run. It does not change AWS resources, the accepted workload, NMF,
predictor, projection, placement, native execution, scientific gates, seeds, or
canonical artifact formats.

## Diagnosis and evidence

- The exact Phase 2 image (`fa19e2cd...`, Linux/aarch64) produced two
  byte-identical 30-file roots on one Docker host.
- Its NumPy wheel uses OpenBLAS 0.3.30 and selected 12 threads plus the
  `neoversen1` kernel on that host.
- Changing only the OpenBLAS thread count changed NMF results at roughly
  `1e-16` to `1e-13`, beginning in `learned_structure.npz` and
  `recovery_report.json`, while convergence, factor matching, gates, and all
  discrete outputs remained unchanged.
- Forced generic ARM kernels produced other last-bit results. A Neoverse V1
  kernel could not execute on the local CPU. The reported AWS result is another
  value in this same numerical envelope and was not reproduced by thread-count
  control alone.

The boundary is therefore host-selected floating-point kernel and reduction
execution, not workload generation, input identity, missing random seeds, or
AWS orchestration. A fixed seed and pinned wheel determine the mathematical
procedure but do not require bitwise-identical BLAS reductions on distinct
CPUs. Thread pinning alone is not a demonstrated cross-host byte-identity fix.

## Implementation plan

1. Preserve `repeat` as complete byte identity for repeated execution in one
   numerical host/runtime environment.
2. Add `cross-host` for distinct hosts running the same declared container
   runtime. Require exact package, source, input, Python, OS, architecture, and
   dependency identity.
3. Verify every manifest artifact: compare non-numerical artifacts byte for
   byte; compare NPZ names, shapes, dtypes, and discrete arrays exactly; compare
   floating arrays and numerical JSON fields with the existing absolute
   `1e-9` scientific tolerance; and compare all statuses, IDs, keys, list
   structure, booleans, integers, and other values exactly.
4. Keep `cross-platform` backward compatible while applying the same complete
   artifact classification. It intentionally does not require identical
   declared runtime metadata.
5. Add focused regression tests, update the Phase 1/Phase 2 documentation and
   verification record, run local exact-image repeat and cross-host checks, and
   run the full repository suite.

## Assumptions, risks, and non-goals

- The existing `1e-9` absolute tolerance is retained because it is the accepted
  Phase 1 numerical evaluation tolerance and the observed cross-host drift
  remains below it.
- Hash fields derived from numerical artifacts may differ. Their key structure
  remains exact, each root's manifest hashes are independently validated, and
  source/input identity is checked separately.
- Exact cross-host floating bytes would require a separately demonstrated
  portable numerical kernel or a more invasive exact-arithmetic/canonical
  quantization design. Neither is justified by this diagnosis.
- No AWS API call, image publication, infrastructure mutation, or Batch job is
  part of this fix.
