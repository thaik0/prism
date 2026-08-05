# Milestone 1 Scientific Validation Execution Plan

**Status:** Complete

## Scope

Validate persisted Milestone 1 workload runs for structural consistency and for
the intended learnable-but-imperfect synthetic signal. Preserve the generator,
its API, and its four-artifact contract. Do not implement Milestone 2 behavior.

## Implementation

1. Add one standard-library validation module with a callable API and module CLI.
2. Read and hash the four source artifacts, validate their schemas and
   relationships, and reconstruct unique requests and sessions from observable
   events.
3. Classify eligible trials per working set with inclusive quartiles, excluding
   insufficient and degenerate sets from demonstration counts.
4. Calculate context-signal, working-set structure, burst-diversity,
   demand-decomposition, and supported observable-association diagnostics.
5. Deterministically replace only `workload_validation.json`; never modify source
   artifacts. Make missing demonstrations fatal only with
   `--require-demonstrations`.
6. Add focused corruption, classification, arithmetic, persistence, CLI, and
   representative-run tests, then document the report and its scientific limits.

## Verification

- Capture the existing representative run before any configuration decision.
- Report its three demonstration counts with the new validator.
- Prove report byte determinism and source-artifact preservation.
- Run the complete pytest suite, compile validation, representative CLI gate,
  diff hygiene, scope review, and final Git status.

## Assumptions and non-goals

- Schema version 1 is the supported persisted input contract.
- Structural numeric comparisons use an absolute tolerance of `1e-12`.
- Observable association ranges use a descriptive support floor of 10 paired
  eligible-trial observations.
- Diagnostics do not impose universal scientific-quality thresholds.
- No generator retuning, learning, demand matrices, factorization, prediction,
  caching, placement, storage simulation, or new dependency is planned.
