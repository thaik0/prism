# Milestone 1 Execution Plan

**Status:** Complete

## Scope

Implement only the controlled, reproducible Python workload generator described in
the Milestone 1 brief. No learning, prediction, placement, cache, storage, latency,
or later-phase interfaces are in scope.

## Implementation

1. Add minimal standard-library-only Python packaging and a narrow
   `prism.workload` package.
2. Define a strict `WorkloadConfig` JSON boundary, exact observable schema, hidden
   truth structures, run summary, and `WorkloadResult`.
3. Generate records and hidden distributions, then process fixed logical windows
   in deterministic order. Activation uses only the prior window's precursor
   score; requests in the current window produce the score for the next window.
4. Expose `generate_workload(config)` and a module CLI. Persist exactly
   `config.json`, `observable_events.jsonl`, `hidden_ground_truth.json`, and
   `summary.json` with stable JSON formatting and safe output-directory handling.
5. Add a representative configuration and focused documentation for the API,
   causal model, schemas, reproducibility contract, CLI, and limitations.

## Validation and verification

- Reject missing/unknown configuration keys, invalid types, non-finite values,
  invalid bounds/categories/probabilities, and undefined source mixtures.
- Test deterministic in-memory and byte-level output, distinct fixed seeds,
  observable-field allowlisting, hidden/observable separation, session/request
  relationships, fuzzy memberships, vector normalization, one-window delay,
  burst invariants, source counts, source sampling, operation labels, CLI errors,
  and representative-run acceptance conditions.
- Run the complete pytest suite, two independent representative CLI runs, a
  recursive byte diff, artifact inspection, and Git diff/status checks.

## Assumptions and risks

- Python 3.11 or newer and pytest are available; the runtime has no third-party
  dependencies.
- Floating-point sums are validated with a tight documented tolerance.
- Working-set overlap emerges only from independent support sampling. Tests use a
  full-support configuration for guaranteed overlap; the representative seed is
  selected to demonstrate natural overlap without changing generator semantics.
- Probabilistic properties are tested through formulas and deterministic
  invariants, not unstable empirical frequency thresholds.
