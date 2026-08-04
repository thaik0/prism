# Milestone 5.5 Predictive Actionability Diagnosis Execution Plan

**Status:** In progress

## Scope

Diagnose whether the frozen Prism architecture produces economically useful
proactive placement under exactly three precommitted activation regimes, matched
cumulative horizons `1`, `2`, and `4`, and seeds `1729`, `2718`, and `31415`.
Extend the Milestone 5 sequential harness and preserve all accepted Milestone
1--5 direct interfaces and artifact bytes.

## Implementation

1. Add an optional per-working-set post-burst cooldown whose omitted value is
   exactly legacy zero behavior. Materialize only baseline, sparse, and very
   sparse regime transformations and record their exact field diffs.
2. Build one common horizon-safe training/validation/test population. Reuse the
   accepted feature schemas, logistic/ridge models, frozen training-only NMF,
   and training-only factor matching while changing only cumulative labels.
3. Fit separate training-only NNLS calibration and residual record baselines for
   each horizon. Mechanically reuse fitted coefficients for all Prism ablations.
4. Replay all policies independently on identical events with rolling eligible
   boundary decisions, validation carry-only embargo windows, and an excluded
   final tail. Align recent, static, oracle, and controller economics to the
   same cumulative horizon.
5. Persist deterministic factor, record-rank, controller, oracle-agreement, and
   proactive-promotion repayment diagnostics. Calculate the precommitted Gates
   A--D only for the four candidate cells.
6. Extend the existing runner to exactly 27 ordered runs, explicit status and
   resume hashing, deterministic aggregation, and one thesis decision.

## Verification

- Prove legacy zero-cooldown workloads and direct one-window predictor,
  simulation, and Milestone 5 behavior remain byte-identical.
- Add focused cooldown, regime, horizon, projection, policy, embargo, diagnostic,
  repayment, gate, persistence, leakage, and reduced full-run tests.
- Inspect the frozen manifest before outcomes; run the three specified smoke
  cells, two complete sequential sweeps, resume validation, and a recursive byte
  comparison.
- Run the full tests, compile check, dependency check, whitespace check, scope
  review, and clean remote-aligned Git checks at every checkpoint.

## Assumptions, risks, and non-goals

The accepted base has precursor scale `0.55`, spontaneous probability `0.08`,
and maximum burst duration `5`. Thus the fixed sparse transforms are
`0.275/0.04/cooldown=5` and `0.1375/0.02/cooldown=10`. Storage remains 25% of
trace bytes with a median-record promotion price of two saved reads.

The main risks are preserving legacy bytes, correctly carrying reactive metadata
through embargo windows, and deterministic runtime of repeated numerical fits
and exact solves. New behavior is isolated behind versioned Milestone 5.5
configuration and explicit eligible-window inputs.

No new feature, model family, tuning, seed, horizon, regime, policy, controller
objective, trajectory optimization, parallelism, real storage, C++, pybind11,
GPU, or later milestone is included.
