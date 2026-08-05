# Milestone 3 Fast Predictor Execution Plan

**Status:** Complete

## Scope

Implement one deterministic next-window activation and conditional-intensity
predictor over the dedicated frozen 1,000-window trace. Preserve the Milestone 1
artifact contract and the accepted Milestone 2 NMF algorithm. Do not add
record-level forecasting, placement, caching, storage, online learning, model
selection, neural models, or future-milestone infrastructure.

## Implementation

1. Split examples chronologically by target window using fixed 60/20/20
   boundaries and three observable factor-demand lags.
2. Build the complete raw observable demand matrix, fit the accepted NMF only on
   training rows, freeze normalized memberships, and project all source windows
   through that frozen matrix.
3. Match training-fitted learned memberships to planted memberships once for
   controlled labels. Keep matching, eligibility, burst intensity, and source
   counts outside every model feature and deployable artifact.
4. Reconstruct unique observable requests and create the exact recent-state and
   factor-specific user/request-type interaction schemas in deterministic order.
5. Fit per-factor activation and intensity baselines, two shared fixed logistic
   regressions, and one fixed ridge regression with train-only preprocessing.
6. Evaluate validation and untouched test examples, including all-example and
   hidden-eligible activation populations, calibration, positive-only intensity
   metrics, realized-demand diagnostics, and three strict test gates.
7. Persist exactly four deterministic artifacts with allowlisted arrays and raw
   model parameters sufficient to reproduce in-memory predictions.

## Verification and checkpoints

- Checkpoint 1: split, training-only NMF, matching, features, targets, models, and
  focused tests.
- Checkpoint 2: metrics, gates, leakage protections, deterministic persistence,
  reconstruction tests, and representative end-to-end verification.
- Checkpoint 3: focused documentation, full test suite, two-run byte comparison,
  compile/dependency/diff checks, and final clean pushed branch.

## Assumptions, risks, and non-goals

The planted factor count remains a deliberately supplied controlled-experiment
input. Simulator truth supplies scientific labels but is not a deployable
labeling mechanism for real traces. Validation/test labels may change evaluation
without changing fitted state; training labels necessarily define the supervised
fit. Convergence or scientific-gate failure is reported without tuning, feature
expansion, resampling, or model replacement.
