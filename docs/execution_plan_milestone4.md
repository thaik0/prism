# Milestone 4 Simulated Placement Execution Plan

**Status:** In progress

## Scope

Consume one frozen Milestone 1 trace and one frozen, gate-passing Milestone 3
predictor run. Fit only the observable record-demand projection on training
windows, then compare exactly six deterministic simulated placement policies on
identical validation warm-up and test events. Do not modify the accepted
workload, structure, or predictor algorithms.

## Implementation

1. Validate strict source, predictor, split, hash, dimension, identifier, array,
   and scientific-gate contracts before fitting or replay.
2. Fit per-factor nonnegative calibration on training target windows only, add
   one fixed training residual baseline per record, and report held-out factor
   and record forecast diagnostics.
3. Add an indivisible-record, byte-constrained simulated fast tier plus one
   shared expected-benefit objective, deterministic greedy controller, and
   SciPy MILP exact controller.
4. Replay LRU and LFU at event level and recent-demand greedy, predictive
   greedy, oracle greedy, and oracle exact at window boundaries. Begin validation
   empty and carry independent policy state into test without resets.
5. Account for access and promotion cost, migrations, occupancy, promotion
   episodes, wasted bytes, access-cost percentiles, and transition-region cost.
6. Evaluate all five fixed test gates and the oracle-ordering diagnostic, then
   persist exactly four deterministic artifacts with strict array allowlists.
7. Add focused unit, leakage, fairness, persistence, and representative
   end-to-end tests before completing focused documentation.

## Verification and checkpoints

- Checkpoint 1: configuration, projection, storage state, greedy/exact
  controllers, and focused tests.
- Checkpoint 2: six policies, chronological replay, accounting, timing, and
  focused tests.
- Checkpoint 3: diagnostics, transitions, gates, persistence, representative
  run, and two-run byte comparison.
- Checkpoint 4: documentation, complete test suite, compile/dependency/diff
  checks, final scope review, and clean pushed branch.

## Assumptions, risks, and non-goals

The representative source exposes every record's immutable size in observable
events. Exact selection is conditioned on SciPy's deterministic behavior in the
verified environment. Myopic policies can develop different trajectories, so
oracle ordering is diagnostic while the exact-versus-greedy gate uses the fixed
specified tolerance. Gate failures are reported without tuning.

No real storage, C++, concurrency, TinyLFU, learned placement, parameter sweep,
test-set tuning, prediction-overhead charging, or later-milestone implementation
is included.
