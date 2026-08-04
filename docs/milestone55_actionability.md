# Milestone 5.5 Predictive Actionability Diagnosis

## Why this milestone exists

Milestone 5 showed that Predictive Greedy usually preserved a good placement
developed during validation, but did not establish continued predictive action.
It matched Validation-Final Frozen in 35 of 36 test runs, matched
Recent-State-Only in cost in 34 of 36, and made a test target change in only one
run. Milestone 5.5 therefore diagnoses the complete path from a moving factor
forecast to an economically useful proactive record promotion. It is a frozen
diagnostic experiment, not a new predictor search.

## Frozen experiment

The schema-2 manifest is `configs/milestone55_actionability.json`. It crosses
three regimes, cumulative decision horizons `H in {1, 2, 4}`, and seeds `1729`,
`2718`, and `31415`, for exactly 27 runs.

| Regime | Precursor scale | Spontaneous probability | Post-burst cooldown |
|---|---:|---:|---:|
| baseline | 0.55 | 0.08 | 0 |
| sparse | 0.275 | 0.04 | 5 |
| very_sparse | 0.1375 | 0.02 | 10 |

Cooldown is maintained independently for each working set. If a burst ends at
window `e`, a cooldown of `c` suppresses activation trials in windows `e`
through `e + c - 1`; the next eligible trial is at `e + c`. Active and cooling
sets do not consume an activation draw. Simultaneous sets remain independent.
The optional field defaults to zero, and ordinary Milestone 1 persistence omits
zero so accepted legacy artifact bytes do not change. Milestone 5.5 canonical
workload artifacts explicitly record the resolved value, including zero.

The sparsity hypothesis was that lower activation probabilities plus cooldown
would create longer dormant intervals and fewer overlapping active bursts, giving
a cumulative predictor time to act before demand arrives. The realized ordering
is checked from hidden truth only after generation; hidden values never enter a
model-visible input.

## Common eligible-window protocol

All horizons use the same feature rows and decision populations:

| Purpose | Half-open window range |
|---|---|
| Training targets | `[3, 597)` |
| Validation evaluation | `[600, 797)` |
| Validation carry-only embargo | `[797, 800)` |
| Test evaluation | `[800, 997)` |
| Final excluded tail | `[997, 1000)` |

The maximum horizon is four. Targets at decision window `t` sum demand over
`[t, t + H)`, so the common tail prevents truncated labels. Validation embargo
decisions update each policy's independent state but do not enter validation
metrics. Placements roll at every eligible boundary; they are not locked for `H`
windows. All policies replay the same frozen observable trace.

Activation labels mean at least one activation in `[t, t + H)`. Conditional
intensity labels and factor-demand targets are sums over the same interval.
Recent-state and oracle inputs are cumulative over that identical horizon.
Separate activation and intensity models are fit for each horizon using the
accepted feature schemas and fixed logistic/ridge estimators. Slow NMF structure,
factor matching, preprocessing, calibration, and residual baselines remain
training-only.

## Factor-to-record diagnosis

The cumulative factor forecast is decomposed into continuation,
activation/intensity, and factor-intercept terms. Training-only nonnegative
calibration projects those terms through frozen fuzzy memberships; a fitted
per-record residual baseline is added separately. This permits measurement of:

- factor forecast movement and cumulative factor error;
- each component's share of the record forecast;
- cumulative record-demand error;
- consecutive record-rank correlation and normalized rank change;
- turnover of the gross-benefit, byte-constrained candidate set.

The deterministic controller then records profitable nonresident records,
movement-cost rejections, capacity rejections, benefit-density margins, target
changes, promotions, and evictions. A movement-cost rejection has nonpositive
net benefit after promotion cost. A capacity rejection is positive-net demand
that loses byte-constrained competition. These are classifications of the fixed
controller objective, not learned actions.

Matched-horizon Oracle Greedy and Oracle Exact expose unused placement
opportunity. Target Jaccard, byte overlap, cumulative demand coverage, missed
oracle records/accesses, and regret separate poor forecast alignment from a lack
of opportunity. Hidden burst intervals are opened only for controlled transition
coverage and promotion timing diagnostics.

A promotion episode begins when Predictive Greedy adds a record during test and
ends at its eviction or the evaluation boundary. A pre-demand promotion precedes
its first access by at most `H` windows. Repayment counts only saved slow-read
cost while resident and compares it with that episode's promotion cost.

## Precommitted thesis decision

Only `sparse__h2`, `sparse__h4`, `very_sparse__h2`, and `very_sparse__h4` may
satisfy the thesis. Baseline and all `H=1` cells remain diagnostic. A candidate
passes only when all four gates pass across its three seeds:

- Gate A: mean Predictive-versus-Frozen target difference is at least 10%, with
  at least two seeds individually at or above 10%.
- Gate B: Predictive combined cost is lower than both Validation-Final Frozen
  and Recent-State-Only in at least two seeds per comparator and in the
  three-seed mean.
- Gate C: Predictive either lowers access cost versus Frozen in at least two
  seeds and in the mean, or improves mean transition coverage by at least 0.05
  with at least two positive seeds.
- Gate D: at least 10 aggregate pre-demand promotions, at least 50% repaid, and
  strictly positive aggregate realized savings minus promotion cost.

Deterministic comparisons use the frozen absolute numerical tolerance `1e-9`;
strict cost and net-value inequalities remain strict as specified.

At least one candidate must pass all four gates for
`actionable_predictive_tiering_demonstrated`. With 27 engineering-valid runs and
meaningful regime separation but no passing candidate, the result is
`stable_cost_aware_tiering_reframe`. Missing engineering coverage or regime
separation yields `insufficient_evidence`.

The manifest, candidate region, thresholds, policies, seeds, and numerical
tolerance were frozen before the full sweep. No post-result tuning is permitted:
changing them would turn a confirmatory gate into a search. Real RAM/SSD work
remains deferred because simulated actionability should be established before
adding systems complexity.
