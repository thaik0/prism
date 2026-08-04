# Milestone 4 Simulated Predictive Placement

## Purpose and boundary

Milestone 4 asks whether frozen Prism forecasts reduce one controlled simulated
storage cost relative to reactive and recent-demand policies. Every policy sees
the same frozen observable events, record sizes, byte capacity, read costs, and
promotion cost.

This is not a real storage engine. The fast and slow tiers are conceptual state,
and costs are deterministic units rather than measured latency. There is no RAM
allocation, SSD or filesystem access, writeback, prediction-time charge, queue,
bandwidth contention, asynchronous migration, concurrency, C++, or pybind11.

## Frozen input contracts

The CLI consumes one Milestone 1 workload directory, one Milestone 3 predictor
directory, and one strict four-field simulator configuration. Before projection
or replay it verifies:

- the required source and exact predictor artifacts;
- SHA-256 correspondence between predictor and workload;
- workload engineering and scientific gates;
- all three predictor scientific gates;
- split boundaries and factor/window coverage;
- record, window, and factor identifiers and dimensions;
- exact predictor NPZ array allowlists; and
- finite predictor probabilities, intensities, and memberships.

The simulator never regenerates or refits Milestone 3. Every record must expose
one consistent immutable size in observable events. Hidden ground truth is opened
only after replay to select controlled transition windows.

## Chronological periods and projection

The dedicated 1,000-window trace retains the accepted split:

```text
training   [0, 600)
validation [600, 800)
test       [800, 1000)
```

Raw observable record demand `X[t, i]` is projected through frozen normalized
learned membership `M[k, i]`:

```text
D[t, k] = sum_i X[t, i] * M[k, i]
```

For each factor, SciPy nonnegative least squares fits `a`, `b`, and `c` using
training target windows only:

```text
factor_forecast[t, k] = max(
    0,
    a[k] * D[t - 1, k]
    + b[k] * activation_probability_for_target[t, k]
             * conditional_intensity_for_target[t, k]
    + c[k],
)
```

Predictor rows use feature window `t - 1` and target window `t`; persisted
predictions are aligned by their target-window IDs. No hidden activation label,
precursor score, source count, validation target, or test target enters the fit.

The factor forecasts are mapped to records. A fixed residual baseline is the
training-target mean of positive observable demand not explained by that factor
projection:

```text
residual[i] = mean_training max(0, X[t, i] - factor_projection[t, i])

record_forecast[t, i] = residual[i]
                      + sum_k factor_forecast[t, k] * M[k, i]
```

Expected demands remain finite nonnegative floating-point values. Validation and
test diagnostics report factor and record MAE, RMSE, and correlation against a
previous-window demand baseline. Diagnostics do not select or tune the model.

## Simulated storage and timing

The slow tier is authoritative and unlimited. The fast tier stores complete
indivisible record IDs under an exact byte limit. Oversized records are always
slow and cannot be selected or admitted. Eviction is free because records are
immutable; promotions and evictions still record counts and bytes.

LRU and LFU operate per event. A miss first pays the slow-read cost, then may
evict and promote the accessed record. The causing access never becomes a hit.
LRU evicts by oldest access index and ascending record ID. LFU maintains
non-decaying counts from validation start and evicts by ascending frequency,
oldest access index, then ascending record ID.

The four boundary policies compute a complete target set before events in each
window, evict unselected residents, promote newly selected records, and hold the
result fixed during the window:

- recent-demand greedy uses observable demand from `t - 1`;
- predictive greedy uses the calibrated record forecast for `t`;
- oracle greedy uses actual observable demand in `t`;
- oracle exact uses that same current-window demand and an exact solver.

The oracle policies never inspect windows after the current target and are not
full-horizon optima.

Every policy begins validation with empty residency and metadata. Validation is
replayed normally, then residency, LRU/LFU metadata, and promotion episodes carry
into test. Validation costs are excluded from primary metrics and gates.

## Shared placement objective

For each record:

```text
gross savings = forecast accesses * (slow read cost - fast read cost)
promotion cost = 0 if resident else size * promotion cost per byte
net benefit = gross savings - promotion cost
```

Nonpositive-benefit and oversized records are excluded. Greedy sorts remaining
records by descending benefit per byte, descending benefit, ascending size, and
ascending record ID, then accepts each fitting record. Unselected residents are
not retained merely to fill unused capacity.

Oracle exact passes the same benefits and byte constraint to
`scipy.optimize.milp` with binary variables and requires an optimal solver
status. Its role is diagnostic: it isolates greedy approximation under perfect
one-window demand.

## Cost, migration, and transition metrics

A resident access costs `fast_read_cost`; a nonresident access costs
`slow_read_cost`. Every nonresident-to-resident transition costs
`record_size_bytes * promotion_cost_per_byte`. Combined cost is access plus
promotion cost. Prediction, calibration, NMF, and controller execution time are
not charged.

The report includes access totals and p50/p95/p99 tier-cost percentiles; hits,
misses, promotions, evictions, and bytes; occupancy; and combined cost. Promotion
episodes begun during test are wasted if they receive no resident access before
eviction or test end. Validation-started episodes retain state but are never
classified as test promotions.

Controlled transition evaluation uses hidden burst intervals only after replay.
It aggregates the unique test windows containing at least one burst start and the
union of each start plus its next window. Unique-window aggregates avoid double
counting simultaneous or overlapping bursts; per-burst rows remain descriptive.

## Fixed scientific gates

All gates use test metrics only:

1. predictive greedy combined cost is below recent-demand greedy;
2. predictive greedy combined cost is below both LRU and LFU;
3. predictive greedy unique burst-start-window cost is below recent-demand
   greedy;
4. oracle greedy combined cost is below recent-demand greedy; and
5. oracle exact combined cost is no greater than oracle greedy, with only a
   `1e-9` absolute floating tolerance.

Oracle greedy versus predictive greedy ordering is a prominent diagnostic rather
than a gate because the two myopic policies can develop different trajectories.

## Configuration and representative evidence

The committed configuration was resolved from static record metadata before
placement outcomes were inspected:

```text
records                         64
total record bytes              618,914
median record size              10,013 bytes
fast capacity                   154,728 bytes
capacity fraction               0.2499991921
fast / slow read cost           1.0 / 10.0
promotion cost per byte         0.0017976630380505342
median-record promotion cost    18.0
```

On the frozen seed-1729 trace, test combined costs were:

| Policy | Combined cost | Hit rate | Bytes promoted |
|---|---:|---:|---:|
| LRU | 142,642.9002 | 0.5443 | 47,416,506 |
| LFU | 117,450.8765 | 0.6458 | 39,120,166 |
| Recent-demand greedy | 70,440.7827 | 0.5356 | 6,766,442 |
| Predictive greedy | 57,764.0000 | 0.5407 | 0 |
| Oracle greedy | 45,154.7717 | 0.7855 | 6,778,674 |
| Oracle exact | 45,122.9486 | 0.7862 | 6,796,017 |

Predictive greedy retained its validation-developed target throughout test, so
it incurred zero test promotion bytes. The result is reported rather than
retuned. It reduced combined cost by `12,676.7827` (`17.9964%`) versus
recent-demand greedy and reduced unique burst-start-window cost by `9,161.4265`
(`19.9589%`). All five gates passed, all 400 validation/test exact solves proved
optimal, the oracle-ordering diagnostic passed, and capacity violations were
zero.

These values describe one controlled trace and one frozen cost configuration.
They are not confidence intervals, parameter sweeps, or evidence of real storage
latency; those questions belong to later milestones.

Milestone 5 supplies that multi-configuration qualification. Across its frozen
36-run sweep, the complete test residency of Predictive Greedy (Prism) matched
Validation-Final Frozen (Prism) in 35 runs, and only the zero-promotion-cost seed
`31415`
run changed test targets. The Milestone 4 cost win therefore cannot be read as
evidence of continued proactive test-time response. See
`docs/milestone5_results.md` for controls, ablations, paired results, transition
diagnostics, scientific-gate outcomes, and determinism evidence.

## Artifacts and reproducibility

The CLI writes exactly:

```text
simulation_config.json
projection_model.npz
policy_traces.npz
evaluation_report.json
```

`projection_model.npz` contains only frozen membership, IDs, calibration
coefficients and residual norms, residual baselines, and training target IDs.
`policy_traces.npz` contains only test event/window policy traces and final
residency. Neither deployable NPZ contains hidden labels, planted identities,
bursts, precursors, hidden source counts, or oracle-demand arrays.

Stable ordering, JSON formatting, NPZ allowlists, no random state, no timestamps,
and no absolute paths make repeated outputs byte-identical in the verified
dependency environment. Exact selections are included in that two-run directory
comparison.
