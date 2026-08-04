# Milestone 5.5 Results

## Outcome

All 27 frozen runs completed with no engineering failures, both full sequential
sweeps were byte-identical, and a resume pass hash-validated and reused all 27.
The regimes separated in the intended order. Nevertheless, none of the four
precommitted candidate cells passed any thesis gate.

The final status is:

```text
stable_cost_aware_tiering_reframe
```

The evidence supports Prism as learned latent-demand structure for stable,
cost-aware storage tiering. It does not demonstrate that the current fast
activation/intensity forecasts cause useful dynamic tiering under the tested
sparse regimes and horizons.

## Frozen design and reproducibility

- Manifest: `configs/milestone55_actionability.json`
- File SHA-256: `5eac5971ba067ade7fbf0efa5044d7778e35eacdddaa2d2979b23b607b1bd8d4`
- Canonical resolved-manifest SHA-256 recorded by reports:
  `c4bc5728721476da6cd8c2cf2a264d6bc4ec8ad79cb658d82bebb3d66d4ea64d`
- Regimes: `baseline`, `sparse`, `very_sparse`
- Horizons: `1`, `2`, `4`
- Seeds: `1729`, `2718`, `31415`
- Candidate cells: `sparse__h2`, `sparse__h4`, `very_sparse__h2`,
  `very_sparse__h4`
- Run IDs use every ordered Cartesian product as
  `<regime>__h<horizon>__seed_<seed>`.

The exact ordered IDs are:

```text
baseline__h1__seed_1729      baseline__h1__seed_2718      baseline__h1__seed_31415
baseline__h2__seed_1729      baseline__h2__seed_2718      baseline__h2__seed_31415
baseline__h4__seed_1729      baseline__h4__seed_2718      baseline__h4__seed_31415
sparse__h1__seed_1729        sparse__h1__seed_2718        sparse__h1__seed_31415
sparse__h2__seed_1729        sparse__h2__seed_2718        sparse__h2__seed_31415
sparse__h4__seed_1729        sparse__h4__seed_2718        sparse__h4__seed_31415
very_sparse__h1__seed_1729   very_sparse__h1__seed_2718   very_sparse__h1__seed_31415
very_sparse__h2__seed_1729   very_sparse__h2__seed_2718   very_sparse__h2__seed_31415
very_sparse__h4__seed_1729   very_sparse__h4__seed_2718   very_sparse__h4__seed_31415
```

The manifest and its 27 ordered IDs were printed and inspected before smoke or
full outcomes. No field, threshold, candidate, seed, or horizon changed after
results were inspected.

Final aggregate artifact hashes from the first full root were:

| Artifact | SHA-256 |
|---|---|
| `aggregate_report.json` | `00b777798de69bfd158d2ac362563fec8df1e177975780bcba629ffefe9222fe` |
| `aggregate_tables.md` | `4145a8b8662dbc05d918b7703e2fd2a0086b0c8b60e9b18743499cf4e4a01e82` |
| `thesis_gate_report.json` | `08bcc69ad634f60026230d0197bc31f48d98d75bc9ce04499972cd3c26b9ca31` |

## Engineering and retained scientific outcomes

Every run reached `completed`, its strict artifact allowlist and hashes validated,
and all policies retained independent state on identical events. Across 10,638
Oracle Exact decision windows, every solver status was `optimal_status_0`; there
were zero capacity violations. Scientific results did not control completion:
all 27 workload demonstration, intensity-signal, and structure gates passed,
while 25 predictor gate sets and eight legacy Milestone 4 simulation gate sets
failed and were retained as evidence.

Nine simulation reports contain explicit warnings: eight note a failed legacy
Milestone 4 scientific gate, and one (`baseline__h4__seed_2718`) notes that
Oracle Greedy costs more than Predictive Greedy under different myopic
trajectories. Oracle Exact was optimal per decision, but its independently
evolving trajectory is not a global trajectory optimum. No engineering run was
discarded because of these unfavorable scientific outcomes.

Backward compatibility was checked against detached accepted Milestone 5 commit
`89e05ae`. The old and current direct workload CLIs generated the same summary
for `configs/milestone3_predictor_workload.json`, and recursive directory diff of
their four artifacts exited zero. Omitted and explicit zero cooldowns also have
identical in-memory outcomes and legacy persistence bytes. The full regression
suite passed; new canonical cooldown metadata is enabled only by the versioned
Milestone 5.5 harness.

## Realized regime separation

`Multi-active` is the fraction of windows with at least two active working sets.
The dormant median is pooled across working sets. Precursor rate is successful
activation among eligible trials with positive prior-window precursor score.

| Regime | Seed | Starts | Starts/100 | Active | Multi-active | Dormant median | Precursor rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1729 | 878 | 87.8 | 0.996 | 0.958 | 1 | 0.481 |
| baseline | 2718 | 890 | 89.0 | 0.996 | 0.962 | 0 | 0.510 |
| baseline | 31415 | 861 | 86.1 | 0.997 | 0.950 | 1 | 0.466 |
| sparse | 1729 | 347 | 34.7 | 0.766 | 0.357 | 7 | 0.245 |
| sparse | 2718 | 350 | 35.0 | 0.779 | 0.377 | 7 | 0.259 |
| sparse | 31415 | 351 | 35.1 | 0.778 | 0.370 | 7 | 0.255 |
| very_sparse | 1729 | 208 | 20.8 | 0.515 | 0.192 | 14 | 0.148 |
| very_sparse | 2718 | 205 | 20.5 | 0.535 | 0.139 | 14 | 0.138 |
| very_sparse | 31415 | 197 | 19.7 | 0.516 | 0.145 | 15 | 0.127 |

Across seeds, starts/100 decreased `87.633 -> 34.933 -> 20.333`, multi-active
fraction decreased `0.957 -> 0.368 -> 0.159`, and dormant median increased
`0.667 -> 7.000 -> 14.333`. All three directional checks passed without
warnings, so this is not an insufficient-evidence result.

## Forecast quality by cell

These are three-seed means on the common test population. Factor and record
errors are for cumulative matched-horizon demand.

| Regime | H | Activation Brier | Intensity RMSE | Factor MAE | Factor RMSE | Record MAE | Record RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 0.1609 | 0.3644 | 1.0421 | 1.3268 | 0.7516 | 1.2802 |
| baseline | 2 | 0.2383 | 0.3914 | 1.5703 | 2.0331 | 1.1252 | 1.9077 |
| baseline | 4 | 0.1664 | 1.1309 | 2.3313 | 3.0409 | 1.6519 | 2.7815 |
| sparse | 1 | 0.0783 | 0.3680 | 1.0837 | 1.5519 | 1.0087 | 1.6590 |
| sparse | 2 | 0.1356 | 0.3859 | 1.9699 | 2.6841 | 1.6534 | 2.6830 |
| sparse | 4 | 0.1778 | 0.3883 | 3.2373 | 4.2328 | 2.5361 | 4.0768 |
| very_sparse | 1 | 0.0506 | 0.4207 | 0.9115 | 1.4217 | 1.0055 | 1.6038 |
| very_sparse | 2 | 0.0932 | 0.4178 | 1.6457 | 2.4688 | 1.6405 | 2.6054 |
| very_sparse | 4 | 0.1615 | 0.4312 | 2.9933 | 4.2546 | 2.6004 | 4.1354 |

The detailed deterministic aggregate records every seed's metrics, fitted
calibration coefficients, residual contribution, scientific outcomes, and
warnings. Longer horizons increased cumulative-demand error; a lower activation
Brier in sparse regimes did not imply controller actionability.

## Where actionability disappeared

Factor forecasts moved in every candidate cell, but the residual record baseline
dominated the projected score and consecutive record order barely changed.
Values below are three-seed means except rejection counts, which are totals.

| Candidate | Factor absolute movement | Rank change | Candidate Jaccard | Continuation share | Activation share | Intercept share | Residual share | Movement rejects | Capacity rejects | Oracle Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse H2 | 0.7629 | 0.0041 | 0.9911 | 0.0381 | 0.0150 | 0.0707 | 0.8761 | 22,801 | 1,627 | 0.4838 |
| sparse H4 | 0.9601 | 0.0023 | 0.9929 | 0.0225 | 0.0297 | 0.0743 | 0.8734 | 15,918 | 8,116 | 0.5833 |
| very-sparse H2 | 0.7012 | 0.0053 | 0.9816 | 0.0345 | 0.0113 | 0.0572 | 0.8969 | 22,878 | 1,353 | 0.4502 |
| very-sparse H4 | 0.9038 | 0.0035 | 0.9867 | 0.0215 | 0.0214 | 0.0626 | 0.8945 | 12,588 | 11,052 | 0.5417 |

Thus the signal weakened mainly during factor-to-record projection: 87--90% of
the projected magnitude came from the stable residual baseline. Gross-benefit
candidate sets had Jaccard `0.982--0.993`; promotion cost removed many remaining
nonresident candidates, and byte capacity removed more at `H=4`. Every candidate
cell made zero test target changes, promotions, or evictions. Yet matched-horizon
oracle Jaccard remained only `0.450--0.583` and missed-oracle demand remained
material, so the result is not explained by absence of placement opportunity.

## Static controls, ablations, and promotion value

In all four candidate cells, Predictive Greedy was exactly identical to both
Validation-Final Frozen and Recent-State-Only for every seed: access, promotion,
and combined-cost differences were all zero, target disagreement was zero, and
transition coverage differences were zero. Training-Popularity Static and the
Activation/Intensity-Only and Residual-Baseline-Only ablations remain reported
per cell and seed in `aggregate_report.json` and `aggregate_tables.md`; they do
not alter the precommitted decision.

There are no successful or failed candidate promotion episodes to sample because
all four candidate cells had zero promotions. Consequently each has zero
pre-demand, later-used, and repaid promotions, zero savings and promotion cost,
undefined repayment fraction, and net value zero. Two diagnostic
`very_sparse, H=1` runs did produce one repaid pre-demand promotion each, but
`H=1` was precommitted as ineligible for thesis passage.

## Exact thesis gates

All seed-level Gate A fractions were `[0, 0, 0]`. Gate B combined-cost
differences against both Frozen and Recent-State-Only were `[0, 0, 0]`. Gate C
access-cost and transition-coverage differences were `[0, 0, 0]`. Gate D had no
promotions and net value zero.

| Candidate | Gate A mean / passing seeds | Gate B mean vs Frozen / Recent | Gate C access / coverage mean | Gate D pre-demand / repayment / net | Result |
|---|---|---|---|---|---|
| sparse H2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / n.a. / 0 | fail A--D |
| sparse H4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / n.a. / 0 | fail A--D |
| very-sparse H2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / n.a. / 0 | fail A--D |
| very-sparse H4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / n.a. / 0 | fail A--D |

Because engineering completeness and regime separation passed, but no eligible
cell passed all four gates, the precommitted rule requires the stable cost-aware
tiering reframe. No model, parameter, or regime was tuned to rescue this result.
Real storage remains deferred, and no later milestone capability is claimed.
