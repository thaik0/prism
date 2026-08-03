# Milestone 3 Prerequisite Execution Plan

**Status:** Complete

## Scope

Amend only successful burst-intensity sampling so prior-window precursor context
provides useful but imperfect information, then prove the dedicated longer trace
retains Milestone 2 structural recoverability. Do not implement prediction,
features, targets, splits, learning, projection, placement, caching, storage, or
future-phase infrastructure.

## Implementation

1. Add optional `burst_intensity_context_weight`, resolved to `0.0` and strictly
   validated as a finite non-boolean number in `[0, 1]`.
2. Preserve activation, duration, and RNG ordering. After a successful activation,
   blend exactly one uniform intensity draw with the intensity implied by that
   trial's previous-window precursor score.
3. Extend the persisted-run validator with deterministic burst-start intensity
   diagnostics, undefined-metric warnings, and one fixed scientific gate exposed
   by `--require-intensity-signal`.
4. Add the accepted seed-1729, 1000-window, weight-0.6 dedicated configuration,
   focused unit/regression tests, and an end-to-end Milestone 2 compatibility test.
5. Document the mechanism, default, statistics, gate, trace, and explicit limits.

## Verification

- Prove omitted weight and explicit zero resolve identically and persist
  byte-identical four-file runs.
- Retain the accepted representative trace's exact legacy random intensities,
  activation outcomes, source counts, and observable schema at weight zero.
- Generate the dedicated trace twice and recursively compare its source artifacts.
- Run both scientific gates twice and compare the derived validation report.
- Run the accepted Milestone 2 pipeline without retuning and require convergence
  plus mean support recall strictly above analytic chance.
- Run the complete pytest suite, compileall, dependency check, Git whitespace and
  scope review, then commit and push the two verified checkpoints.

## Assumptions, risks, and non-goals

Existing activation trials and bursts contain all simulator-only data needed for
the diagnostics, so hidden and observable schemas remain unchanged. Statistics
describe one deterministic controlled trace and are not claims of general model
performance. Exact reproducibility remains conditioned on the explicit seed and,
for NMF artifacts, the numerical dependency environment. No predictor or
Milestone 2 algorithm change is included.
