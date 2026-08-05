# Milestone 5 Rigorous Evaluation Execution Plan

**Status:** Complete

## Scope

Evaluate the frozen Milestone 1--4 pipeline across exactly twelve committed
configurations and three fixed seeds. Add only the two required Prism static
controls, three mechanical forecast ablations, causal diagnostics, sequential
experiment orchestration, deterministic aggregation, and focused reporting.

Milestone 4 is scientifically qualified: Predictive Greedy (Prism) made no
test-period promotions, improved access cost only slightly over Recent-Demand
Greedy, and derived most of its combined-cost advantage from avoiding migration.
Three of four training-fitted factor calibrations assigned zero weight to the
activation/intensity term. Milestone 5 therefore tests whether prediction causes
useful dynamic behavior rather than assuming that the prior win demonstrated it.

## Implementation

1. Commit and strictly validate one frozen manifest containing twelve ordered
   variants, seeds `1729`, `2718`, and `31415`, eleven ordered policies, exact
   transformed workload values, and trace-resolved storage formulas.
2. Materialize every workload from the accepted base configuration without
   mutation, then resolve capacity and promotion cost from that generated trace.
3. Reuse the original fitted projection coefficients to construct full,
   recent-state-only, activation/intensity-only, and residual-only forecasts.
   Do not refit any ablation.
4. Extend chronological replay with Training-Popularity Static (Prism) and
   Validation-Final Frozen (Prism) controls, independent state, pre-window
   residency, target, and promotion traces for all eleven policies.
5. Preserve direct Milestone 3 and 4 CLI behavior. Add only an explicit
   evaluation mode that records scientific gate failures while still rejecting
   engineering-invalid workloads, fits, exact solves, or incomplete artifacts.
6. Add a sequential `prism.experiments` harness with deterministic IDs, explicit
   statuses, complete run indexing, resolved-input hashes, one-run selection,
   safe nonempty-output handling, and hash-verified resume.
7. Compute dynamic-action, pre-transition, oracle-miss, paired-cost,
   fixed-trajectory break-even, family, and deterministic hypothesis summaries.
   Persist deterministic JSON and Markdown; do not generate plots.

## Verification and checkpoints

- Checkpoint 1: controls, ablations, manifest, materialization, and focused tests.
- Checkpoint 2: orchestration, statuses, one-run execution, resume, engineering
  failure handling, and reduced integration tests.
- Checkpoint 3: causal diagnostics, aggregation, break-even calculations,
  hypothesis rules, deterministic reports, and focused tests.
- Checkpoint 4: one smoke run, two complete 36-run sweeps, resume verification,
  recursive byte comparison, full regression suite, compile/dependency checks,
  compact honest results, documentation updates, and final clean pushed branch.

## Assumptions, risks, and non-goals

Model convergence and exact-solver optimality are engineering requirements;
scientific gates are recorded outcomes. Three seeds support descriptive sample
statistics only, not confidence intervals or significance tests. Full sweeps may
be slow because execution is deliberately sequential.

No new model, feature, refit, controller objective, policy beyond the required
eleven, parameter tuning, adaptive factor count, additional horizon, TinyLFU,
random replacement, full factorial design, pandas, notebook, plot, database,
parallelism, real storage, C++, pybind11, GPU, LLM, or later milestone is included.
