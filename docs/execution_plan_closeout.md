# Prism Project Closeout Execution Plan

**Status:** In progress

## Scope

Close Prism as a completed experimental system. The closeout will synthesize the
committed Milestone 1--8 evidence, explain why the original dynamic-actionability
hypothesis failed, document the narrower supported thesis, and publish a
reproducible `v1.0.0` release.

This work is limited to documentation, release metadata, verification, and fixes
that are necessary to support an existing claim. It adds no model, policy,
experiment, deployment, or product feature. Milestone 9 is intentionally
canceled by the closeout decision; its former place in the plan remains visible
as project history.

## Evidence and writing

1. Treat the frozen Milestone 5 and 5.5 reports as authoritative for causal
   placement conclusions.
2. Treat the Milestone 6 and 7 reports as correctness and semantic-parity
   evidence, not performance evidence.
3. Treat the Milestone 8 report as one pinned simulator result, not a hardware
   or production-serving result.
4. Quote only exact committed or regenerated deterministic results. Explain what
   the numbers establish and what they do not establish.
5. Write one technical synthesis and one candid first-person retrospective;
   neither should read as a milestone checklist.

## Documentation changes

1. Replace the top-level README with a concise landing page.
2. Add `docs/final_report.md`, `docs/lessons_learned.md`, `docs/index.md`, and
   `docs/reproducibility.md`.
3. Add one compact Mermaid architecture diagram that makes the ML/controller
   boundary explicit.
4. Mark the project complete and Milestone 9 canceled without erasing the
   original hypothesis or earlier plan.
5. Correct stale status, future tense, links, commands, and policy labels across
   the documentation tree.
6. Add release notes and align package/build release metadata with `v1.0.0`.

## Verification and release

1. Check internal Markdown links, active-Milestone-9 language, thesis wording,
   `(Prism)` and `(Prism ablation)` labels, diff whitespace, and generated files.
2. Run the full Python suite, bytecode compilation, dependency consistency, and
   an accepted representative workload/structure/predictor path.
3. Run native Debug, AddressSanitizer, and UndefinedBehaviorSanitizer suites,
   plus the forced Python/native parity fixture.
4. Run the pinned Milestone 8 tiny integration when its documented Docker
   environment is available; disclose any environmental limitation.
5. Use at most three focused commits, pushing only passing checkpoints.
6. Tag and publish `v1.0.0` only from a clean branch whose pushed commit matches
   the verified local commit.

## Risks and explicit non-goals

- Stable validation-developed placement must not be described as continued
  fast-predictor action.
- Simulated cost, simulator-native latency, and native semantic correctness are
  different evidence classes and must stay separate.
- Three-seed experiments support paired descriptive conclusions, not confidence
  intervals or general performance guarantees.
- The native store is synchronous, single-process, and untimed. The LLM result
  is pinned to one simulator configuration. Neither is production evidence.
- No post-result tuning, new workload, deployment, benchmark, or Milestone 9
  implementation is part of this closeout.
