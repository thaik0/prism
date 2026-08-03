# Milestone 2 Execution Plan

**Status:** Complete

## Scope

Implement one observable-only raw demand matrix, one deterministic trace-local
NMF fit, and controlled comparison with planted Milestone 1 working sets. Do not
add prediction, transformed features, placement, caching, storage, C++, model
selection, or alternative learners.

## Implementation

1. Add strict four-field learner configuration and a committed representative
   configuration with planted factor count `K=4`.
2. Build a dense integer window-by-record matrix from `config.json`,
   `observable_events.jsonl`, and `summary.json` only.
3. Fit exactly one fixed scikit-learn NMF using `float64` inputs, capture
   convergence diagnostics, and normalize membership rows with reciprocal
   activation scaling.
4. Load hidden memberships and per-window source counts only after fitting. Use
   cosine similarity and SciPy optimal assignment, then report fuzzy, support,
   reconstruction, chance, and activation-alignment metrics.
5. Persist exactly four deterministic artifacts with allowlisted arrays, stable
   JSON formatting, source hashes, dependency versions, and no absolute paths or
   timestamps.
6. Add focused unit, leakage, persistence, determinism, and representative
   end-to-end tests plus Milestone 2 documentation.

## Validation

- Regenerate and scientifically validate the committed Milestone 1 trace before
  running the learner.
- Run two independent Milestone 2 CLI outputs and compare them recursively.
- Inspect artifact names, array allowlists, dimensions, normalization, metrics,
  convergence, and the strict above-chance representative gate.
- Run the full pytest suite, compileall, Git whitespace checks, diff review, and
  final status inspection.

## Assumptions, risks, and non-goals

- Source schema version 1 uses contiguous integer window and record IDs.
- The source `num_working_sets` is the deliberately supplied planted `K`; no
  factor-count discovery is attempted.
- Exact reproducibility is conditioned on identical source bytes, configuration,
  dependency environment, and seed.
- A valid baseline may fail to converge or recover above chance. Such a result is
  reported without automatic tuning, preprocessing, restarts, or model changes.
- Milestone 1 generator behavior and its four-file contract remain unchanged.
