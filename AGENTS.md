# Prism Agent Instructions

## Project goal

Prism is a predictive storage-tiering research system. It learns latent working sets, forecasts near-future activation and conditional intensity, and passes record-demand forecasts to a deterministic placement controller.

Read `docs/technical_plan.md` before making architectural decisions.

## Current development rule

Implement only the milestone explicitly requested in the task prompt. Do not build future phases merely because they appear in the technical plan.

Prefer the smallest complete, tested implementation that validates the current milestone.

## Core architectural constraints

- ML predicts demand; it does not directly choose storage actions.
- Placement decisions remain deterministic unless a future milestone explicitly changes this.
- Synthetic hidden ground truth must remain separate from model-visible inputs.
- Runs must be reproducible from explicit seeds.
- Policies compared experimentally must receive identical frozen traces.
- Keep slow learning, fast prediction, demand projection, placement, storage, and evaluation separable.
- Do not put expensive prediction or training work on a latency-sensitive request path.
- Do not add dependencies or frameworks without a concrete need.

## Early-phase non-goals

Unless explicitly requested, do not introduce:

- C++ or pybind11 during the initial Python simulator phase
- GPU or CUDA integration
- LLM serving infrastructure
- databases or distributed services
- reinforcement learning
- learned placement controllers
- graph neural networks or transformers
- adaptive working-set counts
- production cloud infrastructure
- premature performance optimizations

## Engineering expectations

- Use clear names and typed interfaces.
- Keep modules narrow and avoid speculative abstractions.
- Validate configuration at boundaries.
- Write tests for deterministic behavior, invariants, and failure cases.
- Never rely on global random state when reproducibility matters.
- Avoid hidden fallbacks that make invalid inputs appear successful.
- Update documentation when behavior or interfaces change.
- Preserve backward compatibility only when the task requires it.

## Before implementation

For a nontrivial milestone:

1. Inspect the repository and relevant documentation.
2. State the proposed implementation plan.
3. Identify assumptions, risks, and explicit non-goals.
4. Confirm that the plan does not extend beyond the requested milestone.

Use an execution-plan document for work spanning multiple substantial components.

## Verification

Before reporting completion:

1. Run the relevant test suite.
2. Run formatting, linting, and type checking when configured.
3. Exercise at least one representative end-to-end path.
4. Review the diff for scope creep and accidental complexity.
5. Report exact commands and results.
6. Clearly disclose anything incomplete, unverified, or intentionally postponed.

## Final response format

Report:

- summary of implemented behavior,
- important design decisions,
- files changed,
- commands run and results,
- representative output,
- assumptions and limitations,
- remaining work for the next milestone.

Do not claim later-phase capability that was not implemented and tested.