# Prism — Design Decisions

This document records design decisions that are currently considered settled.

It is not a general planning document. Each entry should describe:

- the decision,
- why it was made,
- alternatives considered,
- consequences or limitations.

Implementation details may change without updating this document unless they represent a meaningful architectural decision.

---

## D001 — Project identity

**Status:** Accepted

**Decision**

The project is named **Prism**.

Prism is a general predictive tier-management system. It is not specifically a database, LLM cache, or quantitative-finance system.

LLM inference will be the first realistic application after the controlled workload phase.

**Rationale**

A general system provides a cleaner technical thesis and makes later cross-domain evaluation possible.

The project should not be tied to one workload before its central mechanism has been validated.

---

## D002 — Workload uncertainty is the primary problem

**Status:** Accepted

**Decision**

Prism focuses on uncertainty in future access demand.

It does not initially model uncertain values, probabilistic database records, or probabilistic query semantics.

**Rationale**

Predicting future demand directly supports proactive storage placement.

Probabilistic data semantics is a separate research direction and would substantially increase project scope.

---

## D003 — Machine learning predicts demand, not storage actions

**Status:** Accepted

**Decision**

The ML components produce forecasts of future demand.

A deterministic controller combines those forecasts with known system state and decides which records should occupy each tier.

**Rationale**

Future demand is uncertain, but capacity, residency, record size, and storage cost are directly observable.

Using ML only for the uncertain portion keeps the controller faster, easier to audit, and easier to evaluate independently.

A learned controller may be considered only as a future extension if deterministic control is shown to be a meaningful bottleneck.

---

## D004 — Two-timescale prediction architecture

**Status:** Accepted

**Decision**

Prism uses two conceptual learning timescales:

1. A slow learner discovers fuzzy, overlapping latent working sets.
2. A fast predictor estimates near-future activation and intensity for those working sets.

**Rationale**

Relationships among records may persist longer than individual workload activations.

Separating structural learning from fast activation prediction provides a compressed representation and avoids scoring every record independently from raw context.

---

## D005 — Fuzzy latent working sets

**Status:** Accepted

**Decision**

A record may belong to multiple working sets with different nonnegative membership strengths.

Working sets are predictive latent structures and do not need to correspond to human-labelled workflows.

**Rationale**

Real access patterns may overlap. A record may be useful to several recurring workflows or regimes.

Strict clustering would lose this structure.

---

## D006 — Initial slow learner family

**Status:** Accepted

**Decision**

The first slow learner uses one deterministic scikit-learn nonnegative matrix
factorization of raw window-by-record access counts. The controlled experiment
supplies the planted factor count and uses fixed NNDSVDa initialization,
coordinate descent, and Frobenius loss.

**Rationale**

Factorization naturally represents:

- time-varying working-set activation,
- fuzzy record membership,
- overlapping demand patterns,
- a compact latent dimension.

More complex preprocessing, model selection, graph, or neural approaches are
postponed until this measured baseline exposes a concrete limitation.

---

## D007 — Fast predictor output

**Status:** Accepted

**Decision**

For each working set and prediction horizon, the fast predictor should estimate:

1. probability of activation;
2. expected demand intensity conditional on activation.

**Rationale**

Activation probability and activation magnitude represent different uncertainties.

Keeping them separate improves calibration, interpretability, error analysis, and controller flexibility.

---

## D008 — Primary workload pattern

**Status:** Accepted

**Decision**

The controlled simulator will primarily model abrupt bursts:

`dormant → abrupt activation → active interval → dormant`

Contextual precursors will be informative but imperfect.

The workload must include:

- precursor followed by activation,
- precursor followed by no activation,
- activation without a clear precursor.

**Rationale**

Abrupt changes expose the central weakness of reactive caching policies: they adapt only after misses occur.

Imperfect precursors ensure the task remains probabilistic rather than becoming a deterministic rule lookup.

---

## D009 — Controlled workload context

**Status:** Accepted

**Decision**

The initial observable context will include:

- user ID,
- session ID,
- request type,
- operation type,
- recent access history or demand state.

Users and request types will have hidden probabilistic affinities to working sets.

A raw session ID will not itself contain reusable predictive meaning.

**Rationale**

These fields are simple to generate and resemble context that a real storage layer or application could provide.

Context must be statistically related to future demand without directly revealing hidden working-set labels.

---

## D010 — Hidden truth separation

**Status:** Accepted

**Decision**

Observable access events and simulator-only hidden ground truth must use separate schemas and storage structures.

Hidden information must never appear in model-visible event records.

**Rationale**

Synthetic ground truth is needed for evaluation, but accidental leakage would invalidate later ML results.

Structural separation is safer than relying on naming conventions or developer discipline.

---

## D011 — Reproducibility

**Status:** Accepted

**Decision**

All controlled simulations and experiments must be reproducible from explicit configuration and random seeds.

Policies must be compared using identical frozen traces.

**Rationale**

Without deterministic replay, policy comparisons and debugging would be unreliable.

Prism is an experimental system, so reproducibility is a first-class requirement.

---

## D012 — Variable-sized indivisible records

**Status:** Accepted

**Decision**

The system stores variable-sized records.

Records cannot be divided by the placement controller.

Fast-tier capacity is measured in bytes rather than record count.

**Rationale**

Variable sizes create a realistic placement tradeoff and prevent the problem from reducing to selecting a fixed number of objects.

The resulting controller problem resembles 0/1 knapsack rather than fractional allocation.

---

## D013 — Read-focused immutable-record MVP

**Status:** Accepted

**Decision**

The initial real storage engine will focus on reads and immutable records.

The slow tier is authoritative. The fast tier contains cached copies.

**Rationale**

This avoids writeback, dirty-record handling, transactions, durability protocols, and compaction while preserving the core predictive-tiering problem.

Writes, updates, and deletion may be added later.

---

## D014 — Initial storage hierarchy

**Status:** Accepted

**Decision**

The first real system will use:

- process RAM as the fast tier;
- a simple file-backed local SSD store as the slow tier.

**Rationale**

This provides real storage behavior without depending on a database or external cache whose own caching and buffering could obscure measurements.

GPU memory is a later milestone.

---

## D015 — Placement objective

**Status:** Accepted

**Decision**

The initial controller will optimize expected read-latency savings over a fixed horizon.

Conceptually, a record’s value depends on:

- predicted future reads,
- fast-versus-slow latency difference,
- promotion cost,
- record size,
- current residency.

**Rationale**

Expected latency directly connects forecast quality to system performance and is simple enough to audit.

Tail latency will be measured but will not initially be the optimization objective.

---

## D016 — Controller modes

**Status:** Accepted

**Decision**

Prism should eventually support:

- a fast greedy predictive controller;
- an exact controller for sufficiently small evaluation cases;
- an oracle controller using true future demand;
- conventional reactive baselines.

The greedy predictive controller is the intended runtime default.

**Rationale**

Comparing greedy, exact, and oracle modes separates:

- prediction error,
- controller approximation error,
- total opportunity available in the workload.

---

## D017 — Primary system metric

**Status:** Accepted

**Decision**

Mean end-to-end read latency is the primary MVP optimization metric.

Prism will also report:

- p50, p95, and p99 latency;
- transition-period latency;
- hit rate;
- bytes moved;
- wasted promotions;
- prediction and migration overhead.

**Rationale**

Mean latency aligns with the initial controller objective, while the additional metrics prevent misleading conclusions.

---

## D018 — Technology boundary

**Status:** Accepted

**Decision**

The provisional technology split is:

- Python, PyTorch, and NumPy for workloads, learning, experiments, and analysis;
- C++20 for the real storage path, policies, placement, migration, and instrumentation;
- pybind11 for the initial Python/C++ boundary;
- CMake for C++ builds.

**Rationale**

Python supports rapid ML experimentation.

C++ is appropriate for memory ownership, concurrency, I/O, and latency-sensitive storage operations.

---

## D019 — Asynchronous prediction

**Status:** Accepted

**Decision**

Foreground storage requests must not synchronously wait for Python or PyTorch inference.

Prediction and training will update forecast snapshots asynchronously.

**Rationale**

Slightly stale forecasts are preferable to adding Python model latency to every storage request.

This also preserves a realistic deployment path.

---

## D020 — Validation progression

**Status:** Accepted

**Decision**

Prism will progress through:

1. controlled Python simulator;
2. slow working-set learner;
3. fast activation/intensity predictor;
4. simulated placement and rigorous evaluation;
5. real C++ RAM/SSD engine;
6. Python/C++ integration;
7. open-source LLM inference simulator;
8. real CPU/GPU/storage deployment.

Quant or market-data workloads are postponed until after the inference path is established.

**Rationale**

This progression isolates errors early and adds realism incrementally without requiring Prism to build an inference simulator itself.

---

## D021 — Complexity policy

**Status:** Accepted

**Decision**

Prism will begin with the smallest trustworthy implementation at each milestone.

The following are postponed unless evidence justifies them:

- reinforcement learning,
- learned controllers,
- graph neural networks,
- transformers,
- adaptive working-set counts,
- multiscale prediction,
- custom CUDA kernels,
- distributed infrastructure,
- cloud deployment.

**Rationale**

The project’s value comes from testing the predictive-tiering thesis, not from maximizing the number of technologies used.

---

## D022 — Initial fast predictor baseline

**Status:** Accepted

**Decision**

The first fast predictor uses one-window chronological targets, training-only
frozen NMF memberships, per-factor constant baselines, two shared fixed logistic
regressions, and one shared fixed ridge regression. Context is represented by
factor-specific user and request-type fraction blocks. Hidden simulator truth is
limited to controlled matching, labels, and evaluation.

**Rationale**

Fixed linear models directly test whether recent learned-factor demand predicts
transitions and whether observable context adds signal without model selection or
test-driven complexity. Separating activation probability from conditional
intensity preserves the accepted forecasting contract and makes calibration and
error sources auditable.

**Consequences and limitations**

The controlled labels are not available for real traces, and the current model
does not use session history, operation type, multiple horizons, online updates,
neural architectures, record-level projection, or placement. Those capabilities
remain deferred until a later milestone or measured limitation justifies them.

---

## D023 — Controlled simulated placement experiment

**Status:** Accepted

**Decision**

Milestone 4 uses one deterministic two-tier cost simulation with a byte capacity
equal to one quarter of the representative trace's total record bytes, fixed
fast/slow read costs of `1.0` and `10.0`, and a median-record promotion cost equal
to two saved slow reads. It compares exactly LRU, LFU, recent-demand greedy,
predictive greedy, one-window oracle greedy, and one-window oracle exact.

Projection calibration is fit on training windows only. Policies begin validation
empty, carry independent state into test, and replay identical observable events.
Greedy and exact controllers share expected access savings minus promotion cost;
ML does not select placement actions.

**Rationale**

One frozen configuration prevents test-driven storage-cost tuning. Validation
warm-up avoids an artificial synchronized cold start at the primary test
boundary. The recent-demand comparison isolates predictive value while holding
the deterministic controller fixed; reactive and oracle policies expose other
important baselines and available opportunity.

**Consequences and limitations**

The oracle sees only current-window demand and is not a full-horizon optimum.
Independent myopic trajectories can affect ordering. Simulated tier costs omit
prediction CPU time, queues, contention, concurrency, real migration timing,
filesystem behavior, and wall-clock latency. These controlled results do not
establish real RAM/SSD performance.

---

## D024 — Frozen causal evaluation of predictive placement

**Status:** Accepted

**Decision**

Milestone 5 evaluates the frozen Milestone 1--4 pipeline with exactly 12
one-factor-at-a-time variants, seeds `1729`, `2718`, and `31415`, and eleven
policies. Training-Popularity Static and Validation-Final Frozen isolate static
placement value. Recent-State-Only, Activation/Intensity-Only, and
Residual-Baseline-Only mechanically remove forecast terms while retaining the
original training-fitted coefficients and deterministic greedy controller.

Engineering validity and scientific outcomes are separate. A valid fit, exact
solve, and complete artifact set produces a completed run even when an earlier
scientific gate fails. Aggregation retains seed-level paired differences and
uses deterministic `supported`, `mixed`, `not_supported`, or
`insufficient_data` rules. Three seeds do not justify confidence intervals or
significance tests.

**Rationale**

Milestone 4's representative Predictive Greedy result made zero test promotions,
had only a small hit-rate advantage over Recent-Demand Greedy, and assigned zero
activation/intensity weight to three of four factors. Static and mechanical
controls are needed to distinguish a useful validation-developed placement from
continued predictor-driven anticipation.

**Consequences and limitations**

The completed sweep does not support dynamic value or fast-predictor contribution:
35 of 36 runs exactly matched the frozen test residency, while only
`promotion_0__seed_31415` acted dynamically. The full policy still often beat
training-only popularity and always had positive regret to Oracle Greedy. These
findings motivate future investigation but do not change the model, feature set,
projection, objective, policy set, or storage implementation in this milestone.

---

## D025 — Reframe Prism around stable cost-aware tiering

**Status:** Accepted

**Decision**

Following the frozen Milestone 5.5 diagnosis, Prism's supported thesis is learned
latent-demand structure for stable, cost-aware storage tiering. The current fast
activation/intensity predictor remains a separable research component, but is not
claimed to cause useful dynamic placement. The deterministic controller and the
project/package name remain unchanged.

**Rationale**

All 27 engineering-valid runs completed reproducibly and the sparse regimes
separated as intended. In every precommitted candidate cell, Predictive Greedy
made zero test target changes and matched both Validation-Final Frozen and
Recent-State-Only exactly. Factor forecasts moved, but a dominant stable residual
record baseline, stable ranking, promotion cost, and byte competition prevented
action. All candidates therefore failed all four precommitted gates.

**Consequences and limitations**

No post-result feature, model, parameter, threshold, seed, horizon, or regime
search is authorized by this decision. Real storage remains deferred, so the
evidence concerns deterministic simulated costs rather than measured latency.
Positive oracle regret preserves a future research question, but not a claim of
demonstrated dynamic predictive tiering.

---

## D026 — Policy-agnostic native immutable two-tier engine

**Status:** Accepted

**Decision**

Milestone 6 uses a standalone C++17 engine with one deterministic FlatBuffers
version-1 `PRSM` index, zlib CRC32 per immutable payload, a read-only `pread`
slow tier, and explicitly owned in-memory fast buffers. Placement callers supply
complete exact target sets. The engine performs no ranking, admission, or victim
selection.

Target transitions verify and retain all incoming payloads and construct the
complete next state before one no-throw swap. Runtime failures use pinned
`tl::expected` structured results. Build, inspection, logical counters, snapshots,
and replay artifacts are deterministic; timing is excluded from canonical output.

**Rationale**

This isolates systems correctness from the unsupported dynamic-prediction claim
and lets future deterministic policies execute against real file bytes and owned
memory without policy behavior leaking into storage. FlatBuffers, zlib,
nlohmann/json, CLI11, tl::expected, and Catch2 avoid custom serialization,
checksum, parser, result, CLI, and test infrastructure.

**Consequences and limitations**

The directory rename is atomically visible on one supported filesystem but not a
crash-durable transaction. macOS and Linux are supported; Windows, writes,
concurrency, asynchronous I/O, memory mapping, direct I/O, Python bindings,
policies, and wall-clock performance claims are deferred.
