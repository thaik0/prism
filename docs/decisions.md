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

**Status:** Provisional

**Decision**

The first slow learner should use a simple interpretable matrix-factorization approach, likely nonnegative factorization.

The exact algorithm will be selected during the slow-learner milestone.

**Rationale**

Factorization naturally represents:

- time-varying working-set activation,
- fuzzy record membership,
- overlapping demand patterns,
- a compact latent dimension.

More complex graph or neural approaches are postponed until a simple model exposes a concrete limitation.

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