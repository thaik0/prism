# Prism — Technical Planning Document

## Document status

**Status:** Initial architecture approved  
**Current implementation phase:** Milestone 3 complete — Fast Activation and Conditional Intensity

**Last major planning revision:** August 2026

This document is Prism’s architectural source of truth.

It describes the intended system, research thesis, component boundaries, evaluation philosophy, MVP, and long-term progression.

It is not an instruction to implement every component immediately.

Implementation must follow `docs/milestones.md`, and workers must implement only the active milestone.

Settled architectural decisions are summarized in `docs/decisions.md`.

---

# 1. Project Thesis

Prism is a predictive storage-tiering system that learns recurring structure in data demand, forecasts which groups of records are likely to become valuable in the near future, and proactively reorganizes constrained fast-storage tiers before that demand fully arrives.

The central research question is:

**Can a storage system predict context-triggered changes in future working-set demand early enough to prepare storage proactively, while keeping prediction and movement overhead below the resulting latency benefit?**

Prism targets workloads with recurring, context-predictable demand transitions where reactive policies adapt only after expensive misses occur.

---

# 2. Core Architecture

Prism separates uncertain prediction from deterministic systems control.

## Slow learner

Discovers fuzzy overlapping latent working sets from historical access behavior.

## Fast predictor

Uses recent workload state and optional application context to predict, for each working set:

- probability of activation within horizon \(H\);
- expected demand intensity conditional on activation.

## Record-demand projection

Maps working-set forecasts through fuzzy memberships to estimated near-future record demand.

## Deterministic controller

Combines predicted demand with known system state:

- record sizes;
- current residency;
- tier capacity;
- measured latency;
- promotion cost.

It then selects a feasible placement intended to minimize expected read latency.

## Tier engine

Executes promotion, eviction, retrieval, and measurement across real storage tiers.

---

# 3. Design Principle

Machine learning should be used only where meaningful uncertainty exists.

ML predicts future demand because future demand is unknown.

The controller remains deterministic because:

- capacity is known;
- record sizes are known;
- residency is known;
- storage costs can be measured;
- movement costs can be estimated.

A learned controller is postponed unless deterministic placement is later demonstrated to be insufficient.

---

# 4. Initial Scope

The initial complete system will support:

- record-oriented storage;
- variable-sized indivisible records;
- immutable record contents;
- a constrained RAM fast tier;
- a file-backed local SSD slow tier;
- contextual access events;
- fuzzy overlapping working sets;
- abrupt demand bursts;
- asynchronous prediction;
- deterministic predictive placement;
- conventional caching baselines;
- reproducible experimentation;
- oracle evaluation and error decomposition.

The initial optimization target is mean end-to-end read latency.

Tail metrics and movement overhead will also be measured.

---

# 5. Explicit Non-Goals

The initial project does not include:

- a complete database;
- transactions;
- WAL or durability protocols;
- dirty writeback;
- compaction;
- distributed storage;
- cloud deployment;
- Kubernetes;
- reinforcement learning;
- learned placement control;
- graph neural networks;
- transformers for workload prediction;
- adaptive working-set counts;
- multiscale horizons;
- custom CUDA kernels;
- a custom LLM simulator.

These may be reconsidered only when experiments reveal a concrete need.

---

# 6. Controlled Workload

The first workload is a configurable generative family with:

- \(N\) variable-sized records;
- \(K\) sparse fuzzy latent working sets;
- overlapping memberships;
- configurable users;
- sessions containing multiple requests;
- request types;
- hidden context-to-working-set affinities;
- abrupt burst activations;
- variable burst duration;
- variable burst intensity;
- contextual precursors;
- false precursors;
- unannounced bursts;
- baseline accesses;
- background noise;
- reproducible seeds.

The hidden generation process is approximately:

**observable context and recent state**  
→ **hidden activation probabilities**  
→ **working-set bursts and intensities**  
→ **record demand through fuzzy memberships**  
→ **observable access events**

Synthetic hidden truth exists only for evaluation.

It must be structurally separate from model-visible data.

---

# 7. Observable Event Contract

A generic access event should contain only information plausibly available in deployment.

Initial fields may include:

- logical time or timestamp;
- record ID;
- record size;
- user ID;
- session ID;
- request type;
- operation type.

A raw session ID should not itself encode reusable semantic information.

Session history may later become a predictive feature.

This event contract should eventually support:

- controlled workload generation;
- trace replay;
- LLM simulator integration;
- real-system telemetry.

---

# 8. Slow Working-Set Learner

Historical access events are transformed into demand windows.

A conceptual demand matrix has:

- rows representing time windows;
- columns representing records;
- values representing demand or normalized demand movement.

The first learner should use a simple factorization-based representation:

\[
D \approx A M
\]

where:

- \(A\) represents working-set activity over time;
- \(M\) represents fuzzy record membership.

The first implementation should prioritize:

- interpretability;
- nonnegative memberships;
- reproducibility;
- controlled recovery evaluation.

The exact factorization algorithm remains an implementation-level decision for its milestone.

---

# 9. Fast Predictor

For each working set \(W_k\), the predictor estimates:

\[
P(W_k \text{ activates within } H)
\]

and:

\[
E[\text{intensity}_k \mid W_k \text{ activates}]
\]

Possible inputs include:

- user identity or representation;
- request type;
- session history;
- recent working-set activity;
- recent record demand;
- operation information;
- optional application metadata later.

The predictor should remain small initially.

Model sophistication must be justified by measured limitations.

---

# 10. Record-Level Forecast

Working-set forecasts are projected into expected record demand using fuzzy memberships.

Conceptually:

\[
\hat n_i(H)
=
\sum_k
P(W_k \text{ activates})
\cdot
E[\text{intensity}_k \mid \text{activation}]
\cdot
M_{k,i}
\]

The exact scaling and normalization will be selected during implementation.

The output contract should remain a record-level near-future demand estimate suitable for deterministic placement.

---

# 11. Placement Objective

For record \(i\), define conceptually:

- \(\hat n_i(H)\): expected reads over horizon \(H\);
- \(L_i^{slow}\): slow-tier latency;
- \(L_i^{fast}\): fast-tier latency;
- \(M_i\): promotion or movement cost;
- \(s_i\): record size.

Expected latency benefit may be approximated as:

\[
B_i
=
\hat n_i(H)
\left(
L_i^{slow}-L_i^{fast}
\right)
-
M_i
\]

The controller selects binary residency variables:

\[
x_i \in \{0,1\}
\]

subject to:

\[
\sum_i s_i x_i \le C_{\text{fast}}
\]

while maximizing expected total benefit.

Because records are indivisible, this resembles 0/1 knapsack.

The initial runtime controller should use a greedy benefit-per-byte approximation.

Exact and oracle modes are primarily diagnostic.

---

# 12. Baselines

All policies must operate on identical frozen traces.

Initial baselines should include:

- random replacement;
- LRU;
- LFU;
- TinyLFU or a representative TinyLFU-style policy;
- a size/cost-aware reactive policy;
- predictive greedy placement;
- predictive exact placement for small cases;
- oracle placement using true future demand.

A size-aware reactive baseline is necessary to separate prediction gains from basic variable-size optimization gains.

---

# 13. Evaluation

## System metrics

- mean latency;
- p50 latency;
- p95 latency;
- p99 latency;
- hit rate;
- slow-tier reads;
- throughput;
- bytes promoted;
- bytes evicted;
- wasted promotions;
- prediction overhead;
- migration overhead;
- CPU overhead.

## Transition metric

Measure cumulative latency immediately after abrupt working-set activation.

This is the period where predictive placement should have the clearest advantage over reactive caching.

## ML metrics

Slow learner:

- factor recovery;
- membership similarity;
- merged and split factors;
- reconstruction error.

Fast predictor:

- activation discrimination;
- false positives;
- false negatives;
- calibration;
- conditional intensity error.

## System-weighted error analysis

Prediction errors should also be ranked by latency regret.

A statistically incorrect prediction that causes no harmful placement is less important than a slightly miscalibrated prediction that causes a costly migration or miss.

---

# 14. Oracle Decomposition

Prism should support component substitution experiments.

Examples:

- learned structure + learned predictor + greedy controller;
- true structure + learned predictor + greedy controller;
- learned structure + true future demand + greedy controller;
- learned structure + learned predictor + exact controller;
- true structure + true future demand + exact controller.

This separates:

- slow-learning error;
- fast-prediction error;
- controller approximation;
- migration/system overhead;
- total opportunity available.

---

# 15. Runtime Architecture

The foreground request path must not block on Python inference.

Conceptual flow:

**Foreground**

request  
→ C++ lookup  
→ RAM or SSD read  
→ response

**Background**

access event  
→ event queue  
→ feature aggregation  
→ Python/PyTorch prediction  
→ forecast snapshot  
→ deterministic C++ controller  
→ promotion and eviction worker

Forecast staleness is acceptable within measured limits.

Synchronous Python calls on each request are not.

---

# 16. Technology Stack

## Python

Used for:

- controlled workload generation;
- trace processing;
- machine learning;
- experiment orchestration;
- metrics;
- analysis.

## PyTorch

Used for:

- the fast predictor;
- later factor-learning implementations where useful;
- training and calibration;
- optional GPU training.

## NumPy

Used for:

- numerical generation;
- matrices;
- factorization prototypes;
- validation.

## pandas and matplotlib

Used only for offline analysis and visualization.

## C++20

Used for:

- real storage runtime;
- RAM ownership;
- SSD access;
- policies;
- placement;
- queues;
- concurrency;
- timing;
- instrumentation.

## CMake

Used for the C++ build.

## pybind11

Used for the initial Python/C++ boundary.

The boundary should use compact batched numerical data rather than per-request Python callbacks.

---

# 17. Storage MVP

The first real tiering engine will use:

**Fast tier:** process RAM  
**Slow tier:** simple local file-backed storage

Records are immutable.

The slow tier remains authoritative.

The RAM tier contains cached copies and is constrained by bytes.

The storage implementation should remain deliberately simple to avoid hidden behavior from databases, external caches, or multiple buffering layers.

---

# 18. Development Progression

1. Controlled workload generator
2. Demand windows and slow working-set learner
3. Fast activation/intensity predictor
4. Simulated predictive placement
5. Rigorous experiments and error analysis
6. Real C++ RAM/SSD tiering engine
7. Python/C++ integration
8. Open-source LLM inference simulator integration
9. Real heterogeneous CPU/GPU/storage deployment
10. Other application domains, including possible market-data workloads

Detailed completion gates are maintained in `docs/milestones.md`.

---

# 19. MVP Definition

The first complete MVP consists of:

- controlled contextual abrupt-burst workloads;
- fuzzy overlapping working-set learning;
- activation probability prediction;
- conditional intensity prediction;
- record-level demand projection;
- deterministic greedy placement;
- strong reactive baselines;
- oracle and exact diagnostic modes;
- real C++ RAM/SSD tiering;
- asynchronous Python/C++ interaction;
- rigorous end-to-end evaluation.

The MVP does not require:

- GPU memory;
- LLM simulator integration;
- real inference deployment;
- adaptive factors;
- learned placement.

---

# 20. Long-Term Direction

After the RAM/SSD MVP, Prism should integrate with an existing open-source LLM inference simulator.

Prism should replace or augment placement behavior rather than simulate:

- transformer execution;
- CUDA kernels;
- token scheduling;
- GPU topology;
- continuous batching.

After simulator validation, Prism may be integrated with a real inference-serving or cache system managing data across:

\[
\text{GPU HBM}
\leftrightarrow
\text{CPU RAM}
\leftrightarrow
\text{NVMe}
\]

Quant or market-data workloads remain possible later generalization experiments.

---

# 21. Final Principle

Prism should use ML to estimate uncertain future demand and conventional systems reasoning to act on that information efficiently.

Complexity must be earned through evidence.

A simple model that produces a measurable real-system benefit is more valuable than a sophisticated architecture whose contribution cannot be isolated.
