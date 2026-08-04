# Prism — Technical Planning Document

## Document status

**Status:** Initial architecture approved  
**Current implementation phase:** Milestone 6 complete — Native C++ two-tier storage substrate

**Last major planning revision:** August 2026

This document is Prism’s architectural source of truth.

It describes the intended system, research thesis, component boundaries, evaluation philosophy, MVP, and long-term progression.

It is not an instruction to implement every component immediately.

Implementation must follow `docs/milestones.md`, and workers must implement only the active milestone.

Settled architectural decisions are summarized in `docs/decisions.md`.

---

# 1. Project Thesis

Prism is a stable, cost-aware storage-tiering research system that learns
recurring latent structure in data demand and feeds demand estimates to a
deterministic placement controller. The current evidence supports the learned
structure and validation-developed stable placement, but not useful dynamic
placement caused by the current activation/intensity forecast.

The original predictive research question remains open rather than demonstrated:

**Can a storage system predict context-triggered changes in future working-set demand early enough to prepare storage proactively, while keeping prediction and movement overhead below the resulting latency benefit?**

Future predictive work must first explain how forecast signal survives
factor-to-record projection, stable ranking, movement cost, and capacity
competition. It is not part of the completed Milestone 5.5 implementation.

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

## C++17

Currently used for:

- real storage runtime;
- RAM ownership;
- SSD access;
- explicit storage operations;
- deterministic logical instrumentation.

Milestone 7 keeps policies and placement decisions in Python. Queues,
concurrency, and timing remain future work.

## CMake

Used for the C++ build.

## pybind11

Milestone 7 uses pybind11 for a private synchronous storage boundary. Python
issues explicit per-operation storage calls; no callbacks, forecast matrices,
or policy state cross into C++. Compact batching and asynchronous interaction
remain deferred until a separately scoped concurrency milestone.

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
6. Predictive actionability diagnosis and thesis reframe (Milestone 5.5)
7. Real C++ RAM/file-backed tiering engine (Milestone 6 complete)
8. Synchronous Python/C++ execution parity (Milestone 7 complete)
9. Open-source LLM inference simulator integration
10. Real heterogeneous CPU/GPU/storage deployment
11. Other application domains, including possible market-data workloads

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
- verified synchronous Python/C++ execution parity;
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

---

# 22. Milestone 4 Controlled Simulation Instantiation

The completed controlled simulator instantiates the record forecast as one
training-only nonnegative calibration per learned factor plus one fixed
nonnegative residual baseline per record. It consumes frozen Milestone 3
probabilities and conditional-intensity predictions; it does not refit structure
or supervised models.

The fast tier is simulated as an exact set of indivisible record IDs constrained
by bytes. LRU and LFU react after a slow miss. Recent-demand, predictive, and the
two one-window oracle policies replace their complete target residency at each
window boundary. Greedy and exact placement use the same expected access-savings
minus promotion-cost objective.

All policies begin validation empty, retain independent state into test, and
replay identical observable events and costs. Hidden burst intervals are opened
only after replay for controlled transition analysis. The four deterministic
Milestone 4 artifacts contain no hidden truth in deployable projection or policy
trace arrays.

This milestone establishes controlled simulated evidence only. It does not
implement or measure the real tier engine described in later sections.

---

# 23. Milestone 5 Evaluation Instantiation

The completed evaluation freezes one manifest of 12 one-factor-at-a-time
variants and three fixed seeds. Every engineering-valid run executes the same
eleven policies: the six Milestone 4 policies, Training-Popularity Static,
Validation-Final Frozen, and three mechanical forecast ablations. Ablations
reuse training-fitted calibration coefficients and never refit. Hidden burst
truth and oracle demand enter diagnostics only.

The sequential harness preserves fully resolved trace-specific storage settings,
stage artifacts, hashes, explicit run status, scientific-gate outcomes, paired
seed differences, dynamic-action traces, pre-transition coverage, and descriptive
oracle-regret terms. Engineering failures are excluded from aggregation;
scientific failures are retained. Three seeds provide sample summaries, not
confidence intervals or significance tests.

The full sweep shows that a low-migration result is not evidence of successful
anticipation. Predictive Greedy remained frozen through test in 35 of 36 runs
and the fast-predictor contribution was not supported by the deterministic
hypothesis rule. It often beat training-only popularity, demonstrating value in
the validation-developed learned-factor placement, while positive regret to the
one-window oracle in every run shows substantial unrealized opportunity. This
narrows later work toward forecast-to-placement responsiveness; it does not
authorize new models or controller objectives inside Milestone 5.

---

# 24. Milestone 5.5 Actionability Diagnosis and Thesis Reframe

Milestone 5.5 freezes three activation regimes, cumulative horizons `1`, `2`,
and `4`, three seeds, one common horizon-safe population, and four eligible
sparse multi-window candidate cells. It retains the accepted models and
controller while adding per-working-set cooldown, cumulative targets,
factor-to-record component accounting, rank turnover, rejection margins,
matched-horizon oracle agreement, and proactive-promotion repayment.

All 27 runs completed reproducibly and the regimes separated as intended, but
none of the four candidate cells changed a test target or made a promotion.
Moving factor forecasts were diluted by a record residual baseline contributing
roughly 87--90% of projected magnitude; record candidate sets consequently
remained almost fixed, while promotion cost and capacity rejected remaining
nonresident opportunities. Oracle disagreement remained material, so the result
does not imply that ideal future knowledge lacks value.

The precommitted decision is `stable_cost_aware_tiering_reframe`. Prism's
supported near-term thesis is learned latent-demand structure feeding a
deterministic, auditable, stable cost-aware placement controller. Dynamic
activation/intensity forecasts remain a measured but presently unactionable
research component. Package names and implementation history remain unchanged,
and this result does not authorize post-result predictor tuning. The subsequently
authorized Milestone 6 adds the policy-agnostic real storage substrate without
changing that scientific conclusion.

---

# 25. Milestone 6 Native Storage Instantiation

Milestone 6 implements a standalone C++17 immutable record engine. A deterministic
builder concatenates payloads into `store.data` and writes a FlatBuffers version-1
`PRSM` index containing checked 64-bit offsets and lengths plus zlib CRC32 values.
Opening validates the full metadata and exact file length without eagerly reading
all payloads.

The authoritative file remains read-only. Nonresident client reads and every
promotion use `pread` and verify the complete payload checksum. The fast tier
owns complete byte buffers under strict capacity. Individual promotion never
evicts, slow reads never promote, and eviction never writes back. Exact-target
transitions stage and verify all incoming data and construct the complete next
state before a no-throw commit swap, so precommit failures preserve residency.

The engine is policy-agnostic. Native inspection and JSONL replay expose stable
errors, logical counters, exact snapshots, corruption handling, and deterministic
artifacts. The milestone adds no predictor/Python integration, policy ranking,
concurrency, asynchronous I/O, mutation, or performance claim. Ordinary file I/O
remains subject to the operating-system page cache.

---

# 26. Milestone 7 Python/C++ Integration Instantiation

Milestone 7 links a private pybind11 module directly to the Milestone 6 storage
library. Python supplies immutable deterministic payload bytes to the shared C++
builder and issues explicit reads, promotions, evictions, exact-target calls,
snapshots, and metadata/statistics queries. The public Python package translates
one stable structured error type and returns detached immutable values.

An independent Python ledger replays native storage semantics and counters. Four
accepted policy paths—Training-Popularity Static, Predictive Greedy, LRU, and
LFU—retain their existing Python controller and tie-breaking behavior. Every
native operation is compared immediately, and divergence invalidates only the
affected policy without resetting native state.

The seed-1729 accepted representative run and forced fixture both pass with zero
mismatches, invalidations, unexpected exceptions, or capacity violations; repeat
output roots are byte-identical. This certifies synchronous integration
semantics only. It adds no timing gate, concurrency, GIL release, background
migration, C++ inference, native policy, or new predictive-actionability claim.
