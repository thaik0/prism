# Prism — Milestones

## Current status

**Current milestone:** Milestone 6 — Native C++ Two-Tier Storage Engine

**Project stage:** Milestone 6 implemented and verified; policy integration deferred

A milestone is complete only when:

- required behavior is implemented;
- acceptance tests pass;
- representative output has been inspected;
- documentation is updated;
- limitations are recorded;
- no future milestone was silently implemented.

---

# Milestone 0 — Repository Foundation

**Status:** In progress

## Objective

Create the minimal repository structure and instructions required for reliable milestone-based development.

## Deliverables

- `AGENTS.md`
- `README.md`
- `docs/technical_planning.md`
- `docs/decisions.md`
- `docs/milestones.md`
- minimal Python project configuration
- test command documented

## Completion gate

- Repository instructions are present.
- Technical scope and non-goals are documented.
- A clean development environment can install the Python package and run an empty or starter test suite.
- No simulator, ML, cache, or storage functionality is required.

---

# Milestone 1 — Controlled Workload Generator

**Status:** Complete

## Objective

Generate reproducible access traces containing hidden fuzzy working sets, abrupt bursts, contextual signals, and simulator-only ground truth.

## Required capabilities

- configurable variable-sized records;
- sparse fuzzy overlapping memberships;
- configurable users and request types;
- multi-request sessions;
- hidden user/request-type affinities;
- abrupt working-set bursts;
- variable burst intensity and duration;
- true, false, and missing precursors;
- baseline demand;
- background noise;
- explicit seeded randomness;
- separate observable and hidden schemas;
- human-readable demonstration output.

## Explicit non-goals

- machine learning;
- matrix factorization;
- prediction;
- cache policies;
- placement;
- simulated storage latency;
- C++;
- LLM-specific behavior;
- drift;
- adaptive working-set count.

## Completion gate

- Same configuration and seed produce identical output.
- Different seeds normally produce different output.
- Hidden truth cannot appear in observable event schemas.
- Memberships are nonnegative, sparse, and overlapping when configured.
- Events reference valid records, users, sessions, and request types.
- Event time ordering is valid.
- Tests demonstrate true precursors, false precursors, and unannounced bursts using reliable configurations.
- A representative trace can be generated from a documented command.
- Tests and documentation pass review.

---

# Milestone 2 — Demand Windows and Slow Working-Set Learner

**Status:** Complete

## Objective

Transform access traces into demand matrices and learn fuzzy latent working sets from historical demand.

## Required capabilities

- raw per-window, per-record access counts;
- one fixed deterministic NMF baseline;
- fuzzy record memberships;
- factor activation history;
- comparison with planted simulator truth;
- factor matching despite label permutation;
- fuzzy, support, reconstruction, and activation-alignment metrics;
- deterministic four-artifact persistence.

## Explicit non-goals

- fast contextual prediction;
- placement;
- C++;
- adaptive factor count;
- graph or transformer models;
- online production training.

## Completion gate

- Raw demand-window construction is deterministic and tested.
- The representative learner converges and recovers support above analytic chance.
- Hidden truth is used only after fitting for controlled evaluation.
- Factor labels are matched with globally optimal one-to-one assignment.
- Repeated identical runs produce byte-identical artifacts.
- Non-convergence is persisted and reported as a nonzero CLI result.

---

# Milestone 3 — Fast Activation and Intensity Predictor

**Status:** Complete

**Prerequisite status:** Complete. The dedicated longer controlled trace plants
and scientifically validates stochastic context-informed burst intensity while
preserving the Milestone 1 and Milestone 2 contracts.

## Objective

Predict near-future working-set activation probability and conditional intensity from context and recent workload state.

## Required capabilities

- model-visible contextual feature construction;
- one fixed one-window prediction horizon;
- activation targets;
- conditional intensity targets;
- fixed scikit-learn logistic and ridge models;
- train/validation/test splitting that avoids leakage;
- activation evaluation;
- intensity evaluation;
- probability calibration analysis;
- context ablations.

## Explicit non-goals

- learned placement;
- C++;
- real storage;
- sequence transformers;
- multiscale prediction;
- production online training.

## Completion gate

- Predictor beats simple non-contextual prediction baselines on controlled learnable workloads.
- Activation probabilities are evaluated for calibration.
- Intensity errors are analyzed separately.
- Removing context materially affects performance when the workload is configured to contain contextual signal.
- No hidden simulator variables enter model inputs.
- NMF, preprocessing, and supervised models are fit only on training populations.
- All three strict untouched-test scientific gates pass on the dedicated trace.
- Repeated complete runs produce byte-identical four-artifact outputs.

---

# Milestone 4 — Simulated Predictive Placement

**Status:** Complete

## Objective

Convert forecasts into record-level demand estimates and compare predictive placement with reactive cache policies under a controlled cost model.

## Required capabilities

- working-set-to-record demand projection;
- variable-size byte-constrained fast tier;
- expected-latency benefit calculation;
- greedy predictive controller;
- LRU baseline;
- LFU baseline;
- recent-demand size/cost-aware greedy baseline;
- exact controller for small cases;
- one-window oracle greedy and exact controllers;
- frozen-trace policy replay.

## Completion gate

- Every policy receives identical traces and storage parameters.
- Greedy, exact, and oracle modes are distinguishable.
- Metrics include latency, hit rate, bytes moved, and transition-period regret.
- At least one predictable abrupt-burst workload shows an opportunity for predictive improvement.
- Random or uninformative workloads expose prediction overhead rather than artificial gains.

The representative 1,000-window trace passes all five fixed scientific gates.
Predictive greedy reduces test combined cost relative to recent-demand greedy,
LRU, and LFU; it also reduces unique burst-start-window combined cost relative
to recent-demand greedy. One-window oracle greedy demonstrates opportunity, and
one-window oracle exact is no worse than oracle greedy. These are controlled
simulated-cost results, not measured storage latency.

---

# Milestone 5 — Rigorous Evaluation and Error Analysis

**Status:** Complete

## Objective

Determine when Prism wins, when it loses, and which components cause remaining regret.

## Required capabilities

- one frozen 12-variant manifest covering capacity, promotion cost, noise,
  burst duration, and context reliability;
- three repeated fixed-seed trials per variant;
- Training-Popularity Static and Validation-Final Frozen controls;
- recent-state, activation/intensity, and residual-only forecast ablations;
- dynamic-action and hidden-truth-only pre-transition diagnostics;
- paired seed comparisons and descriptive sample uncertainty;
- behavioral and fixed-trajectory migration-cost analysis;
- structure, predictor, projection, placement, and oracle-regret diagnostics;
- deterministic reports, resume validation, and byte-identical full sweeps.

## Completion gate

- Results are reproducible from checked-in configurations.
- Conclusions distinguish predictive opportunity from predictor quality.
- Important failures can be traced to slow learner, fast predictor, controller, migration cost, or lack of workload opportunity.
- No result is reported from one favorable seed alone.

The committed manifest fixes 12 variants and seeds `1729`, `2718`, and `31415`.
All 36 engineering-valid runs completed twice and the two experiment roots were
byte-identical. The sweep records earlier scientific gates as outcomes rather
than filtering runs.

The principal conclusion is unfavorable to the dynamic-value hypothesis:
Predictive Greedy was behaviorally identical to Validation-Final Frozen in 35
of 36 runs and its total cost was identical to Recent-State-Only in 34 of 36.
It beat Training-Popularity Static in a majority of seeds for 9 variants, but
that advantage usually came from a better validation-developed static target.
Only `promotion_0__seed_31415` changed test targets; it reduced cost by `243`
relative to frozen while slightly reducing mean pre-transition realized-demand
coverage. Oracle regret remained positive in every run, so predictive opportunity
still exists. Full results and limitations are in `docs/milestone5_results.md`.

---

# Milestone 5.5 — Predictive Actionability Diagnosis

**Status:** Complete

## Objective

Determine whether the frozen predictor-to-controller architecture becomes
dynamically actionable under sparse activation regimes and matched cumulative
horizons, or whether Prism's supported thesis must be narrowed.

## Completion evidence

- A frozen 3-regime by 3-horizon by 3-seed manifest completed all 27 runs.
- Sparse and very-sparse regimes showed strictly fewer starts, less simultaneous
  activity, and longer dormant intervals in three-seed aggregate.
- Common eligible windows, cumulative labels/baselines, rolling placement,
  factor-to-record decomposition, controller rejection, matched oracle, and
  promotion repayment were persisted deterministically.
- Two complete roots were byte-identical and resume reused all 27 hash-validated
  runs.
- All four candidate cells failed Gates A--D without post-result tuning.

The final status is `stable_cost_aware_tiering_reframe`. The current evidence
supports learned latent-demand structure for stable, cost-aware tiering, but not
useful dynamic predictive tiering from the current fast forecast. This result
does not begin a predictor search or real-storage implementation.

---

# Milestone 6 — C++ RAM/SSD Tiering Engine

**Status:** Complete

## Objective

Build a real two-tier record engine using process RAM and a simple file-backed local SSD store.

## Required capabilities

- immutable variable-sized records;
- authoritative slow-tier copies;
- byte-constrained RAM cache;
- metadata index;
- real file reads;
- promotion and eviction;
- policy-agnostic deterministic exact-target interface;
- deterministic logical instrumentation;
- structured corruption and I/O errors;
- deterministic inspection and JSONL replay;
- C++ unit tests;
- CMake build.

## Explicit non-goals

- PyTorch integration;
- GPU memory;
- distributed storage;
- transactions;
- dirty writeback;
- RocksDB or database integration;
- custom async-I/O frameworks.

## Completion gate

- Capacity invariants hold for variable-sized records.
- Eviction never loses authoritative data.
- Reads return correct bytes.
- Logical read and movement metrics are captured deterministically.
- Failed target transitions preserve exact prior residency.
- Every slow-tier and migration read verifies CRC32.
- Builder, inspection, and replay outputs are deterministic.
- The engine replays a representative trace independently of Python ML.
- Debug, AddressSanitizer, UndefinedBehaviorSanitizer, and Python regressions pass.

Milestone 6 deliberately contains no placement policy. The attached milestone
brief superseded the older baseline-policy wording: policies supply exact targets
in a later integration milestone. No wall-clock performance claim is made.

---

# Milestone 7 — Python/C++ Integration

**Status:** Not started

## Objective

Connect Python forecasting to the C++ tiering engine without placing Python inference on the foreground request path.

## Required capabilities

- thin pybind11 interface;
- compact batched forecast updates;
- asynchronous event collection;
- forecast snapshots;
- C++ placement worker;
- end-to-end metrics;
- integration tests.

## Completion gate

- Foreground reads do not call Python synchronously.
- Forecast updates cross the boundary in batches.
- Ownership and lifetimes across the boundary are documented and tested.
- End-to-end runs combine Python prediction with real C++ RAM/SSD placement.
- Prediction and migration overhead are measured.

---

# Milestone 8 — Open-Source LLM Simulator Integration

**Status:** Not started

## Objective

Evaluate Prism inside an independently developed LLM inference simulation environment.

## Required capabilities

- adapter from simulator events to Prism’s generic event interface;
- mapping from simulator cache objects to Prism records;
- predictive and reactive placement comparisons;
- simulator-native latency and throughput metrics;
- no reconstruction of transformer or GPU execution simulation.

## Completion gate

- Prism operates without synthetic hidden labels.
- Existing simulator policies and Prism receive equivalent workloads.
- Integration does not require rewriting Prism’s core predictor/controller contract.
- Results show whether the controlled-workload conclusions transfer.

---

# Milestone 9 — Real Heterogeneous Deployment

**Status:** Not started

## Objective

Apply Prism to real inference state across GPU memory, CPU memory, and possibly local SSD.

## Potential capabilities

- real GPU-resident cache objects;
- CPU/GPU transfer;
- integration with an inference serving or cache framework;
- real TTFT, throughput, and memory-pressure measurements;
- concurrency and contention analysis.

## Completion gate

This milestone will be defined only after the LLM simulator integration exposes a specific and credible deployment target.

---

# Deferred Extensions

These are not implementation milestones until evidence justifies them:

- working-set drift;
- persistent regime memory;
- adaptive factor count;
- factor splitting and merging;
- multiscale prediction horizons;
- temporal ordering inside working sets;
- tail-aware or risk-sensitive objectives;
- learned placement controller;
- mutable records;
- additional storage tiers;
- quant or market-data workload integrations.
