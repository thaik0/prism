# Milestone 2 Demand Windows and Structural Recovery

## Purpose and boundary

Milestone 2 asks whether one standard, interpretable factorization can recover
nontrivial fuzzy working-set structure from one frozen Milestone 1 access trace.
It is a slow, trace-local scientific baseline. It does not forecast future
activation or intensity and does not choose storage actions.

The learner receives observable access history plus the deliberately supplied
factor count `K`. Hidden simulator truth is opened only after fitting for
controlled evaluation.

## Raw demand matrix

For every configured logical window and record:

```text
X[window_id, record_id] = observable access-event count
```

`X` has shape `(num_windows, num_records)`. Rows and columns follow ascending
contiguous schema-v1 IDs, including windows or records whose count is zero. Each
event contributes exactly one integer count. The persisted matrix retains an
integer dtype; only the array passed to NMF is converted to `float64`.

Demand construction reads only:

```text
config.json
observable_events.jsonl
summary.json
```

It ignores record size, bytes, user, session, request, request type, and operation
type. It performs no centering, scaling, TF-IDF, logarithm, binary conversion,
smoothing, baseline subtraction, or window normalization. Those transformations
would answer a different question and are deferred until raw-count behavior has
been measured.

Context is excluded because contextual activation prediction belongs to
Milestone 3. This milestone isolates historical record co-demand structure.

## Factorization

The model is:

```text
X ≈ A @ M
```

- `X` has shape `(windows, records)` and stores observed counts.
- `A` has shape `(windows, factors)` and stores learned historical activation.
- `M` has shape `(factors, records)` and stores learned fuzzy record membership.

The baseline uses scikit-learn instead of a custom optimizer because a maintained,
well-tested implementation is sufficient for this controlled question. The
committed configuration supplies `K=4`, equal to the source
`num_working_sets`. Adaptive `K`, model selection, and factor-count discovery are
explicitly deferred.

Exactly one model is fit:

```python
NMF(
    n_components=config.n_components,
    init="nndsvda",
    solver="cd",
    beta_loss="frobenius",
    max_iter=config.max_iter,
    tol=config.tolerance,
    random_state=config.fit_seed,
)
```

The representative configuration uses seed `2718`, `max_iter=1000`, and
`tolerance=1e-4`. There are no restarts, regularization searches, preprocessing
modes, validation selection, or hidden-metric tuning. A convergence warning or
reaching the iteration limit is recorded as non-convergence. The CLI writes all
diagnostics and then exits nonzero.

## Factor normalization

NMF permits reciprocal scaling between each activation column and membership
row. For factor `k`, Prism computes:

```text
scale = sum(M_raw[k, :])
M[k, :] = M_raw[k, :] / scale
A[:, k] = A_raw[:, k] * scale
```

Each nonzero membership row therefore sums to one while `A @ M` remains equal to
the raw reconstruction within floating-point tolerance. A zero-sum factor is not
divided; it remains zero and produces a deterministic warning and zero membership
similarity.

## Matching planted factors

NMF factor labels are arbitrary. Evaluation expands planted sparse memberships
to the learned record order, calculates cosine similarity for every learned and
planted pair, and calls `scipy.optimize.linear_sum_assignment(..., maximize=True)`
for a globally optimal one-to-one assignment. Matching occurs only in the
recovery report. Learned arrays never contain planted IDs.

## Recovery metrics

Fuzzy membership recovery reports each matched cosine similarity and its count,
minimum, maximum, mean, and median.

Sparse support recovery uses the planted support size `s`. The matched learned
factor's `s` largest weights form the predicted support; equal weights are broken
by ascending record ID. Each factor reports overlap, precision, recall, F1,
Jaccard similarity, and the analytic random-support expectation:

```text
expected random overlap fraction = s / num_records
```

The representative scientific gate is the strict comparison:

```text
mean learned support recall > mean analytic random-support expectation
```

Reconstruction quality includes absolute Frobenius error, demand-matrix norm,
and their ratio. The ratio is explicitly `null` for an all-zero matrix. Low
reconstruction error alone is not evidence of planted-factor recovery.

Historical activation alignment compares each matched learned `A` column with
the simulator's per-window source-access counts for that planted working set.
Cosine similarities are descriptive and not part of the hard gate. Alignment is
`null` with a warning if a planted factor generated no working-set traffic.

## Python API

```python
from prism.structure import (
    StructureLearnerConfig,
    build_demand_matrix,
    evaluate_recovery,
    fit_structure,
)

demand = build_demand_matrix("/tmp/prism_milestone1_run")
config = StructureLearnerConfig.from_json(
    "configs/milestone2_representative.json"
)
learned = fit_structure(demand, config)
evaluation = evaluate_recovery(
    "/tmp/prism_milestone1_run", demand, learned, config
)
```

`build_demand_matrix` and `fit_structure` have no hidden-truth input. The final
call is the explicit simulator-only evaluation boundary.

## CLI and output contract

```bash
PYTHONPATH=src python3 -m prism.structure.cli \
  --run-dir /tmp/prism_milestone1_run \
  --config configs/milestone2_representative.json \
  --output-dir /tmp/prism_milestone2_run
```

The source directory is read-only. The destination must be absent or empty. The
command writes exactly:

```text
learner_config.json
demand_matrix.npz
learned_structure.npz
recovery_report.json
```

`learner_config.json` contains the resolved configurable and fixed settings,
source hashes and dimensions, factor count, and Python/NumPy/SciPy/scikit-learn
versions. `demand_matrix.npz` contains only `X`, `window_ids`, and `record_ids`.
`learned_structure.npz` contains only normalized activation and membership
matrices plus factor, window, and record IDs. `recovery_report.json` contains the
controlled hidden-truth comparison. No estimator or reconstructed matrix is
serialized.

## Reproducibility

The fit has explicit input order, `float64` conversion, and `random_state`. JSON
uses sorted keys, stable indentation, and one final newline. NumPy NPZ output was
empirically checked for stable bytes in the verification environment. No output
contains a timestamp, UUID, absolute source path, or machine-specific location.

Given identical four source artifacts, learner configuration, dependency
environment, and seed, all four output files are byte-identical. Dependency
versions are persisted because numerical identity is conditioned on the numerical
environment.

## Current limitations and explicit non-goals

This is one raw-count NMF fit on one trace. It does not discover `K`, combine
traces, retrain online, run hyperparameter searches, or compare factorization
families. It has no context features, forecasting split, future activation or
intensity predictor, record-demand projection, placement, caching, latency,
storage tiers, C++, PyTorch, plots, notebooks, or dashboards.

If the raw baseline fails on another trace, the correct next step is to inspect
its diagnostics and report the result before proposing additional complexity.
