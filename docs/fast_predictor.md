# Milestone 3 Fast Activation and Conditional-Intensity Predictor

## Purpose and boundary

Milestone 3 predicts, for every learned factor and observable feature window
`t`, the probability that a new burst begins in `t + 1` and the sampled burst
intensity conditional on that start. It does not project forecasts to records or
choose placement, caching, storage, or controller actions.

The controlled causal task is:

```text
observable context and recent learned-factor demand in t
→ new-burst probability in t + 1
→ conditional sampled burst intensity
```

## Chronological experiment and frozen structure

For a trace with `T` windows, the predictor uses:

```text
train_end      = floor(0.60 * T)
validation_end = floor(0.80 * T)
```

An example uses feature window `t` and target window `t + 1`; three demand lags
make `t = 2` the first usable feature window. Examples are assigned by target
window without shuffling. On the dedicated 1,000-window trace, NMF sees only raw
observable demand `X[0:600, :]`, training has 2,388 factor-window examples, and
validation and test each have 800.

The learner is exactly the accepted Milestone 2 NMF. Its normalized membership
matrix is frozen after training. All-window learned-factor demand is then the
observable projection:

```text
D = X @ M.T
```

Validation and test demand never refit or update `M`.

## Controlled matching and labels

NMF factor IDs are arbitrary. Prism compares the training-fitted memberships to
planted memberships with the accepted cosine similarity and Hungarian assignment,
then freezes the learned-to-planted mapping. The mapping appears only in the
controlled evaluation report.

The matched simulator factor supplies the next-window burst-start label, sampled
intensity for positive starts, hidden eligibility diagnostic, and realized
working-set source count. None of these values enters a model feature. Simulator
truth is appropriate for controlled scientific evaluation, but it is not a
deployable labeling mechanism. Real traces will require observable-derived target
construction before this training procedure can be deployed.

Every usable feature-window/factor pair is a prediction and training candidate.
Hidden eligibility never filters examples or affects preprocessing, fitting,
weights, or prediction; it defines only a secondary evaluation population.

## Observable request reconstruction

Unique requests are reconstructed from access events and counted once regardless
of access count. Request metadata must agree on window, session, user, and request
type. Each window records unique session and request counts, access-event count,
user request fractions, and request-type fractions in deterministic order.
Absent categories are zero. Raw request and session IDs are excluded because they
are opaque identities rather than reusable semantics. Operation type is also
excluded from this initial model.

## Fixed feature schemas and models

The recent-state schema contains the current and two lagged factor demands,
one-window demand delta, three-window demand mean, current session/request/access
counts, and learned-factor one-hot indicators.

The context-plus-state schema adds factor-specific interaction blocks. For an
example for factor `k`, only factor `k`'s user-fraction and request-type-fraction
blocks contain the current window's values. This permits one shared linear model
to learn different context relationships per factor without adding user/type
pairs, polynomial expansion, session embeddings, all-factor demand vectors, or
hidden state.

Continuous columns are standardized with scikit-learn `StandardScaler`; factor
one-hot columns are unchanged. Each scaler is fit only on its training population.
The intensity scaler sees only positive training activations. Zero-variance
columns remain present and receive the scaler's safe unit scale.

Activation compares each factor's raw training activation rate with one shared
L2 logistic regression over recent state and one over context plus state. Both
logistic fits use `C=1.0`, `lbfgs`, no class weighting, no resampling, the
configured seed/tolerance/iteration limit, and no threshold or hyperparameter
selection. Context plus state is the primary predictor regardless of gate result.

Conditional intensity compares each factor's positive-training mean with one
shared context-plus-state ridge regression using configured alpha. A missing
factor mean falls back explicitly to the global positive-training mean. Ridge
uses sampled latent intensity directly, with no target transform or clipping.
Out-of-bound predictions are reported as warnings.

## Evaluation and scientific gates

Activation metrics are reported on validation and untouched test data for all
factor-window examples and for the hidden-eligible diagnostic subset. Pooled and
per-factor outputs include Brier score, binary log loss, average precision, and
AUROC where defined. Fixed equal-width calibration tables cover `[0, 1]`, with
probability `1.0` in the final bin.

Intensity metrics use positive burst starts only and report MAE, RMSE, latent
intensity correlation, and per-factor errors. Correlation with realized
next-window working-set access count is a secondary diagnostic because the two
quantities have different units.

The untouched test gates are strict:

1. context-plus-state all-example Brier score is below the per-factor constant;
2. context-plus-state eligible-subset Brier score is below recent demand only;
3. context-plus-state ridge RMSE is below the per-factor conditional mean.

Gate failure is reported without test-driven tuning, resampling, feature changes,
or model replacement.

## CLI and artifact contract

```bash
PYTHONPATH=src python3 -m prism.predictor.cli \
  --run-dir /tmp/prism_m3_predictor_source \
  --structure-config configs/milestone2_representative.json \
  --config configs/milestone3_predictor.json \
  --output-dir /tmp/prism_milestone3_run
```

The empty destination receives exactly:

```text
predictor_config.json
predictor_bundle.npz
predictions.npz
evaluation_report.json
```

`predictor_bundle.npz` contains only frozen learned memberships, identifiers,
feature schemas, train-only scaler state, raw linear coefficients/intercepts, and
the two baseline states. `predictions.npz` contains identifiers, split codes, and
the five predictions for every usable example. Neither contains labels,
eligibility, planted IDs, matching, precursor scores, burst state, intensity
truth, or source counts. Estimators are not pickled. The evaluation report is the
explicit simulator-only boundary and contains matching, target counts, recovery
diagnostics, metrics, gates, and warnings.

## Leakage protection and reproducibility

Tests establish training-only NMF and scaler populations, frozen membership and
matching, full example inclusion independent of eligibility, observable-only
features, held-out-label independence of fitted state, deployable array
allowlists, source preservation, and prediction reconstruction from persisted
parameters. Repeated complete CLI runs are byte-identical in the verified
dependency environment.

Reproducibility depends on source bytes, both committed configurations, dependency
versions, and explicit seeds. Artifacts contain no timestamp, UUID, absolute path,
or machine-specific location.

## Current limitations and non-goals

This is one fixed one-window model family on one dedicated controlled trace. It
does not discover factor count, tune models, use operation/session history,
retrain online, handle drift, project record demand, implement placement or cache
policies, simulate storage, add C++/PyTorch/neural models, or begin Milestone 4.
