# Milestone 1 Controlled Workload Generator

## Purpose and boundary

Milestone 1 generates synthetic access histories with planted fuzzy working sets,
contextual signals, abrupt bursts, baseline demand, and noise. It is temporary
research infrastructure for later Prism milestones, not a storage simulator.

The implementation contains no learning, prediction, feature pipeline, cache,
placement, latency model, storage tier, database, C++, or LLM-specific behavior.

## Fixed-window causal model

Time is a sequence of integer windows from `0` through `num_windows - 1`. There
are no timestamps or continuous-time scheduling. Globally contiguous
`event_index` values establish total event order.

At initialization the generator uses its locally owned `random.Random(seed)` to
create:

- fixed variable record sizes;
- independently sampled sparse working-set supports and positive normalized
  fuzzy memberships;
- positive normalized user and request-type affinities over working sets;
- positive normalized user preferences over request types; and
- one positive normalized baseline-popularity vector over records.

For each window the generator then performs these steps in order:

1. Expire bursts whose end is at or before the current window.
2. Run activation trials for inactive working sets using only the prior window's
   precursor score. Successful bursts are active immediately.
3. Generate window-local sessions, requests, and accesses.
4. Aggregate each request's context once, regardless of its access count, to
   produce scores that can be used only in the next window.

Session and request IDs are deterministic opaque integers. They have no causal or
semantic effect and are never inputs to activation or demand generation.

## One-window precursor delay

For request context `(user_id, request_type)` and working set `k`:

```text
request_contribution[k] =
    0.5 * user_affinity[user_id][k]
    + 0.5 * request_type_affinity[request_type][k]
```

Contributions are averaged across requests in the current window, then scaled:

```text
precursor_score[k] =
    min(1, num_working_sets * mean_request_contribution[k])
```

The score from window `t` is used only for an eligible activation trial in window
`t + 1`. Every window-0 trial uses a score of exactly zero.

For an eligible inactive working set:

```text
context_probability = precursor_probability_scale * previous_precursor_score

activation_probability =
    1 - (1 - spontaneous_activation_probability)
        * (1 - context_probability)
```

The formula treats spontaneous and contextual activation as independent
opportunities. A nonzero precursor can fail, and a zero precursor can still be
followed by a spontaneous burst. Active sets do not receive activation trials and
can retrigger only after expiration.

## Sessions, requests, and accesses

Every session belongs to one window and selects one user uniformly. Requests in a
session share that user; each request type is sampled through that user's hidden
request-type preference vector. Every request samples a bounded positive access
count.

For each access, the source weights are:

```text
active working set k: burst_access_weight * burst_intensity[k]
baseline:             baseline_access_weight
noise:                noise_access_weight
```

An active working-set source samples only from its sparse membership distribution.
Baseline demand samples from the hidden baseline-popularity vector. Noise samples
uniformly from all records. Once a burst is active, current request context does
not influence its demand share; only burst intensity does.

Operation types are independently sampled categorical labels. A label such as
`write` does not mutate records and has no storage, consistency, or future-demand
semantics.

Per-event source provenance is intentionally not retained. Hidden ground truth
contains aggregate baseline, noise, and per-working-set source counts per window.

## Observable schema

Each line of `observable_events.jsonl` has exactly these fields:

```json
{
  "event_index": 0,
  "operation_type": "read",
  "record_id": 0,
  "record_size_bytes": 4096,
  "request_id": 0,
  "request_type": "interactive",
  "session_id": 0,
  "user_id": 0,
  "window_id": 0
}
```

True working-set labels are never observable. Events never include memberships,
affinities, precursor scores, activation probabilities or outcomes, burst state,
source labels, popularity weights, or random draws.

## Hidden-ground-truth schema

`hidden_ground_truth.json` is simulator-only and contains:

- schema version and seed;
- record sizes in record-ID order;
- sparse memberships in working-set-ID and record-ID order;
- user affinities, request-type affinities, and user preferences in deterministic
  ID/configuration order;
- baseline record popularity in record-ID order;
- precursor scores produced by each window;
- all eligible activation trials and the exact probabilities they used;
- all burst intervals, durations, and intensities; and
- per-window aggregate source counts, including per-working-set counts.

An absent record in a sparse working-set membership has membership zero. The
hidden file is never safe as predictor input. There is no combined
observable-plus-hidden event table.

`summary.json` contains human-readable run totals. `config.json` is the validated,
fully resolved input, including the seed. Treat configuration values that describe
the hidden generation process as experiment metadata rather than model features.

## Configuration

Every field is required. Unknown fields and likely typos are rejected.

| Field | Meaning |
|---|---|
| `seed` | Integer seed for the run-owned RNG. |
| `num_windows` | Logical window count; at least 2. |
| `num_records` | Positive record count. |
| `num_working_sets` | Positive fixed hidden working-set count. |
| `num_users` | Positive user count. |
| `request_types` | Nonempty unique nonempty strings. |
| `operation_type_probabilities` | Nonempty label-to-probability mapping summing to 1. |
| `record_size_min_bytes`, `record_size_max_bytes` | Inclusive positive integer size bounds. |
| `working_set_support_min`, `working_set_support_max` | Inclusive positive support bounds; maximum cannot exceed record count. |
| `min_sessions_per_window`, `max_sessions_per_window` | Inclusive positive session-count bounds. |
| `min_requests_per_session`, `max_requests_per_session` | Inclusive positive request-count bounds. |
| `min_accesses_per_request`, `max_accesses_per_request` | Inclusive positive access-count bounds. |
| `spontaneous_activation_probability` | Independent activation opportunity in `[0, 1]`. |
| `precursor_probability_scale` | Context scaling factor in `[0, 1]`. |
| `burst_duration_min_windows`, `burst_duration_max_windows` | Inclusive positive integer duration bounds. |
| `burst_intensity_min`, `burst_intensity_max` | Inclusive positive intensity bounds. |
| `burst_access_weight` | Nonnegative active-burst source multiplier. |
| `baseline_access_weight` | Nonnegative baseline source weight. |
| `noise_access_weight` | Nonnegative noise source weight. |

Validation rejects booleans used as integers, non-finite numbers, invalid bound
ordering, out-of-range probabilities, duplicate/empty categories, negative
weights, operation probabilities outside a `1e-12` absolute sum tolerance, and a
configuration where baseline and noise weights are both zero. User-supplied
operation probabilities are never silently normalized. The JSON loader also
rejects duplicate object keys.

## API and CLI

The public API is:

```python
from prism.workload import WorkloadConfig, generate_workload, persist_workload

config = WorkloadConfig.from_json("configs/milestone1_representative.json")
result = generate_workload(config)
persist_workload(result, "/tmp/prism_milestone1_run")
```

From an uninstalled checkout, run the CLI with the src directory on the module
path:

```bash
PYTHONPATH=src python3 -m prism.workload.cli \
  --config configs/milestone1_representative.json \
  --output-dir /tmp/prism_milestone1_run
```

The output directory may be absent or empty. A nonempty directory is rejected and
left unchanged. The generator writes exactly the four documented artifacts.

## Reproducibility

Every random decision uses the generator-owned RNG. The implementation does not
seed or call module-global randomness and stores no UUID, timestamp, temporary
path, or machine-specific value. IDs, collection order, event order, JSON key
order, formatting, and final newlines are deterministic.

Identical validated configuration and seed produce equal in-memory results and
byte-identical artifacts. Policies in later milestones can therefore consume the
same frozen `observable_events.jsonl` trace, but no replay or policy exists in
this milestone.

## Representative run

The committed configuration is `configs/milestone1_representative.json` with seed
`1729`. Regenerate it with the CLI command above. Its tested stable summary is:

```text
windows=40, records=64, sessions=157, requests=466, events=2604
bursts=34, baseline=240, noise=84, working_set=2280
records_in_multiple_working_sets=17
```

The generated artifacts are intentionally not committed.

## Current limitations

- Working-set count and all distributions are fixed for a run; there is no drift.
- Overlap emerges only from independently sampled supports; there is no explicit
  overlap-control algorithm.
- IDs have no reusable semantics and request ordering inside a window is purely
  generator order.
- Records are synthetic identities and sizes, not stored contents.
- Operation labels have no mutation semantics.
- The generator offers no learning labels, model features, prediction,
  forecasting evaluation, caching, placement, storage, or latency behavior.
