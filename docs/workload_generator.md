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

## Scientific workload validation

Engineering correctness establishes that a trace follows the configured schemas,
probability formulas, causal delay, and reproducibility contract. Scientific
workload validation is a separate question: does a particular persisted trace
actually exhibit the learnable but imperfect signal intended for later work?

Validate an already-generated run with:

```bash
PYTHONPATH=src python3 -m prism.workload.validate \
  --run-dir /tmp/prism_milestone1_run
```

Add `--require-demonstrations` to require all three representative cases. The
callable API is:

```python
from prism.workload.validate import validate_workload_run, write_validation_report

result = validate_workload_run("/tmp/prism_milestone1_run")
write_validation_report(
    result,
    "/tmp/prism_milestone1_run/workload_validation.json",
)
```

### Derived report and source preservation

The validator reads, rather than regenerates, the original four artifacts. It
deterministically creates or replaces only `workload_validation.json`. The report
includes SHA-256 hashes of all four sources, stable ordering, and a final newline.
It contains no timestamp, random value, absolute input path, or machine-specific
field. Revalidating unchanged sources produces a byte-identical report. The
generator's four-file contract is unchanged.

Malformed artifacts, invalid references, inconsistent counts, non-normalized
vectors, invalid burst/trial state, probability-formula errors, or inconsistent
request/session relationships are structural failures and make the CLI exit
nonzero. Scientifically interesting but structurally valid conditions are emitted
as factual warnings.

### Precursor definition and demonstration gate

Thresholds are calculated independently per working set from eligible activation
trials using:

```python
statistics.quantiles(scores, n=4, method="inclusive")
```

- `score >= Q3` is a clear precursor.
- `score <= Q1` is no clear precursor.
- Scores between the thresholds are intermediate.
- Fewer than two trials is insufficient; `Q1 == Q3` is degenerate.

Insufficient and degenerate sets are reported but excluded, so a trial is never in
both boundary groups. The representative gate requires positive counts for clear
precursor followed by a burst, clear precursor followed by no burst, and no clear
precursor followed by a burst.

Without the flag, an absent category is a warning rather than a structural
failure. With the flag, it makes the CLI exit 1 after writing and printing the
report. The committed seed `1729` needs no adjustment: its counts are `11`, `7`,
and `3`. Working set 0 has degenerate precursor variation and is explicitly
excluded.

### Diagnostic groups

The report contains:

- **Context signal:** eligible/successful trials, conditioned rates, precision,
  recall, bursts without clear precursors, failed clear precursors, and average
  stored probabilities. Ratios include their numerator and denominator.
- **Working-set structure:** supports, memberships per record, overlap, membership
  strengths, per-set maximum weights, and per-set demand balance.
- **Burst diversity:** counts, duration/intensity summaries, starts, and
  concurrency calculated from half-open burst intervals.
- **Demand decomposition:** global and per-window baseline, noise, and working-set
  counts/fractions plus event-count summaries.
- **Observable associations:** next-window activation rates by user, request type,
  and their pair. Raw categories are retained; min/max ranges use a documented
  support floor of 10 paired eligible-trial observations.

Warnings cover missing demonstrations, insufficient/degenerate variation,
inactive or traffic-free sets, absent concurrency or overlap, deterministic
supported observable categories, undefined ratios, and total traffic
concentration.

These diagnostics are descriptive. No universal precision, recall, balance,
noise, or shortcut threshold is imposed because scientific usefulness depends on
the experiment. The validator performs no feature construction, demand-matrix
creation, learning, factorization, prediction, or other Milestone 2 work.

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
