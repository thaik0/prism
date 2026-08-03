"""Scientific validation for persisted Prism Milestone 1 workload runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Hashable, Mapping, Sequence

from prism.workload.config import WorkloadConfig, WorkloadConfigError
from prism.workload.models import OBSERVABLE_EVENT_FIELDS


VALIDATION_SCHEMA_VERSION = 1
SUPPORTED_SOURCE_SCHEMA_VERSION = 1
NUMERIC_TOLERANCE = 1e-12
OBSERVABLE_ASSOCIATION_SUPPORT_FLOOR = 10
SOURCE_ARTIFACT_FILENAMES = (
    "config.json",
    "observable_events.jsonl",
    "hidden_ground_truth.json",
    "summary.json",
)
VALIDATION_REPORT_FILENAME = "workload_validation.json"

HIDDEN_FIELDS = frozenset(
    {
        "working_set_id",
        "memberships",
        "affinities",
        "precursor_score",
        "activation_probability",
        "activated",
        "burst_id",
        "intensity",
        "source",
        "baseline_popularity",
        "random_draw",
    }
)


class WorkloadValidationError(ValueError):
    """Raised when persisted workload artifacts are structurally invalid."""


@dataclass(frozen=True)
class WorkloadValidationResult:
    """A deterministic scientific validation report."""

    report: dict[str, Any]

    @property
    def demonstrations_passed(self) -> bool:
        return bool(
            self.report["demonstration_checks"][
                "all_required_demonstrations_passed"
            ]
        )

    @property
    def intensity_signal_passed(self) -> bool:
        return bool(
            self.report["intensity_predictability"]["signal_gate"][
                "all_required_conditions_passed"
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return self.report


@dataclass(frozen=True)
class _RequestContext:
    request_id: int
    session_id: int
    window_id: int
    user_id: int
    request_type: str


@dataclass(frozen=True)
class _ValidatedRun:
    config: WorkloadConfig
    events: tuple[dict[str, Any], ...]
    hidden: dict[str, Any]
    summary: dict[str, Any]
    requests: tuple[_RequestContext, ...]
    event_counts_by_window: tuple[int, ...]


@dataclass(frozen=True)
class _PrecursorClassification:
    report: dict[str, Any]
    trial_classes: dict[tuple[int, int], str | None]


def validate_workload_run(run_dir: str | Path) -> WorkloadValidationResult:
    """Read, structurally validate, and diagnose one persisted workload run."""

    source_bytes = _read_source_artifacts(Path(run_dir))
    source_hashes = {
        filename: hashlib.sha256(source_bytes[filename]).hexdigest()
        for filename in SOURCE_ARTIFACT_FILENAMES
    }
    config_raw = _load_json(source_bytes["config.json"], "config.json")
    try:
        config = WorkloadConfig.from_dict(config_raw)
    except WorkloadConfigError as error:
        raise WorkloadValidationError(f"config.json: {error}") from error
    events = _load_jsonl(
        source_bytes["observable_events.jsonl"], "observable_events.jsonl"
    )
    hidden = _load_json(
        source_bytes["hidden_ground_truth.json"], "hidden_ground_truth.json"
    )
    summary = _load_json(source_bytes["summary.json"], "summary.json")
    validated = _validate_structure(config, events, hidden, summary)

    warnings: list[str] = []
    classification = _classify_precursors(
        hidden["activation_trials"], config.num_working_sets, warnings
    )
    demonstration_checks = _demonstration_checks(
        hidden["activation_trials"], classification, warnings
    )
    context_signal = _context_signal_diagnostics(
        hidden["activation_trials"], classification.trial_classes, warnings
    )
    working_set_structure = _working_set_structure_diagnostics(
        config, hidden, warnings
    )
    burst_diversity = _burst_diversity_diagnostics(config, hidden, warnings)
    intensity_predictability = _intensity_predictability_diagnostics(
        config, hidden, warnings
    )
    demand_decomposition = _demand_decomposition_diagnostics(
        validated, warnings
    )
    observable_associations = _observable_association_diagnostics(
        validated, warnings
    )

    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "source_artifact_sha256": source_hashes,
        "structural_validation": {
            "passed": True,
            "source_schema_version": SUPPORTED_SOURCE_SCHEMA_VERSION,
            "observable_event_count": len(events),
            "request_count": len(validated.requests),
            "session_count": summary["total_sessions"],
            "activation_trial_count": len(hidden["activation_trials"]),
            "burst_count": len(hidden["bursts"]),
        },
        "demonstration_checks": {
            **demonstration_checks,
            "precursor_classification": classification.report,
        },
        "context_signal": context_signal,
        "working_set_structure": working_set_structure,
        "burst_diversity": burst_diversity,
        "intensity_predictability": intensity_predictability,
        "demand_decomposition": demand_decomposition,
        "observable_associations": observable_associations,
        "warnings": warnings,
    }
    return WorkloadValidationResult(report=report)


def write_validation_report(
    result: WorkloadValidationResult, output_path: str | Path
) -> None:
    """Deterministically replace only the derived validation report."""

    if not isinstance(result, WorkloadValidationResult):
        raise TypeError("result must be a WorkloadValidationResult")
    path = Path(output_path)
    path.write_text(
        json.dumps(result.to_dict(), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _read_source_artifacts(run_dir: Path) -> dict[str, bytes]:
    if not run_dir.is_dir():
        raise WorkloadValidationError(f"run directory does not exist: {run_dir}")
    source_bytes: dict[str, bytes] = {}
    for filename in SOURCE_ARTIFACT_FILENAMES:
        path = run_dir / filename
        if not path.is_file():
            raise WorkloadValidationError(f"missing required artifact: {filename}")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise WorkloadValidationError(f"cannot read {filename}: {error}") from error
        if not content.endswith(b"\n"):
            raise WorkloadValidationError(f"{filename} must end with a newline")
        source_bytes[filename] = content
    return source_bytes


def _load_json(content: bytes, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(content, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkloadValidationError(f"malformed JSON in {filename}: {error}") from error
    if not isinstance(value, dict):
        raise WorkloadValidationError(f"{filename} root must be a JSON object")
    return value


def _load_jsonl(content: bytes, filename: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WorkloadValidationError(f"malformed UTF-8 in {filename}: {error}") from error
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise WorkloadValidationError(
                f"{filename} line {line_number} must not be blank"
            )
        try:
            event = json.loads(line, object_pairs_hook=_unique_json_object)
        except json.JSONDecodeError as error:
            raise WorkloadValidationError(
                f"malformed JSON in {filename} line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise WorkloadValidationError(
                f"{filename} line {line_number} must be a JSON object"
            )
        events.append(event)
    if not events:
        raise WorkloadValidationError(f"{filename} must contain at least one event")
    return events


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkloadValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _validate_structure(
    config: WorkloadConfig,
    events: list[dict[str, Any]],
    hidden: dict[str, Any],
    summary: dict[str, Any],
) -> _ValidatedRun:
    _require_exact_keys(
        "hidden_ground_truth.json",
        hidden,
        {
            "schema_version",
            "seed",
            "record_sizes_bytes",
            "working_set_memberships",
            "user_working_set_affinities",
            "request_type_working_set_affinities",
            "user_request_type_preferences",
            "baseline_record_popularity",
            "precursor_scores_by_window",
            "activation_trials",
            "bursts",
            "access_source_counts_by_window",
        },
    )
    _require_exact_keys(
        "summary.json",
        summary,
        {
            "schema_version",
            "seed",
            "num_windows",
            "num_records",
            "total_sessions",
            "total_requests",
            "total_events",
            "total_bursts",
            "baseline_access_count",
            "noise_access_count",
            "working_set_access_count",
            "records_in_multiple_working_sets",
        },
    )
    if _integer("hidden.schema_version", hidden["schema_version"]) != 1:
        raise WorkloadValidationError("hidden.schema_version must equal 1")
    if _integer("summary.schema_version", summary["schema_version"]) != 1:
        raise WorkloadValidationError("summary.schema_version must equal 1")
    if _integer("hidden.seed", hidden["seed"]) != config.seed:
        raise WorkloadValidationError("config seed and hidden seed do not agree")
    if _integer("summary.seed", summary["seed"]) != config.seed:
        raise WorkloadValidationError("config seed and summary seed do not agree")

    record_sizes = _list("hidden.record_sizes_bytes", hidden["record_sizes_bytes"])
    if len(record_sizes) != config.num_records:
        raise WorkloadValidationError(
            "hidden.record_sizes_bytes length does not match num_records"
        )
    for record_id, size in enumerate(record_sizes):
        resolved = _integer(f"record_sizes_bytes[{record_id}]", size, minimum=1)
        if not config.record_size_min_bytes <= resolved <= config.record_size_max_bytes:
            raise WorkloadValidationError(
                f"record_sizes_bytes[{record_id}] is outside configured bounds"
            )

    event_counts_by_window = [0] * config.num_windows
    session_contexts: dict[int, tuple[int, int]] = {}
    request_contexts: dict[int, _RequestContext] = {}
    previous_window = -1
    for expected_index, event in enumerate(events):
        path = f"observable_events.jsonl event {expected_index}"
        _require_exact_keys(path, event, set(OBSERVABLE_EVENT_FIELDS))
        if set(event).intersection(HIDDEN_FIELDS):
            raise WorkloadValidationError(f"{path} contains hidden fields")
        event_index = _integer(f"{path}.event_index", event["event_index"], minimum=0)
        if event_index != expected_index:
            raise WorkloadValidationError(
                "observable event indices must be contiguous and ordered"
            )
        window_id = _integer(
            f"{path}.window_id", event["window_id"], minimum=0, maximum=config.num_windows - 1
        )
        if window_id < previous_window:
            raise WorkloadValidationError("observable event windows must be ordered")
        previous_window = window_id
        record_id = _integer(
            f"{path}.record_id", event["record_id"], minimum=0, maximum=config.num_records - 1
        )
        if _integer(f"{path}.record_size_bytes", event["record_size_bytes"], minimum=1) != record_sizes[record_id]:
            raise WorkloadValidationError(
                f"{path} record size is inconsistent with hidden record size"
            )
        user_id = _integer(
            f"{path}.user_id", event["user_id"], minimum=0, maximum=config.num_users - 1
        )
        session_id = _integer(f"{path}.session_id", event["session_id"], minimum=0)
        request_id = _integer(f"{path}.request_id", event["request_id"], minimum=0)
        request_type = _category(f"{path}.request_type", event["request_type"])
        if request_type not in config.request_types:
            raise WorkloadValidationError(f"{path}.request_type is not configured")
        operation_type = _category(f"{path}.operation_type", event["operation_type"])
        if operation_type not in config.operation_type_probabilities:
            raise WorkloadValidationError(f"{path}.operation_type is not configured")

        session_context = (window_id, user_id)
        if session_id in session_contexts and session_contexts[session_id] != session_context:
            raise WorkloadValidationError(
                f"session {session_id} belongs to multiple windows or users"
            )
        session_contexts[session_id] = session_context
        request_context = _RequestContext(
            request_id=request_id,
            session_id=session_id,
            window_id=window_id,
            user_id=user_id,
            request_type=request_type,
        )
        if request_id in request_contexts and request_contexts[request_id] != request_context:
            raise WorkloadValidationError(
                f"request {request_id} has inconsistent session or observable context"
            )
        request_contexts[request_id] = request_context
        event_counts_by_window[window_id] += 1

    if sorted(session_contexts) != list(range(len(session_contexts))):
        raise WorkloadValidationError("session IDs must be contiguous from zero")
    if sorted(request_contexts) != list(range(len(request_contexts))):
        raise WorkloadValidationError("request IDs must be contiguous from zero")

    membership_counts = _validate_hidden_vectors(config, hidden)
    bursts = _validate_bursts(config, hidden["bursts"])
    precursor_scores = _validate_precursor_scores(
        config, hidden["precursor_scores_by_window"]
    )
    _validate_activation_trials(
        config, hidden["activation_trials"], bursts, precursor_scores
    )
    source_totals = _validate_source_counts(
        config,
        hidden["access_source_counts_by_window"],
        event_counts_by_window,
    )

    expected_summary = {
        "num_windows": config.num_windows,
        "num_records": config.num_records,
        "total_sessions": len(session_contexts),
        "total_requests": len(request_contexts),
        "total_events": len(events),
        "total_bursts": len(bursts),
        "baseline_access_count": source_totals[0],
        "noise_access_count": source_totals[1],
        "working_set_access_count": source_totals[2],
        "records_in_multiple_working_sets": sum(count > 1 for count in membership_counts),
    }
    for field, expected in expected_summary.items():
        actual = _integer(f"summary.{field}", summary[field], minimum=0)
        if actual != expected:
            raise WorkloadValidationError(
                f"summary.{field} is {actual}, expected {expected} from source artifacts"
            )

    return _ValidatedRun(
        config=config,
        events=tuple(events),
        hidden=hidden,
        summary=summary,
        requests=tuple(request_contexts[index] for index in sorted(request_contexts)),
        event_counts_by_window=tuple(event_counts_by_window),
    )


def _validate_hidden_vectors(
    config: WorkloadConfig, hidden: dict[str, Any]
) -> list[int]:
    memberships = _list(
        "hidden.working_set_memberships", hidden["working_set_memberships"]
    )
    if len(memberships) != config.num_working_sets:
        raise WorkloadValidationError(
            "working_set_memberships length does not match num_working_sets"
        )
    membership_counts = [0] * config.num_records
    for expected_id, membership in enumerate(memberships):
        path = f"working_set_memberships[{expected_id}]"
        mapping = _mapping(path, membership)
        _require_exact_keys(path, mapping, {"working_set_id", "members"})
        if _integer(f"{path}.working_set_id", mapping["working_set_id"]) != expected_id:
            raise WorkloadValidationError(f"{path} working_set_id is out of order")
        members = _list(f"{path}.members", mapping["members"])
        if not config.working_set_support_min <= len(members) <= config.working_set_support_max:
            raise WorkloadValidationError(f"{path} support size is outside configured bounds")
        record_ids: list[int] = []
        weights: list[float] = []
        for member_index, member in enumerate(members):
            member_path = f"{path}.members[{member_index}]"
            member_mapping = _mapping(member_path, member)
            _require_exact_keys(member_path, member_mapping, {"record_id", "weight"})
            record_id = _integer(
                f"{member_path}.record_id",
                member_mapping["record_id"],
                minimum=0,
                maximum=config.num_records - 1,
            )
            record_ids.append(record_id)
            weights.append(_positive_number(f"{member_path}.weight", member_mapping["weight"]))
            membership_counts[record_id] += 1
        if len(set(record_ids)) != len(record_ids):
            raise WorkloadValidationError(f"{path} contains duplicate record IDs")
        _require_normalized(f"{path} membership weights", weights)

    _validate_weight_vectors(
        hidden["user_working_set_affinities"],
        "user_working_set_affinities",
        "user_id",
        tuple(range(config.num_users)),
        config.num_working_sets,
    )
    _validate_weight_vectors(
        hidden["request_type_working_set_affinities"],
        "request_type_working_set_affinities",
        "request_type",
        config.request_types,
        config.num_working_sets,
    )
    _validate_weight_vectors(
        hidden["user_request_type_preferences"],
        "user_request_type_preferences",
        "user_id",
        tuple(range(config.num_users)),
        len(config.request_types),
    )
    baseline = _list(
        "hidden.baseline_record_popularity", hidden["baseline_record_popularity"]
    )
    if len(baseline) != config.num_records:
        raise WorkloadValidationError(
            "baseline_record_popularity length does not match num_records"
        )
    _require_normalized(
        "baseline_record_popularity",
        [_positive_number(f"baseline_record_popularity[{i}]", value) for i, value in enumerate(baseline)],
    )
    return membership_counts


def _validate_weight_vectors(
    raw_vectors: Any,
    path: str,
    id_field: str,
    expected_ids: Sequence[int | str],
    expected_weight_count: int,
) -> None:
    vectors = _list(path, raw_vectors)
    if len(vectors) != len(expected_ids):
        raise WorkloadValidationError(f"{path} length is invalid")
    for index, expected_id in enumerate(expected_ids):
        vector_path = f"{path}[{index}]"
        vector = _mapping(vector_path, vectors[index])
        _require_exact_keys(vector_path, vector, {id_field, "weights"})
        if vector[id_field] != expected_id or isinstance(vector[id_field], bool):
            raise WorkloadValidationError(f"{vector_path}.{id_field} is invalid")
        weights = _list(f"{vector_path}.weights", vector["weights"])
        if len(weights) != expected_weight_count:
            raise WorkloadValidationError(f"{vector_path}.weights length is invalid")
        _require_normalized(
            f"{vector_path}.weights",
            [_positive_number(f"{vector_path}.weights[{i}]", value) for i, value in enumerate(weights)],
        )


def _validate_bursts(
    config: WorkloadConfig, raw_bursts: Any
) -> list[dict[str, Any]]:
    bursts = _list("hidden.bursts", raw_bursts)
    by_working_set: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for expected_id, raw_burst in enumerate(bursts):
        path = f"bursts[{expected_id}]"
        burst = _mapping(path, raw_burst)
        _require_exact_keys(
            path,
            burst,
            {
                "burst_id",
                "working_set_id",
                "start_window",
                "sampled_duration_windows",
                "end_window_exclusive",
                "intensity",
            },
        )
        burst_id = _integer(f"{path}.burst_id", burst["burst_id"], minimum=0)
        if burst_id != expected_id:
            raise WorkloadValidationError("burst IDs must be contiguous and ordered")
        working_set_id = _integer(
            f"{path}.working_set_id",
            burst["working_set_id"],
            minimum=0,
            maximum=config.num_working_sets - 1,
        )
        start = _integer(
            f"{path}.start_window",
            burst["start_window"],
            minimum=0,
            maximum=config.num_windows - 1,
        )
        duration = _integer(
            f"{path}.sampled_duration_windows",
            burst["sampled_duration_windows"],
            minimum=config.burst_duration_min_windows,
            maximum=config.burst_duration_max_windows,
        )
        end = _integer(f"{path}.end_window_exclusive", burst["end_window_exclusive"], minimum=1)
        if end != start + duration:
            raise WorkloadValidationError(f"{path} end window does not match duration")
        intensity = _positive_number(f"{path}.intensity", burst["intensity"])
        if not config.burst_intensity_min <= intensity <= config.burst_intensity_max:
            raise WorkloadValidationError(f"{path} intensity is outside configured bounds")
        by_working_set[working_set_id].append(burst)

    for working_set_id, working_set_bursts in sorted(by_working_set.items()):
        ordered = sorted(working_set_bursts, key=lambda burst: burst["start_window"])
        for previous, current in zip(ordered, ordered[1:]):
            if previous["end_window_exclusive"] > current["start_window"]:
                raise WorkloadValidationError(
                    f"working set {working_set_id} has overlapping bursts"
                )
    return bursts


def _validate_precursor_scores(
    config: WorkloadConfig, raw_scores: Any
) -> list[tuple[float, ...]]:
    rows = _list("hidden.precursor_scores_by_window", raw_scores)
    if len(rows) != config.num_windows:
        raise WorkloadValidationError(
            "precursor_scores_by_window length does not match num_windows"
        )
    resolved_rows: list[tuple[float, ...]] = []
    for window_id, raw_row in enumerate(rows):
        path = f"precursor_scores_by_window[{window_id}]"
        row = _mapping(path, raw_row)
        _require_exact_keys(path, row, {"window_id", "scores"})
        if _integer(f"{path}.window_id", row["window_id"]) != window_id:
            raise WorkloadValidationError(f"{path} window_id is invalid")
        scores = _list(f"{path}.scores", row["scores"])
        if len(scores) != config.num_working_sets:
            raise WorkloadValidationError(f"{path}.scores length is invalid")
        resolved_rows.append(
            tuple(
                _probability(f"{path}.scores[{index}]", value)
                for index, value in enumerate(scores)
            )
        )
    return resolved_rows


def _validate_activation_trials(
    config: WorkloadConfig,
    raw_trials: Any,
    bursts: list[dict[str, Any]],
    precursor_scores: list[tuple[float, ...]],
) -> None:
    trials = _list("hidden.activation_trials", raw_trials)
    trial_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    matched_bursts: dict[int, int] = defaultdict(int)
    previous_key = (-1, -1)
    for trial_index, raw_trial in enumerate(trials):
        path = f"activation_trials[{trial_index}]"
        trial = _mapping(path, raw_trial)
        _require_exact_keys(
            path,
            trial,
            {
                "window_id",
                "working_set_id",
                "previous_window_precursor_score",
                "contextual_probability",
                "spontaneous_probability",
                "activation_probability",
                "activated",
                "created_burst_id",
            },
        )
        window_id = _integer(
            f"{path}.window_id", trial["window_id"], minimum=0, maximum=config.num_windows - 1
        )
        working_set_id = _integer(
            f"{path}.working_set_id",
            trial["working_set_id"],
            minimum=0,
            maximum=config.num_working_sets - 1,
        )
        key = (window_id, working_set_id)
        if key <= previous_key:
            raise WorkloadValidationError(
                "activation trials must be uniquely ordered by window and working set"
            )
        previous_key = key
        trial_by_key[key] = trial
        score = _probability(
            f"{path}.previous_window_precursor_score",
            trial["previous_window_precursor_score"],
        )
        expected_score = 0.0 if window_id == 0 else precursor_scores[window_id - 1][working_set_id]
        _require_close(
            f"{path} previous-window precursor score", score, expected_score
        )
        spontaneous = _probability(
            f"{path}.spontaneous_probability", trial["spontaneous_probability"]
        )
        _require_close(
            f"{path} spontaneous probability",
            spontaneous,
            config.spontaneous_activation_probability,
        )
        contextual = _probability(
            f"{path}.contextual_probability", trial["contextual_probability"]
        )
        expected_contextual = config.precursor_probability_scale * score
        _require_close(
            f"{path} contextual probability", contextual, expected_contextual
        )
        combined = _probability(
            f"{path}.activation_probability", trial["activation_probability"]
        )
        expected_combined = 1.0 - (1.0 - spontaneous) * (1.0 - contextual)
        _require_close(
            f"{path} combined activation probability", combined, expected_combined
        )
        if not isinstance(trial["activated"], bool):
            raise WorkloadValidationError(f"{path}.activated must be a boolean")
        created_burst_id = trial["created_burst_id"]
        if trial["activated"]:
            burst_id = _integer(f"{path}.created_burst_id", created_burst_id, minimum=0)
            if burst_id >= len(bursts):
                raise WorkloadValidationError(
                    f"{path}.created_burst_id references an invalid burst"
                )
            burst = bursts[burst_id]
            if burst["working_set_id"] != working_set_id or burst["start_window"] != window_id:
                raise WorkloadValidationError(
                    f"{path} activation outcome is inconsistent with created burst"
                )
            matched_bursts[burst_id] += 1
        elif created_burst_id is not None:
            raise WorkloadValidationError(
                f"{path} failed activation must not create a burst"
            )

    for window_id in range(config.num_windows):
        for working_set_id in range(config.num_working_sets):
            active_from_prior_window = any(
                burst["working_set_id"] == working_set_id
                and burst["start_window"] < window_id < burst["end_window_exclusive"]
                for burst in bursts
            )
            has_trial = (window_id, working_set_id) in trial_by_key
            if active_from_prior_window == has_trial:
                expected = "absent" if active_from_prior_window else "present"
                raise WorkloadValidationError(
                    f"activation trial for window {window_id}, working set {working_set_id} must be {expected}"
                )
    for burst_id in range(len(bursts)):
        if matched_bursts[burst_id] != 1:
            raise WorkloadValidationError(
                f"burst {burst_id} must correspond to exactly one successful activation trial"
            )


def _validate_source_counts(
    config: WorkloadConfig,
    raw_rows: Any,
    event_counts_by_window: list[int],
) -> tuple[int, int, int]:
    rows = _list("hidden.access_source_counts_by_window", raw_rows)
    if len(rows) != config.num_windows:
        raise WorkloadValidationError(
            "access_source_counts_by_window length does not match num_windows"
        )
    total_baseline = 0
    total_noise = 0
    total_working_set = 0
    for window_id, raw_row in enumerate(rows):
        path = f"access_source_counts_by_window[{window_id}]"
        row = _mapping(path, raw_row)
        _require_exact_keys(
            path,
            row,
            {
                "window_id",
                "baseline_access_count",
                "noise_access_count",
                "working_set_access_count",
                "working_set_access_counts",
            },
        )
        if _integer(f"{path}.window_id", row["window_id"]) != window_id:
            raise WorkloadValidationError(f"{path} window_id is invalid")
        baseline = _integer(
            f"{path}.baseline_access_count", row["baseline_access_count"], minimum=0
        )
        noise = _integer(f"{path}.noise_access_count", row["noise_access_count"], minimum=0)
        per_working_set = _list(
            f"{path}.working_set_access_counts", row["working_set_access_counts"]
        )
        if len(per_working_set) != config.num_working_sets:
            raise WorkloadValidationError(
                f"{path}.working_set_access_counts length is invalid"
            )
        working_set_total = 0
        for working_set_id, raw_count in enumerate(per_working_set):
            count_path = f"{path}.working_set_access_counts[{working_set_id}]"
            count = _mapping(count_path, raw_count)
            _require_exact_keys(count_path, count, {"working_set_id", "access_count"})
            if _integer(f"{count_path}.working_set_id", count["working_set_id"]) != working_set_id:
                raise WorkloadValidationError(f"{count_path} working_set_id is invalid")
            working_set_total += _integer(
                f"{count_path}.access_count", count["access_count"], minimum=0
            )
        stored_working_set_total = _integer(
            f"{path}.working_set_access_count", row["working_set_access_count"], minimum=0
        )
        if stored_working_set_total != working_set_total:
            raise WorkloadValidationError(
                f"{path} working-set source total is inconsistent"
            )
        if baseline + noise + working_set_total != event_counts_by_window[window_id]:
            raise WorkloadValidationError(
                f"{path} source totals do not match observable events"
            )
        total_baseline += baseline
        total_noise += noise
        total_working_set += working_set_total
    return total_baseline, total_noise, total_working_set


def _classify_precursors(
    trials: list[dict[str, Any]], num_working_sets: int, warnings: list[str]
) -> _PrecursorClassification:
    by_working_set: dict[int, list[dict[str, Any]]] = {
        working_set_id: [] for working_set_id in range(num_working_sets)
    }
    for trial in trials:
        by_working_set[trial["working_set_id"]].append(trial)

    trial_classes: dict[tuple[int, int], str | None] = {}
    working_set_reports: list[dict[str, Any]] = []
    clear_count = 0
    no_clear_count = 0
    intermediate_count = 0
    excluded_insufficient = 0
    excluded_degenerate = 0
    for working_set_id in range(num_working_sets):
        working_set_trials = by_working_set[working_set_id]
        scores = [trial["previous_window_precursor_score"] for trial in working_set_trials]
        if len(scores) < 2:
            status = "insufficient"
            q1 = None
            q3 = None
            excluded_insufficient += len(scores)
            _add_warning(
                warnings,
                f"working set {working_set_id} has {len(scores)} eligible trial(s); at least 2 are required for precursor quartiles",
            )
        else:
            q1, _, q3 = statistics.quantiles(scores, n=4, method="inclusive")
            if q1 == q3:
                status = "degenerate"
                excluded_degenerate += len(scores)
                _add_warning(
                    warnings,
                    f"working set {working_set_id} has degenerate precursor variation: Q1=Q3={q1}",
                )
            else:
                status = "classified"

        for trial in working_set_trials:
            key = (trial["window_id"], working_set_id)
            if status != "classified":
                trial_classes[key] = None
                continue
            score = trial["previous_window_precursor_score"]
            if score >= q3:
                trial_classes[key] = "clear_precursor"
                clear_count += 1
            elif score <= q1:
                trial_classes[key] = "no_clear_precursor"
                no_clear_count += 1
            else:
                trial_classes[key] = "intermediate"
                intermediate_count += 1
        working_set_reports.append(
            {
                "working_set_id": working_set_id,
                "eligible_trial_count": len(scores),
                "classification_status": status,
                "q1": q1,
                "q3": q3,
            }
        )

    return _PrecursorClassification(
        report={
            "method": "statistics.quantiles(scores, n=4, method='inclusive') per working set",
            "clear_precursor_rule": "score >= Q3",
            "no_clear_precursor_rule": "score <= Q1",
            "degenerate_rule": "exclude working set when Q1 == Q3",
            "working_sets": working_set_reports,
            "trial_counts": {
                "clear_precursor": clear_count,
                "no_clear_precursor": no_clear_count,
                "intermediate": intermediate_count,
                "excluded_insufficient": excluded_insufficient,
                "excluded_degenerate": excluded_degenerate,
            },
        },
        trial_classes=trial_classes,
    )


def _demonstration_checks(
    trials: list[dict[str, Any]],
    classification: _PrecursorClassification,
    warnings: list[str],
) -> dict[str, Any]:
    counts = {
        "clear_precursor_followed_by_burst": 0,
        "clear_precursor_followed_by_no_burst": 0,
        "no_clear_precursor_followed_by_burst": 0,
    }
    for trial in trials:
        trial_class = classification.trial_classes[
            (trial["window_id"], trial["working_set_id"])
        ]
        if trial_class == "clear_precursor":
            key = (
                "clear_precursor_followed_by_burst"
                if trial["activated"]
                else "clear_precursor_followed_by_no_burst"
            )
            counts[key] += 1
        elif trial_class == "no_clear_precursor" and trial["activated"]:
            counts["no_clear_precursor_followed_by_burst"] += 1

    checks = {
        name: {"count": count, "passed": count > 0}
        for name, count in counts.items()
    }
    for name, check in checks.items():
        if not check["passed"]:
            _add_warning(
                warnings,
                f"representative demonstration category {name} is absent (count=0)",
            )
    return {
        **checks,
        "all_required_demonstrations_passed": all(
            check["passed"] for check in checks.values()
        ),
    }


def _context_signal_diagnostics(
    trials: list[dict[str, Any]],
    trial_classes: dict[tuple[int, int], str | None],
    warnings: list[str],
) -> dict[str, Any]:
    classified = [
        (trial, trial_classes[(trial["window_id"], trial["working_set_id"])])
        for trial in trials
        if trial_classes[(trial["window_id"], trial["working_set_id"])] is not None
    ]
    clear = [trial for trial, label in classified if label == "clear_precursor"]
    no_clear = [trial for trial, label in classified if label == "no_clear_precursor"]
    intermediate = [trial for trial, label in classified if label == "intermediate"]
    successful = sum(trial["activated"] for trial in trials)
    clear_successes = sum(trial["activated"] for trial in clear)
    no_clear_successes = sum(trial["activated"] for trial in no_clear)
    classified_successes = sum(trial["activated"] for trial, _ in classified)
    failed_clear = len(clear) - clear_successes

    return {
        "eligible_activation_trials": len(trials),
        "successful_activations": successful,
        "unconditional_activation_rate": _ratio(
            successful, len(trials), "unconditional activation rate", warnings
        ),
        "activation_rate_after_clear_precursor": _ratio(
            clear_successes,
            len(clear),
            "activation rate after clear precursor",
            warnings,
        ),
        "activation_rate_after_no_clear_precursor": _ratio(
            no_clear_successes,
            len(no_clear),
            "activation rate after no clear precursor",
            warnings,
        ),
        "clear_precursor_precision": _ratio(
            clear_successes, len(clear), "clear-precursor precision", warnings
        ),
        "clear_precursor_recall": _ratio(
            clear_successes,
            classified_successes,
            "clear-precursor recall",
            warnings,
        ),
        "classified_bursts_without_clear_precursor": _ratio(
            classified_successes - clear_successes,
            classified_successes,
            "classified bursts without clear precursor fraction",
            warnings,
        ),
        "clear_precursors_not_followed_by_burst": _ratio(
            failed_clear,
            len(clear),
            "clear precursors not followed by burst fraction",
            warnings,
        ),
        "average_activation_probability": {
            "clear_precursor": _average(
                [trial["activation_probability"] for trial in clear],
                "average clear-precursor activation probability",
                warnings,
            ),
            "no_clear_precursor": _average(
                [trial["activation_probability"] for trial in no_clear],
                "average no-clear-precursor activation probability",
                warnings,
            ),
            "intermediate": _average(
                [trial["activation_probability"] for trial in intermediate],
                "average intermediate activation probability",
                warnings,
            ),
        },
    }


def _working_set_structure_diagnostics(
    config: WorkloadConfig, hidden: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    memberships = hidden["working_set_memberships"]
    support_sizes = [len(membership["members"]) for membership in memberships]
    membership_counts = [0] * config.num_records
    membership_weights: list[float] = []
    per_working_set_maximums: list[dict[str, Any]] = []
    for membership in memberships:
        weights = [member["weight"] for member in membership["members"]]
        membership_weights.extend(weights)
        per_working_set_maximums.append(
            {
                "working_set_id": membership["working_set_id"],
                "maximum_membership_weight": max(weights),
            }
        )
        for member in membership["members"]:
            membership_counts[member["record_id"]] += 1

    unassigned = sum(count == 0 for count in membership_counts)
    single = sum(count == 1 for count in membership_counts)
    multiple = sum(count > 1 for count in membership_counts)
    if multiple == 0:
        _add_warning(warnings, "no records overlap across working sets (count=0)")

    per_working_set_accesses = [0] * config.num_working_sets
    for row in hidden["access_source_counts_by_window"]:
        for count in row["working_set_access_counts"]:
            per_working_set_accesses[count["working_set_id"]] += count["access_count"]
    total_working_set_accesses = sum(per_working_set_accesses)
    demand_rows = []
    for working_set_id, access_count in enumerate(per_working_set_accesses):
        if access_count == 0:
            _add_warning(
                warnings,
                f"working set {working_set_id} generated no burst traffic (access_count=0)",
            )
        demand_rows.append(
            {
                "working_set_id": working_set_id,
                "access_count": access_count,
                "fraction_of_working_set_accesses": (
                    access_count / total_working_set_accesses
                    if total_working_set_accesses
                    else None
                ),
            }
        )
    dominant_share = (
        max(per_working_set_accesses) / total_working_set_accesses
        if total_working_set_accesses
        else None
    )
    if dominant_share == 1.0:
        dominant_id = per_working_set_accesses.index(max(per_working_set_accesses))
        _add_warning(
            warnings,
            f"working set {dominant_id} accounts for all working-set-generated traffic (share=1.0)",
        )
    if total_working_set_accesses == 0:
        _add_warning(warnings, "working-set traffic shares are undefined because total working-set accesses are zero")

    return {
        "support_structure": {
            "num_records": config.num_records,
            "num_working_sets": config.num_working_sets,
            "support_size_by_working_set": [
                {"working_set_id": index, "support_size": size}
                for index, size in enumerate(support_sizes)
            ],
            "support_size_summary": _descriptive_statistics(support_sizes),
            "membership_count_by_record": [
                {"record_id": record_id, "membership_count": count}
                for record_id, count in enumerate(membership_counts)
            ],
            "records_in_zero_working_sets": _ratio(unassigned, config.num_records),
            "records_in_exactly_one_working_set": _ratio(single, config.num_records),
            "records_in_multiple_working_sets": _ratio(multiple, config.num_records),
            "maximum_memberships_per_record": max(membership_counts),
        },
        "membership_strengths": {
            **_descriptive_statistics(membership_weights),
            "maximum_weight_by_working_set": per_working_set_maximums,
        },
        "working_set_demand_balance": {
            "total_working_set_accesses": total_working_set_accesses,
            "by_working_set": demand_rows,
            "dominant_working_set_traffic_share": dominant_share,
            "working_sets_with_zero_generated_accesses": [
                index for index, count in enumerate(per_working_set_accesses) if count == 0
            ],
        },
    }


def _burst_diversity_diagnostics(
    config: WorkloadConfig, hidden: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    bursts = hidden["bursts"]
    burst_counts = [0] * config.num_working_sets
    starts = [0] * config.num_windows
    for burst in bursts:
        burst_counts[burst["working_set_id"]] += 1
        starts[burst["start_window"]] += 1
    zero_burst_sets = [index for index, count in enumerate(burst_counts) if count == 0]
    for working_set_id in zero_burst_sets:
        _add_warning(
            warnings,
            f"working set {working_set_id} never activates (burst_count=0)",
        )

    concurrency = [
        sum(
            burst["start_window"] <= window_id < burst["end_window_exclusive"]
            for burst in bursts
        )
        for window_id in range(config.num_windows)
    ]
    zero_active = sum(count == 0 for count in concurrency)
    one_active = sum(count == 1 for count in concurrency)
    multiple_active = sum(count > 1 for count in concurrency)
    if multiple_active == 0:
        _add_warning(warnings, "no simultaneous bursts occur (multi-active windows=0)")

    return {
        "total_burst_count": len(bursts),
        "burst_count_by_working_set": [
            {"working_set_id": index, "burst_count": count}
            for index, count in enumerate(burst_counts)
        ],
        "working_sets_with_zero_bursts": zero_burst_sets,
        "duration_summary": _descriptive_statistics(
            [burst["sampled_duration_windows"] for burst in bursts]
        ),
        "intensity_summary": _descriptive_statistics(
            [burst["intensity"] for burst in bursts]
        ),
        "burst_start_count_by_window": [
            {"window_id": window_id, "burst_start_count": count}
            for window_id, count in enumerate(starts)
        ],
        "active_working_set_count_by_window": [
            {"window_id": window_id, "active_working_set_count": count}
            for window_id, count in enumerate(concurrency)
        ],
        "windows_with_zero_active_working_sets": _ratio(zero_active, config.num_windows),
        "windows_with_exactly_one_active_working_set": _ratio(one_active, config.num_windows),
        "windows_with_multiple_active_working_sets": _ratio(multiple_active, config.num_windows),
        "maximum_simultaneous_active_working_sets": max(concurrency),
        "simultaneous_burst_window_fraction": multiple_active / config.num_windows,
    }


def _intensity_predictability_diagnostics(
    config: WorkloadConfig,
    hidden: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Diagnose the planted conditional-intensity signal at burst starts."""

    successful_trials = [
        trial for trial in hidden["activation_trials"] if trial["activated"]
    ]
    burst_count_by_working_set = [0] * config.num_working_sets
    scores: list[float] = []
    intensities: list[float] = []
    expected_intensities: list[float] = []
    random_intensity_mean = (
        config.burst_intensity_min + config.burst_intensity_max
    ) / 2.0
    for trial in successful_trials:
        burst = hidden["bursts"][trial["created_burst_id"]]
        score = trial["previous_window_precursor_score"]
        context_implied_intensity = config.burst_intensity_min + score * (
            config.burst_intensity_max - config.burst_intensity_min
        )
        expected_intensity = (
            config.burst_intensity_context_weight * context_implied_intensity
            + (1.0 - config.burst_intensity_context_weight)
            * random_intensity_mean
        )
        burst_count_by_working_set[trial["working_set_id"]] += 1
        scores.append(score)
        intensities.append(burst["intensity"])
        expected_intensities.append(expected_intensity)

    precursor_summary = _descriptive_statistics_with_standard_deviation(scores)
    intensity_summary = _descriptive_statistics_with_standard_deviation(
        intensities
    )
    correlation = _pearson_correlation(
        scores,
        intensities,
        "precursor score and sampled intensity Pearson correlation",
        warnings,
    )
    planted_mae, planted_rmse = _error_metrics(
        intensities,
        expected_intensities,
        "planted-expectation error",
        warnings,
    )
    observed_mean = intensity_summary["mean"]
    constant_predictions = (
        [observed_mean] * len(intensities) if observed_mean is not None else []
    )
    constant_mae, constant_rmse = _error_metrics(
        intensities,
        constant_predictions,
        "constant-observed-mean baseline error",
        warnings,
    )
    residuals = [
        observed - expected
        for observed, expected in zip(
            intensities, expected_intensities, strict=True
        )
    ]
    residual_standard_deviation = (
        statistics.pstdev(residuals) if residuals else None
    )
    if not residuals:
        _add_warning(
            warnings,
            "planted-expectation residual standard deviation is undefined because there are no burst starts",
        )
    quartiles = _intensity_quartile_comparison(scores, intensities, warnings)

    report = {
        "configuration": {
            "burst_intensity_min": config.burst_intensity_min,
            "burst_intensity_max": config.burst_intensity_max,
            "burst_intensity_context_weight": (
                config.burst_intensity_context_weight
            ),
            "random_intensity_mean": random_intensity_mean,
        },
        "burst_start_count": len(successful_trials),
        "burst_start_count_by_working_set": [
            {"working_set_id": working_set_id, "burst_start_count": count}
            for working_set_id, count in enumerate(burst_count_by_working_set)
        ],
        "precursor_score_summary": precursor_summary,
        "sampled_intensity_summary": intensity_summary,
        "distinct_precursor_score_count": len(set(scores)),
        "pearson_correlation": correlation,
        "planted_expectation_error": {
            "mae": planted_mae,
            "rmse": planted_rmse,
        },
        "constant_observed_mean_baseline": {
            "mean_intensity": observed_mean,
            "mae": constant_mae,
            "rmse": constant_rmse,
        },
        "improvement_over_constant_observed_mean": {
            "mae": (
                constant_mae - planted_mae
                if constant_mae is not None and planted_mae is not None
                else None
            ),
            "rmse": (
                constant_rmse - planted_rmse
                if constant_rmse is not None and planted_rmse is not None
                else None
            ),
        },
        "residual_standard_deviation": residual_standard_deviation,
        "quartile_intensity_comparison": quartiles,
    }
    report["signal_gate"] = _intensity_signal_checks(report)
    return report


def _intensity_signal_checks(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the fixed scientific gate without configurable thresholds."""

    weight = report["configuration"]["burst_intensity_context_weight"]
    precursor_standard_deviation = report["precursor_score_summary"][
        "standard_deviation"
    ]
    intensity_standard_deviation = report["sampled_intensity_summary"][
        "standard_deviation"
    ]
    correlation = report["pearson_correlation"]
    planted = report["planted_expectation_error"]
    constant = report["constant_observed_mean_baseline"]
    residual_standard_deviation = report["residual_standard_deviation"]
    quartiles = report["quartile_intensity_comparison"]

    checks = {
        "context_weight_strictly_between_zero_and_one": {
            "value": weight,
            "passed": 0.0 < weight < 1.0,
        },
        "at_least_two_distinct_precursor_scores": {
            "value": report["distinct_precursor_score_count"],
            "passed": report["distinct_precursor_score_count"] >= 2,
        },
        "nonzero_precursor_score_variance": {
            "standard_deviation": precursor_standard_deviation,
            "passed": (
                precursor_standard_deviation is not None
                and precursor_standard_deviation > 0.0
            ),
        },
        "nonzero_sampled_intensity_variance": {
            "standard_deviation": intensity_standard_deviation,
            "passed": (
                intensity_standard_deviation is not None
                and intensity_standard_deviation > 0.0
            ),
        },
        "positive_pearson_correlation": {
            "value": correlation,
            "passed": correlation is not None and correlation > 0.0,
        },
        "planted_mae_lower_than_constant_mean_mae": {
            "planted_mae": planted["mae"],
            "constant_mean_mae": constant["mae"],
            "passed": (
                planted["mae"] is not None
                and constant["mae"] is not None
                and planted["mae"] < constant["mae"]
            ),
        },
        "planted_rmse_lower_than_constant_mean_rmse": {
            "planted_rmse": planted["rmse"],
            "constant_mean_rmse": constant["rmse"],
            "passed": (
                planted["rmse"] is not None
                and constant["rmse"] is not None
                and planted["rmse"] < constant["rmse"]
            ),
        },
        "positive_residual_standard_deviation": {
            "value": residual_standard_deviation,
            "passed": (
                residual_standard_deviation is not None
                and residual_standard_deviation > 0.0
            ),
        },
        "upper_quartile_mean_above_lower_quartile_mean": {
            "lower_mean_intensity": quartiles["lower_mean_intensity"],
            "upper_mean_intensity": quartiles["upper_mean_intensity"],
            "passed": (
                quartiles["lower_mean_intensity"] is not None
                and quartiles["upper_mean_intensity"] is not None
                and quartiles["upper_mean_intensity"]
                > quartiles["lower_mean_intensity"]
            ),
        },
    }
    return {
        "checks": checks,
        "all_required_conditions_passed": all(
            check["passed"] for check in checks.values()
        ),
    }


def _descriptive_statistics_with_standard_deviation(
    values: Sequence[int | float],
) -> dict[str, Any]:
    return {
        **_descriptive_statistics(values),
        "standard_deviation": statistics.pstdev(values) if values else None,
    }


def _pearson_correlation(
    left: Sequence[float],
    right: Sequence[float],
    metric_name: str,
    warnings: list[str],
) -> float | None:
    if len(left) != len(right):
        raise ValueError("Pearson inputs must have equal length")
    if len(left) < 2:
        _add_warning(
            warnings,
            f"{metric_name} is undefined because fewer than 2 observations are available",
        )
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_deviations = [value - left_mean for value in left]
    right_deviations = [value - right_mean for value in right]
    left_squared = math.fsum(value * value for value in left_deviations)
    right_squared = math.fsum(value * value for value in right_deviations)
    if left_squared == 0.0 or right_squared == 0.0:
        _add_warning(
            warnings,
            f"{metric_name} is undefined because at least one input has zero variance",
        )
        return None
    raw = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(
            left_deviations, right_deviations, strict=True
        )
    ) / math.sqrt(left_squared * right_squared)
    return min(1.0, max(-1.0, raw))


def _error_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
    metric_name: str,
    warnings: list[str],
) -> tuple[float | None, float | None]:
    if len(observed) != len(predicted):
        raise ValueError("error-metric inputs must have equal length")
    if not observed:
        _add_warning(
            warnings,
            f"{metric_name} is undefined because there are no burst starts",
        )
        return None, None
    errors = [
        actual - estimate
        for actual, estimate in zip(observed, predicted, strict=True)
    ]
    mae = math.fsum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(
        math.fsum(error * error for error in errors) / len(errors)
    )
    return mae, rmse


def _intensity_quartile_comparison(
    scores: Sequence[float],
    intensities: Sequence[float],
    warnings: list[str],
) -> dict[str, Any]:
    result = {
        "method": "statistics.quantiles(scores, n=4, method='inclusive') across burst starts",
        "lower_rule": "score <= Q1",
        "upper_rule": "score >= Q3",
        "q1": None,
        "q3": None,
        "lower_count": 0,
        "upper_count": 0,
        "lower_mean_intensity": None,
        "upper_mean_intensity": None,
        "upper_minus_lower_mean_intensity": None,
    }
    if len(scores) < 2:
        _add_warning(
            warnings,
            "burst-start intensity quartiles are undefined because fewer than 2 burst starts are available",
        )
        return result
    q1, _, q3 = statistics.quantiles(scores, n=4, method="inclusive")
    result["q1"] = q1
    result["q3"] = q3
    if q1 == q3:
        _add_warning(
            warnings,
            f"burst-start intensity quartiles are undefined because precursor variation is degenerate: Q1=Q3={q1}",
        )
        return result
    lower = [
        intensity
        for score, intensity in zip(scores, intensities, strict=True)
        if score <= q1
    ]
    upper = [
        intensity
        for score, intensity in zip(scores, intensities, strict=True)
        if score >= q3
    ]
    lower_mean = math.fsum(lower) / len(lower)
    upper_mean = math.fsum(upper) / len(upper)
    result.update(
        {
            "lower_count": len(lower),
            "upper_count": len(upper),
            "lower_mean_intensity": lower_mean,
            "upper_mean_intensity": upper_mean,
            "upper_minus_lower_mean_intensity": upper_mean - lower_mean,
        }
    )
    return result


def _demand_decomposition_diagnostics(
    run: _ValidatedRun, warnings: list[str]
) -> dict[str, Any]:
    rows = run.hidden["access_source_counts_by_window"]
    by_window: list[dict[str, Any]] = []
    total_baseline = 0
    total_noise = 0
    total_working_set = 0
    for window_id, row in enumerate(rows):
        total = run.event_counts_by_window[window_id]
        baseline = row["baseline_access_count"]
        noise = row["noise_access_count"]
        working_set = row["working_set_access_count"]
        total_baseline += baseline
        total_noise += noise
        total_working_set += working_set
        by_window.append(
            {
                "window_id": window_id,
                "total_observable_accesses": total,
                "baseline": _ratio(baseline, total, f"window {window_id} baseline fraction", warnings),
                "noise": _ratio(noise, total, f"window {window_id} noise fraction", warnings),
                "working_set": _ratio(
                    working_set,
                    total,
                    f"window {window_id} working-set fraction",
                    warnings,
                ),
            }
        )
    total_events = len(run.events)
    windows_without = sum(row["working_set_access_count"] == 0 for row in rows)
    return {
        "global": {
            "total_observable_accesses": total_events,
            "baseline": _ratio(total_baseline, total_events),
            "noise": _ratio(total_noise, total_events),
            "working_set": _ratio(total_working_set, total_events),
        },
        "by_window": by_window,
        "windows_with_no_working_set_accesses": windows_without,
        "windows_with_working_set_accesses": run.config.num_windows - windows_without,
        "event_count_per_window_summary": _descriptive_statistics(
            run.event_counts_by_window
        ),
    }


def _observable_association_diagnostics(
    run: _ValidatedRun, warnings: list[str]
) -> dict[str, Any]:
    trials_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trial in run.hidden["activation_trials"]:
        trials_by_window[trial["window_id"]].append(trial)

    group_specs: tuple[
        tuple[str, Callable[[_RequestContext], Hashable], Callable[[Hashable], Any]], ...
    ] = (
        ("user_id", lambda request: request.user_id, lambda key: key),
        ("request_type", lambda request: request.request_type, lambda key: key),
        (
            "user_id_request_type",
            lambda request: (request.user_id, request.request_type),
            lambda key: {"user_id": key[0], "request_type": key[1]},
        ),
    )
    reports: dict[str, Any] = {
        "support_floor": OBSERVABLE_ASSOCIATION_SUPPORT_FLOOR,
        "pairing_rule": "each unique request in window t is paired with each eligible activation trial in window t+1",
    }
    for group_name, key_function, display_function in group_specs:
        counts: dict[Hashable, list[int]] = {}
        for request in run.requests:
            counts.setdefault(key_function(request), [0, 0])
        for request in run.requests:
            key = key_function(request)
            for trial in trials_by_window.get(request.window_id + 1, []):
                counts[key][0] += 1
                counts[key][1] += int(trial["activated"])
        ordered_keys = sorted(counts, key=lambda key: repr(key))
        categories: list[dict[str, Any]] = []
        for key in ordered_keys:
            observations, successes = counts[key]
            rate = successes / observations if observations else None
            if observations == 0:
                _add_warning(
                    warnings,
                    f"observable association {group_name} category {display_function(key)!r} has no paired eligible trials",
                )
            if (
                observations >= OBSERVABLE_ASSOCIATION_SUPPORT_FLOOR
                and rate in (0.0, 1.0)
            ):
                _add_warning(
                    warnings,
                    f"observable association {group_name} category {display_function(key)!r} has activation rate {rate} with support {observations}",
                )
            categories.append(
                {
                    "category": display_function(key),
                    "paired_eligible_trial_observations": observations,
                    "activation_successes": successes,
                    "activation_rate": rate,
                }
            )

        supported = [
            row
            for row in categories
            if row["paired_eligible_trial_observations"]
            >= OBSERVABLE_ASSOCIATION_SUPPORT_FLOOR
        ]
        if supported:
            minimum = min(row["activation_rate"] for row in supported)
            maximum = max(row["activation_rate"] for row in supported)
            maximum_row = next(
                row for row in supported if row["activation_rate"] == maximum
            )
            supported_summary = {
                "supported_category_count": len(supported),
                "minimum_activation_rate": minimum,
                "maximum_activation_rate": maximum,
                "activation_rate_range": maximum - minimum,
                "maximum_rate_category": maximum_row["category"],
                "maximum_rate_sample_count": maximum_row[
                    "paired_eligible_trial_observations"
                ],
            }
        else:
            supported_summary = {
                "supported_category_count": 0,
                "minimum_activation_rate": None,
                "maximum_activation_rate": None,
                "activation_rate_range": None,
                "maximum_rate_category": None,
                "maximum_rate_sample_count": None,
            }
            _add_warning(
                warnings,
                f"observable association {group_name} has no category meeting support floor {OBSERVABLE_ASSOCIATION_SUPPORT_FLOOR}",
            )
        reports[group_name] = {
            "categories": categories,
            "supported_rate_summary": supported_summary,
        }
    return reports


def _descriptive_statistics(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": math.fsum(values) / len(values),
        "median": statistics.median(values),
    }


def _ratio(
    numerator: int,
    denominator: int,
    metric_name: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, int | float | None]:
    value = numerator / denominator if denominator else None
    if denominator == 0 and metric_name is not None and warnings is not None:
        _add_warning(
            warnings, f"{metric_name} is undefined because its denominator is zero"
        )
    return {"numerator": numerator, "denominator": denominator, "value": value}


def _average(
    values: Sequence[float], metric_name: str, warnings: list[str]
) -> dict[str, int | float | None]:
    total = math.fsum(values)
    if not values:
        _add_warning(
            warnings, f"{metric_name} is undefined because its denominator is zero"
        )
    return {
        "sum": total,
        "count": len(values),
        "value": total / len(values) if values else None,
    }


def _require_exact_keys(
    path: str, mapping: Mapping[str, Any], expected: set[str]
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise WorkloadValidationError(f"{path} fields are invalid: {', '.join(details)}")


def _mapping(path: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkloadValidationError(f"{path} must be an object")
    return value


def _list(path: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise WorkloadValidationError(f"{path} must be a list")
    return value


def _integer(
    path: str,
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkloadValidationError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise WorkloadValidationError(f"{path} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise WorkloadValidationError(f"{path} must be at most {maximum}")
    return value


def _number(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadValidationError(f"{path} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise WorkloadValidationError(f"{path} must be finite")
    return resolved


def _positive_number(path: str, value: Any) -> float:
    resolved = _number(path, value)
    if resolved <= 0.0:
        raise WorkloadValidationError(f"{path} must be positive")
    return resolved


def _probability(path: str, value: Any) -> float:
    resolved = _number(path, value)
    if not 0.0 <= resolved <= 1.0:
        raise WorkloadValidationError(f"{path} must be in [0, 1]")
    return resolved


def _category(path: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise WorkloadValidationError(f"{path} must be a nonempty string")
    return value


def _require_normalized(path: str, weights: Sequence[float]) -> None:
    if not weights:
        raise WorkloadValidationError(f"{path} must not be empty")
    total = math.fsum(weights)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE):
        raise WorkloadValidationError(f"{path} must sum to 1; got {total}")


def _require_close(path: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE):
        raise WorkloadValidationError(
            f"{path} is {actual}, expected {expected}"
        )


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _print_summary(result: WorkloadValidationResult) -> None:
    report = result.report
    checks = report["demonstration_checks"]
    context = report["context_signal"]
    print("Validated Prism Milestone 1 workload")
    print("Structural validation: passed")
    print(
        "Demonstrations: "
        f"clear->burst={checks['clear_precursor_followed_by_burst']['count']}, "
        f"clear->no_burst={checks['clear_precursor_followed_by_no_burst']['count']}, "
        f"no_clear->burst={checks['no_clear_precursor_followed_by_burst']['count']}"
    )
    print(
        "Context signal: "
        f"eligible_trials={context['eligible_activation_trials']}, "
        f"activation_rate={context['unconditional_activation_rate']['value']}, "
        f"precision={context['clear_precursor_precision']['value']}, "
        f"recall={context['clear_precursor_recall']['value']}"
    )
    intensity = report["intensity_predictability"]
    print(
        "Intensity signal: "
        f"burst_starts={intensity['burst_start_count']}, "
        f"correlation={intensity['pearson_correlation']}, "
        "gate="
        f"{intensity['signal_gate']['all_required_conditions_passed']}"
    )
    print(f"Warnings: {len(report['warnings'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a persisted Prism Milestone 1 workload run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--require-demonstrations",
        action="store_true",
        help="exit nonzero unless all three precursor demonstrations are present",
    )
    parser.add_argument(
        "--require-intensity-signal",
        action="store_true",
        help="exit nonzero unless the planted intensity signal is useful and stochastic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = validate_workload_run(arguments.run_dir)
        write_validation_report(
            result, arguments.run_dir / VALIDATION_REPORT_FILENAME
        )
    except (OSError, WorkloadValidationError) as error:
        parser.error(str(error))
    _print_summary(result)
    failed_gate = False
    if arguments.require_demonstrations and not result.demonstrations_passed:
        print(
            "required representative demonstrations are incomplete",
            file=sys.stderr,
        )
        failed_gate = True
    if arguments.require_intensity_signal and not result.intensity_signal_passed:
        print(
            "required conditional-intensity signal is incomplete",
            file=sys.stderr,
        )
        failed_gate = True
    return int(failed_gate)


if __name__ == "__main__":
    raise SystemExit(main())
