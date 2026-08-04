from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism.native import (
    PARITY_POLICY_ORDER,
    ParitySession,
    TieredStore,
    build_store,
    generate_payloads,
    run_forced_fixture,
)
from prism.native.parity import OperationRecorder
from prism.native.reference import ReferenceLedger
from prism.native.store import ReadResult
from prism.simulation import select_lfu_victim, select_lru_victim


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_forced_fixture_certifies_all_four_policy_paths_and_is_repeatable(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = run_forced_fixture(first_root)
    second = run_forced_fixture(second_root)

    assert {path.name for path in first_root.iterdir()} == {
        "native_store",
        "native_store_manifest.json",
        "parity_operations.jsonl",
        "parity_report.json",
    }
    assert _files(first_root) == _files(second_root)
    assert first.store_artifacts.manifest == second.store_artifacts.manifest
    assert first.parity.report == second.parity.report

    report = first.parity.report
    gates = report["overall_gates"]
    assert report["policy_order"] == list(PARITY_POLICY_ORDER)
    assert gates["overall_parity_passed"] is True
    assert gates["all_four_policies_present"] is True
    assert gates["total_mismatch_count"] == 0
    assert gates["invalidated_policy_count"] == 0
    assert gates["unexpected_exception_count"] == 0
    assert gates["capacity_violation_count"] == 0
    assert set(gates["mismatch_counts_by_category"].values()) == {0}
    assert all(
        summary["parity_passed"]
        for summary in report["policy_summaries"].values()
    )

    operations = first.parity.operations
    static_targets = [
        row
        for row in operations
        if row["policy_id"] == "training_popularity_static"
        and row["operation_type"] == "apply_target_set"
    ]
    assert [row["operation_arguments"]["record_ids"] for row in static_targets] == [
        [0, 1, 2]
    ]
    predictive_targets = [
        row
        for row in operations
        if row["policy_id"] == "predictive_greedy"
        and row["operation_type"] == "apply_target_set"
    ]
    assert [row["operation_arguments"]["record_ids"] for row in predictive_targets] == [
        [0, 1, 2],
        [1, 3],
        [0, 3],
        [0, 3],
    ]
    assert [row["native_result"]["target_changed"] for row in predictive_targets] == [
        True,
        True,
        True,
        False,
    ]
    assert predictive_targets[1]["native_result"]["incoming_record_ids"] == [3]
    assert predictive_targets[1]["native_result"]["outgoing_record_ids"] == [0, 2]

    for policy_id in ("lru", "lfu"):
        summary = report["policy_summaries"][policy_id]
        assert summary["fast_read_count"] > 0
        assert summary["slow_read_count"] > 0
        assert summary["promotion_count"] > 0
        assert summary["eviction_count"] > 0

    json.loads((first_root / "parity_report.json").read_text(encoding="utf-8"))
    for line in (first_root / "parity_operations.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        assert json.loads(line)["parity_status"] == "match"


def test_accepted_lru_and_lfu_tie_breaking_is_deterministic() -> None:
    assert select_lru_victim({5, 2, 8}, {5: 10, 2: 10, 8: 11}) == 2
    assert select_lfu_victim(
        {5, 2, 8},
        {5: 3, 2: 3, 8: 4},
        {5: 10, 2: 10, 8: 1},
    ) == 2
    assert select_lfu_victim(
        {5, 2, 8},
        {5: 3, 2: 3, 8: 3},
        {5: 9, 2: 10, 8: 11},
    ) == 5


class _WrongDigestStore:
    def __init__(self, store: TieredStore) -> None:
        self._store = store
        self.read_calls = 0

    def read(self, record_id: int) -> ReadResult:
        self.read_calls += 1
        result = self._store.read(record_id)
        return ReadResult(b"bad", result.tier, len(b"bad"))

    def snapshot(self):
        return self._store.snapshot()

    def stats(self):
        return self._store.stats()


def test_first_mismatch_invalidates_only_that_policy_and_stops_native_calls(
    tmp_path: Path,
) -> None:
    payload = b"right"
    build_store([(0, payload)], tmp_path / "store")
    native = _WrongDigestStore(TieredStore.open(tmp_path / "store", 5))
    session = ParitySession(
        "lru", ReferenceLedger({0: payload}, 5), native, OperationRecorder()
    )

    first = session.execute("read", {"record_id": 0}, phase="validation")
    second = session.execute("read", {"record_id": 0}, phase="test")

    assert first["parity_status"] == "mismatch"
    assert {item["category"] for item in first["mismatch_details"]} == {
        "payload_size",
        "payload_digest",
    }
    assert session.invalidated_at == 0
    assert second["parity_status"] == "not_compared_due_to_prior_divergence"
    assert native.read_calls == 1


def test_expected_errors_match_native_errors_and_preserve_state(tmp_path: Path) -> None:
    payloads = {0: b"aa", 1: b"bbbb", 2: b"123456"}
    build_store(sorted(payloads.items()), tmp_path / "store")
    reference = ReferenceLedger(payloads, 5)
    native = TieredStore.open(tmp_path / "store", 5)
    session = ParitySession("lru", reference, native, OperationRecorder())

    for operation, arguments, error_code in (
        ("read", {"record_id": 99}, "unknown_record"),
        ("promote", {"record_id": 2}, "oversized_record"),
        ("promote", {"record_id": 0}, None),
        ("promote", {"record_id": 1}, "insufficient_capacity"),
        ("apply_target_set", {"record_ids": [0, 0]}, "invalid_target_set"),
    ):
        row = session.execute(operation, arguments, phase="validation")
        assert row["parity_status"] == "match"
        if error_code is None:
            assert row["python_expected_success"] is True
        else:
            assert row["python_expected_error"]["code"] == error_code
            assert row["native_error"]["code"] == error_code

    assert session.invalidated is False
    assert session.reference.snapshot() == native.snapshot()
    assert session.reference.stats().to_dict() == native.stats().to_dict()


def test_corrupt_and_truncated_reads_match_reference_errors(tmp_path: Path) -> None:
    checksum_dir = tmp_path / "checksum"
    build_store([(0, b"abc")], checksum_dir)
    checksum_native = TieredStore.open(checksum_dir, 3)
    (checksum_dir / "store.data").write_bytes(b"abd")
    checksum_reference = ReferenceLedger({0: b"abc"}, 3)
    checksum_reference.mark_read_failure(0, "checksum_mismatch", offset=0)
    checksum = ParitySession(
        "lru", checksum_reference, checksum_native, OperationRecorder()
    ).execute("read", {"record_id": 0}, phase="validation")
    assert checksum["parity_status"] == "match"
    assert checksum["native_error"]["code"] == "checksum_mismatch"

    truncated_dir = tmp_path / "truncated"
    build_store([(0, b"aa"), (1, b"bbb")], truncated_dir)
    truncated_native = TieredStore.open(truncated_dir, 5)
    (truncated_dir / "store.data").write_bytes(b"aaa")
    truncated_reference = ReferenceLedger({0: b"aa", 1: b"bbb"}, 5)
    truncated_reference.mark_read_failure(1, "truncated_read", offset=3)
    truncated = ParitySession(
        "lru", truncated_reference, truncated_native, OperationRecorder()
    ).execute("read", {"record_id": 1}, phase="validation")
    assert truncated["parity_status"] == "match"
    assert truncated["native_error"]["code"] == "truncated_read"


def test_forced_fixture_rejects_nonempty_output_root(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("owned by caller", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        run_forced_fixture(output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "owned by caller"
