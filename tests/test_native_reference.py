from __future__ import annotations

import pytest

from prism.native.reference import ReferenceLedger, ReferenceStoreError


def test_reference_ledger_exact_hand_calculated_sequence() -> None:
    ledger = ReferenceLedger({1: b"aa", 2: b"bbb", 3: b"cccc"}, 5)

    assert ledger.read(1).tier == "slow"
    assert ledger.promote(1).moved
    assert not ledger.promote(1).moved
    assert ledger.read(1).tier == "fast"
    changed = ledger.apply_target_set([2, 1])
    assert changed.incoming_record_ids == (2,)
    assert changed.outgoing_record_ids == ()
    assert changed.residency.resident_bytes == 5
    unchanged = ledger.apply_target_set([1, 2])
    assert not unchanged.target_changed
    assert ledger.evict(1).bytes_moved == 2
    assert not ledger.evict(1).moved

    before = ledger.snapshot()
    with pytest.raises(ReferenceStoreError) as insufficient:
        ledger.promote(3)
    assert insufficient.value.code == "insufficient_capacity"
    assert ledger.snapshot() == before
    with pytest.raises(ReferenceStoreError) as duplicate:
        ledger.apply_target_set([2, 2])
    assert duplicate.value.code == "invalid_target_set"
    assert ledger.snapshot() == before

    stats = ledger.stats()
    assert stats.successful_slow_reads == 1
    assert stats.successful_slow_read_bytes == 2
    assert stats.successful_fast_reads == 1
    assert stats.successful_fast_read_bytes == 2
    assert stats.promotion_source_reads == 2
    assert stats.promotion_source_read_bytes == 5
    assert stats.committed_promotions == 2
    assert stats.committed_promotion_bytes == 5
    assert stats.committed_evictions == 1
    assert stats.committed_eviction_bytes == 2
    assert stats.target_set_calls == 3
    assert stats.successful_target_set_calls == 2
    assert stats.failed_target_set_calls == 1
    assert stats.current_resident_records == 1
    assert stats.current_resident_bytes == 3
    assert stats.resident_byte_high_water_mark == 5
    assert stats.failures_by_code["insufficient_capacity"] == 1
    assert stats.failures_by_code["invalid_target_set"] == 1


def test_reference_staged_failure_is_atomic_but_counts_successful_staging() -> None:
    ledger = ReferenceLedger({1: b"aa", 2: b"bbb", 3: b"cccc"}, 7)
    ledger.apply_target_set([1])
    ledger.mark_read_failure(3, "checksum_mismatch")
    before = ledger.snapshot()

    with pytest.raises(ReferenceStoreError) as caught:
        ledger.apply_target_set([2, 3])

    assert caught.value.code == "checksum_mismatch"
    assert caught.value.record_id == 3
    assert caught.value.offset == 5
    assert caught.value.path == "store.data"
    assert ledger.snapshot() == before
    stats = ledger.stats()
    assert stats.promotion_source_reads == 2
    assert stats.promotion_source_read_bytes == 5
    assert stats.committed_promotions == 1
    assert stats.committed_promotion_bytes == 2
    assert stats.target_set_calls == 2
    assert stats.successful_target_set_calls == 1
    assert stats.failed_target_set_calls == 1
    assert stats.aborted_staged_bytes == 3
    assert stats.failures_by_code["checksum_mismatch"] == 1


def test_reference_unknown_metadata_does_not_change_native_logical_stats() -> None:
    ledger = ReferenceLedger({1: b"a"}, 1)

    with pytest.raises(ReferenceStoreError, match="not present"):
        ledger.record_metadata(2)

    assert ledger.stats().failures_by_code["unknown_record"] == 0
    with pytest.raises(ReferenceStoreError, match="not present"):
        ledger.read(2)
    assert ledger.stats().failures_by_code["unknown_record"] == 1


def test_reference_clone_is_independent_and_oversized_never_commits() -> None:
    ledger = ReferenceLedger({1: b"aa", 2: b"123456"}, 5)
    clone = ledger.clone()
    clone.promote(1)

    assert ledger.snapshot().resident_record_ids == ()
    assert clone.snapshot().resident_record_ids == (1,)
    with pytest.raises(ReferenceStoreError) as caught:
        ledger.promote(2)
    assert caught.value.code == "oversized_record"
    assert ledger.snapshot().resident_record_ids == ()
