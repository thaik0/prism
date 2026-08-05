#include "prism/storage/tiered_store.hpp"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace prism::storage {
namespace {

StoreError make_error(StoreErrorCode code, std::string message,
                      std::optional<RecordId> record_id = std::nullopt,
                      std::optional<std::uint64_t> byte_offset = std::nullopt,
                      std::optional<std::filesystem::path> path = std::nullopt) {
  return StoreError{code, std::move(message), record_id, byte_offset,
                    std::move(path)};
}

}  // namespace

std::uint64_t StoreStats::failure_count(StoreErrorCode code) const noexcept {
  const auto index = static_cast<std::size_t>(code);
  if (index >= failures_by_code.size()) {
    return 0;
  }
  return failures_by_code[index];
}

Expected<std::unique_ptr<TieredStore>> TieredStore::open(
    const std::filesystem::path& store_directory,
    std::uint64_t fast_capacity_bytes) {
  if (fast_capacity_bytes == 0) {
    return unexpected(make_error(StoreErrorCode::invalid_configuration,
                                 "fast-tier capacity must be positive",
                                 std::nullopt, std::nullopt, store_directory));
  }
  auto index = load_store_index(store_directory);
  if (!index) {
    return unexpected(index.error());
  }
  const bool one_record_fits = std::any_of(
      index->records.begin(), index->records.end(),
      [fast_capacity_bytes](const RecordMetadata& metadata) {
        return metadata.byte_length <= fast_capacity_bytes;
      });
  if (!one_record_fits) {
    return unexpected(make_error(
        StoreErrorCode::invalid_configuration,
        "fast-tier capacity cannot hold any record", std::nullopt,
        std::nullopt, store_directory));
  }
  const int descriptor = ::open(index->data_path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not open authoritative data file",
                                 std::nullopt, std::nullopt, index->data_path));
  }
  try {
    auto store = std::unique_ptr<TieredStore>(
        new TieredStore(std::move(*index), descriptor, fast_capacity_bytes));
    return store;
  } catch (const std::bad_alloc&) {
    ::close(descriptor);
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate tiered-store state",
                                 std::nullopt, std::nullopt, store_directory));
  }
}

TieredStore::TieredStore(LoadedStoreIndex index, int data_descriptor,
                         std::uint64_t fast_capacity_bytes)
    : index_(std::move(index)),
      data_descriptor_(data_descriptor),
      fast_capacity_bytes_(fast_capacity_bytes) {
  for (const auto& metadata : index_.records) {
    metadata_by_id_.emplace(metadata.record_id, &metadata);
  }
  update_residency_stats();
}

TieredStore::~TieredStore() {
  if (data_descriptor_ >= 0) {
    ::close(data_descriptor_);
  }
}

const RecordMetadata* TieredStore::find_metadata(RecordId id) const noexcept {
  const auto found = metadata_by_id_.find(id);
  return found == metadata_by_id_.end() ? nullptr : found->second;
}

Expected<RecordMetadata> TieredStore::metadata(RecordId id) const {
  const auto* found = find_metadata(id);
  if (found == nullptr) {
    return unexpected(make_error(StoreErrorCode::unknown_record,
                                 "record ID is not present in the store", id));
  }
  return *found;
}

Expected<std::vector<std::byte>> TieredStore::read_slow(
    const RecordMetadata& metadata) const {
  if (metadata.byte_length > std::numeric_limits<std::size_t>::max() ||
      metadata.byte_offset >
          static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    return unexpected(make_error(
        StoreErrorCode::arithmetic_overflow,
        "record range cannot be represented by this platform",
        metadata.record_id, metadata.byte_offset, index_.data_path));
  }
  std::vector<std::byte> payload;
  try {
    payload.resize(static_cast<std::size_t>(metadata.byte_length));
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate record destination",
                                 metadata.record_id, metadata.byte_offset,
                                 index_.data_path));
  }
  std::size_t consumed = 0;
  while (consumed < payload.size()) {
    const std::size_t remaining = payload.size() - consumed;
    const std::size_t requested = std::min<std::size_t>(
        remaining, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    const std::uint64_t absolute_offset =
        metadata.byte_offset + static_cast<std::uint64_t>(consumed);
    const ssize_t count = ::pread(data_descriptor_, payload.data() + consumed,
                                  requested,
                                  static_cast<off_t>(absolute_offset));
    if (count < 0) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "offset-based data read failed",
                                   metadata.record_id, absolute_offset,
                                   index_.data_path));
    }
    if (count == 0) {
      return unexpected(make_error(StoreErrorCode::truncated_read,
                                   "data file ended before the record was complete",
                                   metadata.record_id, absolute_offset,
                                   index_.data_path));
    }
    consumed += static_cast<std::size_t>(count);
  }
  if (compute_crc32(payload) != metadata.crc32) {
    payload.clear();
    return unexpected(make_error(StoreErrorCode::checksum_mismatch,
                                 "record checksum does not match metadata",
                                 metadata.record_id, metadata.byte_offset,
                                 index_.data_path));
  }
  return payload;
}

StoreError TieredStore::record_failure(StoreError error) {
  const auto index = static_cast<std::size_t>(error.code);
  if (index < stats_.failures_by_code.size()) {
    ++stats_.failures_by_code[index];
  }
  return error;
}

void TieredStore::update_residency_stats() noexcept {
  stats_.current_resident_records =
      static_cast<std::uint64_t>(resident_.size());
  stats_.current_resident_bytes = resident_bytes_;
  stats_.resident_byte_high_water_mark =
      std::max(stats_.resident_byte_high_water_mark, resident_bytes_);
}

Expected<ReadInfo> TieredStore::read_into(
    RecordId id, std::vector<std::byte>& destination) {
  destination.clear();
  const auto* record = find_metadata(id);
  if (record == nullptr) {
    return unexpected(record_failure(make_error(
        StoreErrorCode::unknown_record,
        "record ID is not present in the store", id)));
  }
  const auto resident = resident_.find(id);
  if (resident != resident_.end()) {
    try {
      destination.assign(resident->second.begin(), resident->second.end());
    } catch (const std::bad_alloc&) {
      destination.clear();
      return unexpected(record_failure(make_error(
          StoreErrorCode::allocation_failure,
          "could not allocate caller read destination", id)));
    }
    ++stats_.successful_fast_reads;
    stats_.successful_fast_read_bytes += record->byte_length;
    return ReadInfo{ReadTier::fast, record->byte_length};
  }
  auto payload = read_slow(*record);
  if (!payload) {
    destination.clear();
    return unexpected(record_failure(payload.error()));
  }
  destination = std::move(*payload);
  ++stats_.successful_slow_reads;
  stats_.successful_slow_read_bytes += record->byte_length;
  return ReadInfo{ReadTier::slow, record->byte_length};
}

Expected<PromotionResult> TieredStore::promote(RecordId id) {
  const auto* record = find_metadata(id);
  if (record == nullptr) {
    return unexpected(record_failure(make_error(
        StoreErrorCode::unknown_record,
        "record ID is not present in the store", id)));
  }
  if (resident_.find(id) != resident_.end()) {
    return PromotionResult{true, 0};
  }
  if (record->byte_length > fast_capacity_bytes_) {
    return unexpected(record_failure(make_error(
        StoreErrorCode::oversized_record,
        "record is larger than total fast-tier capacity", id)));
  }
  if (record->byte_length > fast_capacity_bytes_ - resident_bytes_) {
    return unexpected(record_failure(make_error(
        StoreErrorCode::insufficient_capacity,
        "remaining fast-tier capacity is insufficient", id)));
  }
  auto payload = read_slow(*record);
  if (!payload) {
    return unexpected(record_failure(payload.error()));
  }
  ++stats_.promotion_source_reads;
  stats_.promotion_source_read_bytes += record->byte_length;
  try {
    resident_.emplace(id, std::move(*payload));
  } catch (const std::bad_alloc&) {
    return unexpected(record_failure(make_error(
        StoreErrorCode::allocation_failure,
        "could not install promoted record", id)));
  }
  resident_bytes_ += record->byte_length;
  ++stats_.committed_promotions;
  stats_.committed_promotion_bytes += record->byte_length;
  update_residency_stats();
  return PromotionResult{false, record->byte_length};
}

Expected<EvictionResult> TieredStore::evict(RecordId id) {
  const auto* record = find_metadata(id);
  if (record == nullptr) {
    return unexpected(record_failure(make_error(
        StoreErrorCode::unknown_record,
        "record ID is not present in the store", id)));
  }
  const auto resident = resident_.find(id);
  if (resident == resident_.end()) {
    return EvictionResult{false, 0};
  }
  resident_.erase(resident);
  resident_bytes_ -= record->byte_length;
  ++stats_.committed_evictions;
  stats_.committed_eviction_bytes += record->byte_length;
  update_residency_stats();
  return EvictionResult{true, record->byte_length};
}

Expected<TargetSetResult> TieredStore::apply_target_set(
    const std::vector<RecordId>& target_record_ids) {
  ++stats_.target_set_calls;
  std::uint64_t staged_bytes = 0;
  const auto fail = [this, &staged_bytes](StoreError error)
      -> Expected<TargetSetResult> {
    ++stats_.failed_target_set_calls;
    stats_.aborted_staged_bytes += staged_bytes;
    return unexpected(record_failure(std::move(error)));
  };

  std::vector<RecordId> target;
  try {
    target = target_record_ids;
    std::sort(target.begin(), target.end());
  } catch (const std::bad_alloc&) {
    return fail(make_error(StoreErrorCode::allocation_failure,
                           "could not allocate target-set preflight state"));
  }
  if (std::adjacent_find(target.begin(), target.end()) != target.end()) {
    return fail(make_error(StoreErrorCode::invalid_target_set,
                           "target set contains a duplicate record ID"));
  }

  std::uint64_t total_target_bytes = 0;
  for (const RecordId id : target) {
    const auto* record = find_metadata(id);
    if (record == nullptr) {
      return fail(make_error(StoreErrorCode::unknown_record,
                             "target set contains an unknown record ID", id));
    }
    if (record->byte_length > fast_capacity_bytes_) {
      return fail(make_error(StoreErrorCode::oversized_record,
                             "target set contains an oversized record", id));
    }
    if (record->byte_length >
        std::numeric_limits<std::uint64_t>::max() - total_target_bytes) {
      return fail(make_error(StoreErrorCode::arithmetic_overflow,
                             "target-set byte total overflows", id));
    }
    total_target_bytes += record->byte_length;
  }
  if (total_target_bytes > fast_capacity_bytes_) {
    return fail(make_error(StoreErrorCode::insufficient_capacity,
                           "target set exceeds fast-tier capacity"));
  }

  std::vector<RecordId> current;
  try {
    current.reserve(resident_.size());
    for (const auto& entry : resident_) {
      current.push_back(entry.first);
    }
  } catch (const std::bad_alloc&) {
    return fail(make_error(StoreErrorCode::allocation_failure,
                           "could not allocate current residency state"));
  }
  if (current == target) {
    ++stats_.successful_target_set_calls;
    return TargetSetResult{0, 0, 0, 0, residency_snapshot()};
  }

  std::vector<RecordId> incoming;
  std::vector<RecordId> outgoing;
  try {
    std::set_difference(target.begin(), target.end(), current.begin(), current.end(),
                        std::back_inserter(incoming));
    std::set_difference(current.begin(), current.end(), target.begin(), target.end(),
                        std::back_inserter(outgoing));
  } catch (const std::bad_alloc&) {
    return fail(make_error(StoreErrorCode::allocation_failure,
                           "could not allocate target-set difference state"));
  }

  std::map<RecordId, std::vector<std::byte>> staged;
  for (const RecordId id : incoming) {
    const auto* record = find_metadata(id);
    auto payload = read_slow(*record);
    if (!payload) {
      return fail(payload.error());
    }
    ++stats_.promotion_source_reads;
    stats_.promotion_source_read_bytes += record->byte_length;
    staged_bytes += record->byte_length;
    try {
      staged.emplace(id, std::move(*payload));
    } catch (const std::bad_alloc&) {
      return fail(make_error(StoreErrorCode::allocation_failure,
                             "could not retain staged target record", id));
    }
  }

  std::map<RecordId, std::vector<std::byte>> next;
  try {
    for (const RecordId id : target) {
      const auto resident = resident_.find(id);
      if (resident != resident_.end()) {
        next.emplace(id, resident->second);
      } else {
        auto staged_record = staged.find(id);
        next.emplace(id, std::move(staged_record->second));
      }
    }
  } catch (const std::bad_alloc&) {
    return fail(make_error(StoreErrorCode::allocation_failure,
                           "could not allocate committed target state"));
  }

  std::uint64_t incoming_bytes = 0;
  for (const RecordId id : incoming) {
    incoming_bytes += find_metadata(id)->byte_length;
  }
  std::uint64_t outgoing_bytes = 0;
  for (const RecordId id : outgoing) {
    outgoing_bytes += find_metadata(id)->byte_length;
  }
  resident_.swap(next);
  resident_bytes_ = total_target_bytes;
  stats_.committed_promotions += static_cast<std::uint64_t>(incoming.size());
  stats_.committed_promotion_bytes += incoming_bytes;
  stats_.committed_evictions += static_cast<std::uint64_t>(outgoing.size());
  stats_.committed_eviction_bytes += outgoing_bytes;
  ++stats_.successful_target_set_calls;
  update_residency_stats();
  return TargetSetResult{static_cast<std::uint64_t>(incoming.size()),
                         incoming_bytes,
                         static_cast<std::uint64_t>(outgoing.size()),
                         outgoing_bytes, residency_snapshot()};
}

ResidencySnapshot TieredStore::residency_snapshot() const {
  ResidencySnapshot snapshot{{}, resident_bytes_, fast_capacity_bytes_};
  snapshot.resident_record_ids.reserve(resident_.size());
  for (const auto& entry : resident_) {
    snapshot.resident_record_ids.push_back(entry.first);
  }
  return snapshot;
}

StoreStats TieredStore::stats() const { return stats_; }

}  // namespace prism::storage
