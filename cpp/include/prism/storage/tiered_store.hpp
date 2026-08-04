#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <vector>

#include "prism/storage/error.hpp"
#include "prism/storage/store_format.hpp"

namespace prism::storage {

enum class ReadTier { fast, slow };

struct ReadInfo {
  ReadTier tier;
  std::uint64_t bytes;
};

struct PromotionResult {
  bool already_resident;
  std::uint64_t bytes_moved;
};

struct EvictionResult {
  bool was_resident;
  std::uint64_t bytes_freed;
};

struct ResidencySnapshot {
  std::vector<RecordId> resident_record_ids;
  std::uint64_t resident_bytes;
  std::uint64_t capacity_bytes;

  friend bool operator==(const ResidencySnapshot& left,
                         const ResidencySnapshot& right) {
    return left.resident_record_ids == right.resident_record_ids &&
           left.resident_bytes == right.resident_bytes &&
           left.capacity_bytes == right.capacity_bytes;
  }
};

struct TargetSetResult {
  std::uint64_t promotion_count;
  std::uint64_t promotion_bytes;
  std::uint64_t eviction_count;
  std::uint64_t eviction_bytes;
  ResidencySnapshot residency;
};

struct StoreStats {
  std::uint64_t successful_fast_reads = 0;
  std::uint64_t successful_fast_read_bytes = 0;
  std::uint64_t successful_slow_reads = 0;
  std::uint64_t successful_slow_read_bytes = 0;
  std::uint64_t promotion_source_reads = 0;
  std::uint64_t promotion_source_read_bytes = 0;
  std::uint64_t committed_promotions = 0;
  std::uint64_t committed_promotion_bytes = 0;
  std::uint64_t committed_evictions = 0;
  std::uint64_t committed_eviction_bytes = 0;
  std::uint64_t target_set_calls = 0;
  std::uint64_t successful_target_set_calls = 0;
  std::uint64_t failed_target_set_calls = 0;
  std::uint64_t aborted_staged_bytes = 0;
  std::uint64_t current_resident_records = 0;
  std::uint64_t current_resident_bytes = 0;
  std::uint64_t resident_byte_high_water_mark = 0;
  std::array<std::uint64_t, store_error_code_count()> failures_by_code{};

  [[nodiscard]] std::uint64_t failure_count(StoreErrorCode code) const noexcept;
};

class TieredStore {
 public:
  [[nodiscard]] static Expected<std::unique_ptr<TieredStore>> open(
      const std::filesystem::path& store_directory,
      std::uint64_t fast_capacity_bytes);

  TieredStore(const TieredStore&) = delete;
  TieredStore& operator=(const TieredStore&) = delete;
  TieredStore(TieredStore&&) = delete;
  TieredStore& operator=(TieredStore&&) = delete;
  ~TieredStore();

  [[nodiscard]] Expected<RecordMetadata> metadata(RecordId id) const;
  [[nodiscard]] Expected<ReadInfo> read_into(
      RecordId id, std::vector<std::byte>& destination);
  [[nodiscard]] Expected<PromotionResult> promote(RecordId id);
  [[nodiscard]] Expected<EvictionResult> evict(RecordId id);
  [[nodiscard]] Expected<TargetSetResult> apply_target_set(
      const std::vector<RecordId>& target_record_ids);
  [[nodiscard]] ResidencySnapshot residency_snapshot() const;
  [[nodiscard]] StoreStats stats() const;

 private:
  TieredStore(LoadedStoreIndex index, int data_descriptor,
              std::uint64_t fast_capacity_bytes);

  [[nodiscard]] const RecordMetadata* find_metadata(RecordId id) const noexcept;
  [[nodiscard]] Expected<std::vector<std::byte>> read_slow(
      const RecordMetadata& metadata) const;
  [[nodiscard]] StoreError record_failure(StoreError error);
  void update_residency_stats() noexcept;

  LoadedStoreIndex index_;
  int data_descriptor_;
  std::uint64_t fast_capacity_bytes_;
  std::uint64_t resident_bytes_ = 0;
  std::map<RecordId, const RecordMetadata*> metadata_by_id_;
  std::map<RecordId, std::vector<std::byte>> resident_;
  StoreStats stats_;
};

}  // namespace prism::storage
