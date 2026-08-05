#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "prism/storage/error.hpp"

namespace prism::storage {

inline constexpr std::uint32_t kStoreFormatVersion = 1;
inline constexpr char kStoreFileIdentifier[] = "PRSM";
inline constexpr char kStoreDataFilename[] = "store.data";
inline constexpr char kStoreIndexFilename[] = "store.index";

struct RecordMetadata {
  RecordId record_id;
  std::uint64_t byte_offset;
  std::uint64_t byte_length;
  std::uint32_t crc32;

  friend bool operator==(const RecordMetadata& left,
                         const RecordMetadata& right) {
    return left.record_id == right.record_id &&
           left.byte_offset == right.byte_offset &&
           left.byte_length == right.byte_length && left.crc32 == right.crc32;
  }
};

struct LoadedStoreIndex {
  std::filesystem::path store_directory;
  std::filesystem::path data_path;
  std::filesystem::path index_path;
  std::uint32_t format_version;
  std::uint64_t data_file_length;
  std::vector<RecordMetadata> records;
};

[[nodiscard]] Expected<LoadedStoreIndex> load_store_index(
    const std::filesystem::path& store_directory);

[[nodiscard]] Expected<bool> validate_store_metadata(
    const std::vector<RecordMetadata>& records,
    std::uint64_t declared_data_length,
    std::uint64_t actual_data_length,
    const std::filesystem::path& context_path = {});

[[nodiscard]] Expected<std::vector<std::byte>> read_record_payload(
    const std::filesystem::path& data_path,
    const RecordMetadata& metadata,
    bool verify_crc = true);

[[nodiscard]] Expected<std::uint64_t> verify_all_records(
    const LoadedStoreIndex& index);

[[nodiscard]] std::uint32_t compute_crc32(
    const std::vector<std::byte>& payload) noexcept;

}  // namespace prism::storage
