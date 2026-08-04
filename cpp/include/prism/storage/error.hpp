#pragma once

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>

#include <tl/expected.hpp>

namespace prism::storage {

using RecordId = std::uint64_t;

enum class StoreErrorCode {
  unknown_record,
  duplicate_record,
  invalid_target_set,
  insufficient_capacity,
  oversized_record,
  index_corrupt,
  unsupported_format_version,
  data_file_mismatch,
  truncated_read,
  checksum_mismatch,
  io_error,
  allocation_failure,
  arithmetic_overflow,
  invalid_configuration,
  destination_exists,
  malformed_manifest,
  count,
};

struct StoreError {
  StoreErrorCode code;
  std::string message;
  std::optional<RecordId> record_id;
  std::optional<std::uint64_t> byte_offset;
  std::optional<std::filesystem::path> path;
};

template <typename T>
using Expected = tl::expected<T, StoreError>;

inline tl::unexpected<StoreError> unexpected(StoreError error) {
  return tl::make_unexpected(std::move(error));
}

[[nodiscard]] const char* to_string(StoreErrorCode code) noexcept;
[[nodiscard]] constexpr std::size_t store_error_code_count() noexcept {
  return static_cast<std::size_t>(StoreErrorCode::count);
}

}  // namespace prism::storage
