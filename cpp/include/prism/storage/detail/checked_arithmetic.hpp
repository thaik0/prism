#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <utility>

#include "prism/storage/error.hpp"

namespace prism::storage::detail {

inline Expected<std::uint64_t> checked_add_bytes(
    std::uint64_t left, std::uint64_t right,
    std::optional<RecordId> record_id = std::nullopt,
    std::optional<std::filesystem::path> path = std::nullopt) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return unexpected(StoreError{StoreErrorCode::arithmetic_overflow,
                                 "byte offset addition overflows", record_id,
                                 left, std::move(path)});
  }
  return left + right;
}

}  // namespace prism::storage::detail
