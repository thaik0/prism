#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "prism/storage/error.hpp"

namespace prism::storage {

struct BuildResult {
  std::uint64_t record_count;
  std::uint64_t data_bytes;
};

struct BuildRecord {
  RecordId record_id;
  std::vector<std::byte> payload;
};

[[nodiscard]] Expected<BuildResult> build_store(
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& output_directory);

[[nodiscard]] Expected<BuildResult> build_store(
    std::vector<BuildRecord> records,
    const std::filesystem::path& output_directory);

}  // namespace prism::storage
