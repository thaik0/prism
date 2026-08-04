#pragma once

#include <cstdint>
#include <filesystem>

#include "prism/storage/error.hpp"

namespace prism::storage {

struct BuildResult {
  std::uint64_t record_count;
  std::uint64_t data_bytes;
};

[[nodiscard]] Expected<BuildResult> build_store(
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& output_directory);

}  // namespace prism::storage
