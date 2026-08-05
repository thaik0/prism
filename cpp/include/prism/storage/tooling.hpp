#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "prism/storage/error.hpp"

namespace prism::storage {

struct ReplayReport {
  std::string json;
  std::uint64_t expected_outcome_mismatch_count;
};

[[nodiscard]] Expected<std::string> inspect_store_json(
    const std::filesystem::path& store_directory, bool verify_all,
    std::optional<std::uint64_t> capacity_bytes = std::nullopt);

[[nodiscard]] Expected<ReplayReport> replay_trace_json(
    const std::filesystem::path& store_directory,
    std::uint64_t capacity_bytes,
    const std::filesystem::path& trace_path);

[[nodiscard]] Expected<bool> write_deterministic_report(
    const std::filesystem::path& output_path, const std::string& contents);

}  // namespace prism::storage
