#include "prism/storage/tooling.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "prism/storage/store_format.hpp"
#include "prism/storage/tiered_store.hpp"

namespace prism::storage {
namespace {

using Json = nlohmann::ordered_json;

StoreError make_error(StoreErrorCode code, std::string message,
                      std::optional<std::filesystem::path> path = std::nullopt) {
  return StoreError{code, std::move(message), std::nullopt, std::nullopt,
                    std::move(path)};
}

Json snapshot_json(const ResidencySnapshot& snapshot) {
  return Json{{"resident_record_ids", snapshot.resident_record_ids},
              {"resident_bytes", snapshot.resident_bytes},
              {"capacity_bytes", snapshot.capacity_bytes}};
}

Json stats_json(const StoreStats& stats) {
  Json failures = Json::object();
  for (std::size_t index = 0; index < store_error_code_count(); ++index) {
    const auto code = static_cast<StoreErrorCode>(index);
    failures[to_string(code)] = stats.failures_by_code[index];
  }
  return Json{
      {"successful_fast_reads", stats.successful_fast_reads},
      {"successful_fast_read_bytes", stats.successful_fast_read_bytes},
      {"successful_slow_reads", stats.successful_slow_reads},
      {"successful_slow_read_bytes", stats.successful_slow_read_bytes},
      {"promotion_source_reads", stats.promotion_source_reads},
      {"promotion_source_read_bytes", stats.promotion_source_read_bytes},
      {"committed_promotions", stats.committed_promotions},
      {"committed_promotion_bytes", stats.committed_promotion_bytes},
      {"committed_evictions", stats.committed_evictions},
      {"committed_eviction_bytes", stats.committed_eviction_bytes},
      {"target_set_calls", stats.target_set_calls},
      {"successful_target_set_calls", stats.successful_target_set_calls},
      {"failed_target_set_calls", stats.failed_target_set_calls},
      {"aborted_staged_bytes", stats.aborted_staged_bytes},
      {"current_resident_records", stats.current_resident_records},
      {"current_resident_bytes", stats.current_resident_bytes},
      {"resident_byte_high_water_mark", stats.resident_byte_high_water_mark},
      {"failures_by_code", std::move(failures)},
  };
}

bool has_only_keys(const Json& object, const std::set<std::string>& allowed) {
  for (auto entry = object.begin(); entry != object.end(); ++entry) {
    if (allowed.find(entry.key()) == allowed.end()) {
      return false;
    }
  }
  return true;
}

Expected<bool> validate_expected(const Json& expected,
                                 const std::filesystem::path& trace_path) {
  static const std::set<std::string> allowed{
      "success", "error_code", "read_tier", "bytes",
      "resident_record_ids"};
  if (!expected.is_object() || !has_only_keys(expected, allowed)) {
    return unexpected(make_error(StoreErrorCode::malformed_trace,
                                 "expected result contains unknown fields",
                                 trace_path));
  }
  if (expected.contains("success") && !expected["success"].is_boolean()) {
    return unexpected(make_error(StoreErrorCode::malformed_trace,
                                 "expected.success must be boolean", trace_path));
  }
  for (const char* key : {"error_code", "read_tier"}) {
    if (expected.contains(key) && !expected[key].is_string()) {
      return unexpected(make_error(StoreErrorCode::malformed_trace,
                                   std::string("expected.") + key +
                                       " must be a string",
                                   trace_path));
    }
  }
  if (expected.contains("bytes") && !expected["bytes"].is_number_unsigned()) {
    return unexpected(make_error(StoreErrorCode::malformed_trace,
                                 "expected.bytes must be unsigned", trace_path));
  }
  if (expected.contains("resident_record_ids")) {
    const auto& values = expected["resident_record_ids"];
    if (!values.is_array() ||
        !std::all_of(values.begin(), values.end(),
                     [](const Json& value) { return value.is_number_unsigned(); })) {
      return unexpected(make_error(
          StoreErrorCode::malformed_trace,
          "expected.resident_record_ids must be an unsigned array", trace_path));
    }
  }
  return true;
}

void compare_expected(std::uint64_t sequence, const Json& expected,
                      const Json& operation, Json& mismatches) {
  const auto compare = [&](const std::string& expected_key,
                           const std::string& actual_key) {
    if (expected.contains(expected_key) &&
        expected[expected_key] != operation[actual_key]) {
      mismatches.push_back(Json{{"sequence", sequence},
                                {"field", expected_key},
                                {"expected", expected[expected_key]},
                                {"actual", operation[actual_key]}});
    }
  };
  compare("success", "success");
  compare("error_code", "error_code");
  compare("read_tier", "read_tier");
  compare("bytes", "bytes_read");
  if (expected.contains("resident_record_ids") &&
      expected["resident_record_ids"] !=
          operation["residency"]["resident_record_ids"]) {
    mismatches.push_back(
        Json{{"sequence", sequence},
             {"field", "resident_record_ids"},
             {"expected", expected["resident_record_ids"]},
             {"actual", operation["residency"]["resident_record_ids"]}});
  }
}

Expected<std::vector<RecordId>> record_ids(const Json& row,
                                           const std::filesystem::path& path) {
  if (!row.contains("record_ids") || !row["record_ids"].is_array()) {
    return unexpected(make_error(StoreErrorCode::malformed_trace,
                                 "apply_target_set requires record_ids array", path));
  }
  std::vector<RecordId> result;
  try {
    result.reserve(row["record_ids"].size());
    for (const auto& value : row["record_ids"]) {
      if (!value.is_number_unsigned()) {
        return unexpected(make_error(StoreErrorCode::malformed_trace,
                                     "record_ids values must be unsigned", path));
      }
      result.push_back(value.get<RecordId>());
    }
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate replay target", path));
  }
  return result;
}

}  // namespace

Expected<std::string> inspect_store_json(
    const std::filesystem::path& store_directory, bool verify_all,
    std::optional<std::uint64_t> capacity_bytes) {
  if (capacity_bytes && *capacity_bytes == 0) {
    return unexpected(make_error(StoreErrorCode::invalid_configuration,
                                 "requested capacity must be positive",
                                 store_directory));
  }
  auto index = load_store_index(store_directory);
  if (!index) {
    return unexpected(index.error());
  }
  std::vector<std::uint64_t> sizes;
  try {
    sizes.reserve(index->records.size());
    for (const auto& record : index->records) {
      sizes.push_back(record.byte_length);
    }
    std::sort(sizes.begin(), sizes.end());
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate inspection state",
                                 store_directory));
  }
  std::uint64_t verified_count = 0;
  if (verify_all) {
    auto verified = verify_all_records(*index);
    if (!verified) {
      return unexpected(verified.error());
    }
    verified_count = *verified;
  }
  double median = 0.0;
  const std::size_t middle = sizes.size() / 2U;
  if (sizes.size() % 2U == 1U) {
    median = static_cast<double>(sizes[middle]);
  } else {
    median = static_cast<double>(sizes[middle - 1U]) / 2.0 +
             static_cast<double>(sizes[middle]) / 2.0;
  }
  Json capacity = nullptr;
  if (capacity_bytes) {
    const std::uint64_t fitting = static_cast<std::uint64_t>(std::count_if(
        sizes.begin(), sizes.end(),
        [&](std::uint64_t size) { return size <= *capacity_bytes; }));
    capacity = Json{{"capacity_bytes", *capacity_bytes},
                    {"feasible", fitting > 0},
                    {"records_that_fit", fitting}};
  }
  const Json report = {
      {"format_version", index->format_version},
      {"file_identifier", kStoreFileIdentifier},
      {"record_count", static_cast<std::uint64_t>(index->records.size())},
      {"total_data_bytes", index->data_file_length},
      {"minimum_record_bytes", sizes.front()},
      {"median_record_bytes", median},
      {"maximum_record_bytes", sizes.back()},
      {"index_valid", true},
      {"ranges_valid", true},
      {"verify_all_requested", verify_all},
      {"verified_record_count", verified_count},
      {"all_checksums_valid", verify_all ? Json(true) : Json(nullptr)},
      {"capacity", std::move(capacity)},
  };
  return report.dump(2) + "\n";
}

Expected<ReplayReport> replay_trace_json(
    const std::filesystem::path& store_directory,
    std::uint64_t capacity_bytes, const std::filesystem::path& trace_path) {
  auto store = TieredStore::open(store_directory, capacity_bytes);
  if (!store) {
    return unexpected(store.error());
  }
  std::ifstream trace(trace_path);
  if (!trace) {
    return unexpected(make_error(StoreErrorCode::malformed_trace,
                                 "could not open replay trace", trace_path));
  }
  Json operations = Json::array();
  Json mismatches = Json::array();
  std::optional<std::uint64_t> previous_sequence;
  std::string line;
  try {
    while (std::getline(trace, line)) {
      if (line.empty()) {
        return unexpected(make_error(StoreErrorCode::malformed_trace,
                                     "replay trace contains an empty line",
                                     trace_path));
      }
      Json row;
      try {
        row = Json::parse(line);
      } catch (const nlohmann::json::exception&) {
        return unexpected(make_error(StoreErrorCode::malformed_trace,
                                     "replay trace contains malformed JSON",
                                     trace_path));
      }
      if (!row.is_object() || !row.contains("sequence") ||
          !row["sequence"].is_number_unsigned() ||
          !row.contains("operation") || !row["operation"].is_string()) {
        return unexpected(make_error(
            StoreErrorCode::malformed_trace,
            "each replay row requires unsigned sequence and string operation",
            trace_path));
      }
      const std::uint64_t sequence = row["sequence"].get<std::uint64_t>();
      if (previous_sequence && sequence <= *previous_sequence) {
        return unexpected(make_error(
            StoreErrorCode::malformed_trace,
            "replay sequence numbers must be strictly increasing", trace_path));
      }
      previous_sequence = sequence;
      const std::string operation_name = row["operation"].get<std::string>();
      std::set<std::string> allowed{"sequence", "operation", "expected"};
      if (operation_name == "read" || operation_name == "promote" ||
          operation_name == "evict") {
        allowed.insert("record_id");
      } else if (operation_name == "apply_target_set") {
        allowed.insert("record_ids");
      } else if (operation_name != "snapshot") {
        return unexpected(make_error(StoreErrorCode::malformed_trace,
                                     "replay operation is unsupported", trace_path));
      }
      if (!has_only_keys(row, allowed)) {
        return unexpected(make_error(StoreErrorCode::malformed_trace,
                                     "replay operation contains unknown fields",
                                     trace_path));
      }
      if (row.contains("expected")) {
        auto valid_expected = validate_expected(row["expected"], trace_path);
        if (!valid_expected) {
          return unexpected(valid_expected.error());
        }
      }
      if ((operation_name == "read" || operation_name == "promote" ||
           operation_name == "evict") &&
          (!row.contains("record_id") ||
           !row["record_id"].is_number_unsigned())) {
        return unexpected(make_error(StoreErrorCode::malformed_trace,
                                     "operation requires unsigned record_id",
                                     trace_path));
      }

      const StoreStats before = (*store)->stats();
      bool success = true;
      std::optional<StoreError> operation_error;
      Json read_tier = nullptr;
      Json bytes_read = nullptr;
      Json payload_crc = nullptr;
      std::vector<std::byte> destination;
      if (operation_name == "read") {
        auto result = (*store)->read_into(row["record_id"].get<RecordId>(),
                                         destination);
        if (result) {
          read_tier = result->tier == ReadTier::fast ? "fast" : "slow";
          bytes_read = result->bytes;
          payload_crc = compute_crc32(destination);
        } else {
          success = false;
          operation_error = result.error();
        }
      } else if (operation_name == "promote") {
        auto result = (*store)->promote(row["record_id"].get<RecordId>());
        if (!result) {
          success = false;
          operation_error = result.error();
        }
      } else if (operation_name == "evict") {
        auto result = (*store)->evict(row["record_id"].get<RecordId>());
        if (!result) {
          success = false;
          operation_error = result.error();
        }
      } else if (operation_name == "apply_target_set") {
        auto ids = record_ids(row, trace_path);
        if (!ids) {
          return unexpected(ids.error());
        }
        auto result = (*store)->apply_target_set(*ids);
        if (!result) {
          success = false;
          operation_error = result.error();
        }
      }
      const StoreStats after = (*store)->stats();
      const auto residency = (*store)->residency_snapshot();
      Json operation = {
          {"sequence", sequence},
          {"operation", operation_name},
          {"success", success},
          {"error_code",
           operation_error ? Json(to_string(operation_error->code)) : Json(nullptr)},
          {"read_tier", std::move(read_tier)},
          {"bytes_read", std::move(bytes_read)},
          {"payload_crc32", std::move(payload_crc)},
          {"movement",
           Json{{"promotion_count",
                 after.committed_promotions - before.committed_promotions},
                {"promotion_bytes",
                 after.committed_promotion_bytes -
                     before.committed_promotion_bytes},
                {"eviction_count",
                 after.committed_evictions - before.committed_evictions},
                {"eviction_bytes",
                 after.committed_eviction_bytes -
                     before.committed_eviction_bytes}}},
          {"residency", snapshot_json(residency)},
          {"cumulative_stats", stats_json(after)},
      };
      const std::size_t mismatches_before = mismatches.size();
      if (row.contains("expected")) {
        compare_expected(sequence, row["expected"], operation, mismatches);
      }
      operation["expected_outcome_matches"] =
          !row.contains("expected") || mismatches.size() == mismatches_before;
      operations.push_back(std::move(operation));
    }
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate replay report", trace_path));
  } catch (const nlohmann::json::exception&) {
    return unexpected(make_error(StoreErrorCode::malformed_trace,
                                 "replay value is outside the supported range",
                                 trace_path));
  }
  if (trace.bad()) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not read replay trace", trace_path));
  }
  const auto final_residency = (*store)->residency_snapshot();
  const auto final_stats = (*store)->stats();
  const Json report = {
      {"operations", std::move(operations)},
      {"final_residency", snapshot_json(final_residency)},
      {"final_stats", stats_json(final_stats)},
      {"capacity_invariant",
       final_residency.resident_bytes <= final_residency.capacity_bytes},
      {"expected_outcome_mismatches", std::move(mismatches)},
  };
  return ReplayReport{report.dump(2) + "\n",
                      static_cast<std::uint64_t>(
                          report["expected_outcome_mismatches"].size())};
}

Expected<bool> write_deterministic_report(
    const std::filesystem::path& output_path, const std::string& contents) {
  if (contents.size() >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    return unexpected(make_error(StoreErrorCode::arithmetic_overflow,
                                 "report is too large for stream output",
                                 output_path));
  }
  std::ofstream stream(output_path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not create report", output_path));
  }
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  stream.flush();
  if (!stream) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not write complete report", output_path));
  }
  return true;
}

}  // namespace prism::storage
