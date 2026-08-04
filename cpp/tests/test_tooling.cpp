#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "prism/storage/builder.hpp"
#include "prism/storage/store_format.hpp"
#include "prism/storage/tooling.hpp"

namespace {

std::filesystem::path build_store(const std::string& name) {
  auto directory = std::filesystem::temp_directory_path() / name;
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
  REQUIRE(prism::storage::build_store(
      std::filesystem::path(PRISM_TEST_SOURCE_DIR) / "fixtures" /
          "store_manifest.json",
      directory));
  return directory;
}

std::filesystem::path write_trace(const std::string& name,
                                  const std::string& contents) {
  auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream << contents;
  return path;
}

const auto kReplayTrace = std::filesystem::path(PRISM_TEST_SOURCE_DIR) /
                          "fixtures" / "replay.jsonl";

}  // namespace

TEST_CASE("inspection reports deterministic structure, sizes, capacity, and checksums") {
  const auto store = build_store("prism_tool_inspect");
  auto first = prism::storage::inspect_store_json(store, true, 13);
  auto second = prism::storage::inspect_store_json(store, true, 13);
  REQUIRE(first);
  REQUIRE(second);
  CHECK(*first == *second);
  const auto report = nlohmann::json::parse(*first);
  CHECK(report["format_version"] == 1);
  CHECK(report["file_identifier"] == "PRSM");
  CHECK(report["record_count"] == 3);
  CHECK(report["total_data_bytes"] == 40);
  CHECK(report["minimum_record_bytes"] == 6);
  CHECK(report["median_record_bytes"] == 13.0);
  CHECK(report["maximum_record_bytes"] == 21);
  CHECK(report["index_valid"]);
  CHECK(report["ranges_valid"]);
  CHECK(report["verified_record_count"] == 3);
  CHECK(report["all_checksums_valid"]);
  CHECK(report["capacity"]["feasible"]);
  CHECK(report["capacity"]["records_that_fit"] == 2);
}

TEST_CASE("full inspection returns structured payload corruption") {
  const auto store = build_store("prism_tool_inspect_corrupt");
  auto loaded = prism::storage::load_store_index(store);
  REQUIRE(loaded);
  std::fstream data(loaded->data_path,
                    std::ios::binary | std::ios::in | std::ios::out);
  char value = 0;
  data.read(&value, 1);
  value ^= 1;
  data.seekp(0);
  data.write(&value, 1);
  data.flush();
  auto report = prism::storage::inspect_store_json(store, true);
  REQUIRE_FALSE(report);
  CHECK(report.error().code ==
        prism::storage::StoreErrorCode::checksum_mismatch);
}

TEST_CASE("replay supports every operation with exact final counters") {
  const auto store = build_store("prism_tool_replay");
  auto first = prism::storage::replay_trace_json(store, 40, kReplayTrace);
  auto second = prism::storage::replay_trace_json(store, 40, kReplayTrace);
  REQUIRE(first);
  REQUIRE(second);
  CHECK(first->json == second->json);
  CHECK(first->expected_outcome_mismatch_count == 0);
  const auto report = nlohmann::json::parse(first->json);
  REQUIRE(report["operations"].size() == 7);
  CHECK(report["operations"][0]["read_tier"] == "slow");
  CHECK(report["operations"][2]["read_tier"] == "fast");
  CHECK(report["operations"][0]["payload_crc32"].is_number_unsigned());
  CHECK_FALSE(report["operations"][5]["success"]);
  CHECK(report["operations"][5]["error_code"] == "unknown_record");
  CHECK(report["final_residency"]["resident_record_ids"] ==
        nlohmann::json::array({3}));
  CHECK(report["capacity_invariant"]);
  const auto& stats = report["final_stats"];
  CHECK(stats["successful_fast_reads"] == 1);
  CHECK(stats["successful_slow_reads"] == 1);
  CHECK(stats["promotion_source_reads"] == 3);
  CHECK(stats["promotion_source_read_bytes"] == 40);
  CHECK(stats["committed_promotions"] == 3);
  CHECK(stats["committed_promotion_bytes"] == 40);
  CHECK(stats["committed_evictions"] == 2);
  CHECK(stats["committed_eviction_bytes"] == 19);
  CHECK(stats["target_set_calls"] == 1);
  CHECK(stats["successful_target_set_calls"] == 1);
  CHECK(stats["failed_target_set_calls"] == 0);
  CHECK(stats["current_resident_records"] == 1);
  CHECK(stats["current_resident_bytes"] == 21);
  CHECK(stats["resident_byte_high_water_mark"] == 34);
  CHECK(stats["failures_by_code"]["unknown_record"] == 1);
}

TEST_CASE("replay records expected-outcome mismatches without changing execution") {
  const auto store = build_store("prism_tool_replay_mismatch");
  const auto trace = write_trace(
      "prism_replay_mismatch.jsonl",
      R"({"sequence":1,"operation":"read","record_id":1,"expected":{"success":false,"error_code":"checksum_mismatch"}})"
      "\n");
  auto result = prism::storage::replay_trace_json(store, 40, trace);
  REQUIRE(result);
  CHECK(result->expected_outcome_mismatch_count == 2);
  const auto report = nlohmann::json::parse(result->json);
  CHECK(report["expected_outcome_mismatches"].size() == 2);
  CHECK_FALSE(report["operations"][0]["expected_outcome_matches"]);
  CHECK(report["final_stats"]["successful_slow_reads"] == 1);
}

TEST_CASE("replay rejects malformed JSON, ordering, operations, and unknown fields") {
  const auto store = build_store("prism_tool_replay_invalid");
  const std::vector<std::string> invalid_traces = {
      "{bad json\n",
      "{\"sequence\":1,\"operation\":\"snapshot\"}\n"
      "{\"sequence\":1,\"operation\":\"snapshot\"}\n",
      "{\"sequence\":1,\"operation\":\"unsupported\"}\n",
      "{\"sequence\":1,\"operation\":\"snapshot\",\"extra\":1}\n",
      "{\"sequence\":1,\"operation\":\"read\"}\n",
  };
  std::size_t index = 0;
  for (const auto& contents : invalid_traces) {
    const auto trace = write_trace("prism_replay_invalid_" +
                                       std::to_string(index++) + ".jsonl",
                                   contents);
    auto result = prism::storage::replay_trace_json(store, 40, trace);
    REQUIRE_FALSE(result);
    CHECK(result.error().code ==
          prism::storage::StoreErrorCode::malformed_trace);
  }
}

TEST_CASE("report persistence writes exact deterministic bytes") {
  const auto output = std::filesystem::temp_directory_path() /
                      "prism_deterministic_report.json";
  const std::string contents = "{\n  \"stable\": true\n}\n";
  REQUIRE(prism::storage::write_deterministic_report(output, contents));
  std::ifstream stream(output, std::ios::binary);
  const std::string loaded{std::istreambuf_iterator<char>(stream), {}};
  CHECK(loaded == contents);
}
