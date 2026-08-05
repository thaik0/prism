#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prism/storage/builder.hpp"
#include "prism/storage/detail/checked_arithmetic.hpp"
#include "prism/storage/store_format.hpp"

namespace {

std::filesystem::path fresh_directory(const std::string& name) {
  auto path = std::filesystem::temp_directory_path() / name;
  std::error_code ignored;
  std::filesystem::remove_all(path, ignored);
  return path;
}

std::vector<char> bytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(stream), {});
}

void write_text(const std::filesystem::path& path, const std::string& value) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream << value;
}

std::vector<std::byte> payload_bytes(const std::string& value) {
  std::vector<std::byte> result;
  result.reserve(value.size());
  for (const char byte : value) {
    result.push_back(
        static_cast<std::byte>(static_cast<unsigned char>(byte)));
  }
  return result;
}

const std::filesystem::path kFixtureManifest =
    std::filesystem::path(PRISM_TEST_SOURCE_DIR) / "fixtures" /
    "store_manifest.json";

}  // namespace

TEST_CASE("builder sorts records, writes exact bytes, and verifies checksums") {
  const auto output = fresh_directory("prism_builder_valid");
  const auto result = prism::storage::build_store(kFixtureManifest, output);
  REQUIRE(result);
  CHECK(result->record_count == 3);

  const auto loaded = prism::storage::load_store_index(output);
  REQUIRE(loaded);
  CHECK(loaded->format_version == prism::storage::kStoreFormatVersion);
  CHECK(loaded->records[0].record_id == 1);
  CHECK(loaded->records[1].record_id == 2);
  CHECK(loaded->records[2].record_id == 3);
  CHECK(prism::storage::verify_all_records(*loaded).value() == 3);

  std::vector<char> expected;
  for (const auto id : {1, 2, 3}) {
    const auto payload = std::filesystem::path(PRISM_TEST_SOURCE_DIR) /
                         "fixtures" / "payloads" /
                         ("record_" + std::to_string(id) + ".bin");
    const auto part = bytes(payload);
    expected.insert(expected.end(), part.begin(), part.end());
  }
  CHECK(bytes(output / prism::storage::kStoreDataFilename) == expected);
}

TEST_CASE("repeated builds are byte-identical and input-order independent") {
  const auto root = fresh_directory("prism_builder_determinism");
  std::filesystem::create_directories(root);
  const auto reversed_manifest = root / "manifest.json";
  write_text(reversed_manifest,
             "{\"records\":["
             "{\"record_id\":2,\"payload_path\":\"" +
                 (std::filesystem::path(PRISM_TEST_SOURCE_DIR) / "fixtures" /
                  "payloads" / "record_2.bin")
                     .string() +
                 "\"},"
                 "{\"record_id\":1,\"payload_path\":\"" +
                 (std::filesystem::path(PRISM_TEST_SOURCE_DIR) / "fixtures" /
                  "payloads" / "record_1.bin")
                     .string() +
                 "\"},"
                 "{\"record_id\":3,\"payload_path\":\"" +
                 (std::filesystem::path(PRISM_TEST_SOURCE_DIR) / "fixtures" /
                  "payloads" / "record_3.bin")
                     .string() +
                 "\"}]}\n");
  const auto first = root / "first";
  const auto second = root / "second";
  REQUIRE(prism::storage::build_store(kFixtureManifest, first));
  REQUIRE(prism::storage::build_store(reversed_manifest, second));
  CHECK(bytes(first / prism::storage::kStoreDataFilename) ==
        bytes(second / prism::storage::kStoreDataFilename));
  CHECK(bytes(first / prism::storage::kStoreIndexFilename) ==
        bytes(second / prism::storage::kStoreIndexFilename));
}

TEST_CASE("in-memory records share deterministic builder output") {
  const auto root = fresh_directory("prism_builder_memory");
  std::filesystem::create_directories(root);
  const auto manifest_store = root / "manifest_store";
  const auto memory_store = root / "memory_store";
  REQUIRE(prism::storage::build_store(kFixtureManifest, manifest_store));

  std::vector<prism::storage::BuildRecord> records;
  for (const auto id : {3, 1, 2}) {
    const auto source = std::filesystem::path(PRISM_TEST_SOURCE_DIR) /
                        "fixtures" / "payloads" /
                        ("record_" + std::to_string(id) + ".bin");
    const auto payload = bytes(source);
    records.push_back(prism::storage::BuildRecord{
        static_cast<prism::storage::RecordId>(id),
        payload_bytes(std::string(payload.begin(), payload.end()))});
  }
  const auto built = prism::storage::build_store(std::move(records), memory_store);
  REQUIRE(built);
  CHECK(built->record_count == 3);
  CHECK(bytes(manifest_store / prism::storage::kStoreDataFilename) ==
        bytes(memory_store / prism::storage::kStoreDataFilename));
  CHECK(bytes(manifest_store / prism::storage::kStoreIndexFilename) ==
        bytes(memory_store / prism::storage::kStoreIndexFilename));

  std::vector<prism::storage::BuildRecord> duplicate{
      {1, payload_bytes("one")}, {1, payload_bytes("two")}};
  const auto duplicate_result = prism::storage::build_store(
      std::move(duplicate), root / "duplicate");
  REQUIRE_FALSE(duplicate_result);
  CHECK(duplicate_result.error().code ==
        prism::storage::StoreErrorCode::duplicate_record);

  std::vector<prism::storage::BuildRecord> empty{{1, {}}};
  const auto empty_result =
      prism::storage::build_store(std::move(empty), root / "empty");
  REQUIRE_FALSE(empty_result);
  CHECK(empty_result.error().code ==
        prism::storage::StoreErrorCode::malformed_manifest);
}

TEST_CASE("builder rejects duplicate, missing, and zero-length payloads") {
  const auto root = fresh_directory("prism_builder_invalid");
  std::filesystem::create_directories(root);
  write_text(root / "payload.bin", "payload");
  write_text(root / "empty.bin", "");

  write_text(root / "duplicate.json",
             R"({"records":[{"record_id":1,"payload_path":"payload.bin"},{"record_id":1,"payload_path":"payload.bin"}]})");
  auto duplicate = prism::storage::build_store(root / "duplicate.json",
                                                root / "duplicate_store");
  REQUIRE_FALSE(duplicate);
  CHECK(duplicate.error().code == prism::storage::StoreErrorCode::duplicate_record);

  write_text(root / "missing.json",
             R"({"records":[{"record_id":1,"payload_path":"missing.bin"}]})");
  auto missing = prism::storage::build_store(root / "missing.json",
                                              root / "missing_store");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == prism::storage::StoreErrorCode::io_error);
  CHECK_FALSE(std::filesystem::exists(root / "missing_store"));
  CHECK_FALSE(std::filesystem::exists(root / "missing_store.tmp"));

  write_text(root / "empty.json",
             R"({"records":[{"record_id":1,"payload_path":"empty.bin"}]})");
  auto empty = prism::storage::build_store(root / "empty.json",
                                            root / "empty_store");
  REQUIRE_FALSE(empty);
  CHECK(empty.error().code == prism::storage::StoreErrorCode::malformed_manifest);
}

TEST_CASE("builder preserves occupied destinations and temporary paths") {
  const auto root = fresh_directory("prism_builder_publication");
  std::filesystem::create_directories(root / "store");
  write_text(root / "store" / "keep.txt", "keep");
  auto occupied = prism::storage::build_store(kFixtureManifest, root / "store");
  REQUIRE_FALSE(occupied);
  CHECK(occupied.error().code == prism::storage::StoreErrorCode::destination_exists);
  CHECK(bytes(root / "store" / "keep.txt") == std::vector<char>{'k', 'e', 'e', 'p'});

  std::filesystem::create_directories(root / "other.tmp");
  write_text(root / "other.tmp" / "keep.txt", "keep");
  auto temporary = prism::storage::build_store(kFixtureManifest, root / "other");
  REQUIRE_FALSE(temporary);
  CHECK(temporary.error().code == prism::storage::StoreErrorCode::destination_exists);
  CHECK(std::filesystem::exists(root / "other.tmp" / "keep.txt"));
}

TEST_CASE("builder offset accumulation rejects uint64 overflow") {
  auto result = prism::storage::detail::checked_add_bytes(
      std::numeric_limits<std::uint64_t>::max(), 1, 7);
  REQUIRE_FALSE(result);
  CHECK(result.error().code ==
        prism::storage::StoreErrorCode::arithmetic_overflow);
  CHECK(result.error().record_id == 7);
}

TEST_CASE("builder rejects unreadable payloads and trailing manifest content") {
  const auto root = fresh_directory("prism_builder_unreadable");
  std::filesystem::create_directories(root);
  const auto payload = root / "payload.bin";
  write_text(payload, "payload");
  write_text(root / "manifest.json",
             R"({"records":[{"record_id":1,"payload_path":"payload.bin"}]})");
  std::filesystem::permissions(payload, std::filesystem::perms::none);
  auto unreadable = prism::storage::build_store(root / "manifest.json",
                                                 root / "store");
  std::filesystem::permissions(payload, std::filesystem::perms::owner_read |
                                            std::filesystem::perms::owner_write);
  REQUIRE_FALSE(unreadable);
  CHECK(unreadable.error().code == prism::storage::StoreErrorCode::io_error);
  CHECK_FALSE(std::filesystem::exists(root / "store"));
  CHECK_FALSE(std::filesystem::exists(root / "store.tmp"));

  write_text(root / "trailing.json",
             R"({"records":[{"record_id":1,"payload_path":"payload.bin"}]} {})");
  auto trailing = prism::storage::build_store(root / "trailing.json",
                                               root / "trailing_store");
  REQUIRE_FALSE(trailing);
  CHECK(trailing.error().code ==
        prism::storage::StoreErrorCode::malformed_manifest);
}
