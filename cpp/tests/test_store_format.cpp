#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>

#include "prism/storage/builder.hpp"
#include "prism/storage/store_format.hpp"
#include "prism_store_generated.h"

namespace {

std::filesystem::path built_store(const std::string& name) {
  auto output = std::filesystem::temp_directory_path() / name;
  std::error_code ignored;
  std::filesystem::remove_all(output, ignored);
  auto result = prism::storage::build_store(
      std::filesystem::path(PRISM_TEST_SOURCE_DIR) / "fixtures" /
          "store_manifest.json",
      output);
  REQUIRE(result);
  return output;
}

std::vector<char> read_bytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(stream), {});
}

void write_bytes(const std::filesystem::path& path,
                 const std::vector<char>& value) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_custom_index(
    const std::filesystem::path& path, std::uint32_t version,
    std::uint64_t data_length,
    const std::vector<prism::storage::RecordMetadata>& records) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<prism::storage::fb::Record>> rows;
  for (const auto& record : records) {
    rows.push_back(prism::storage::fb::CreateRecord(
        builder, record.record_id, record.byte_offset, record.byte_length,
        record.crc32));
  }
  const auto root = prism::storage::fb::CreateStoreIndex(
      builder, version, data_length, builder.CreateVector(rows));
  prism::storage::fb::FinishStoreIndexBuffer(builder, root);
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
               static_cast<std::streamsize>(builder.GetSize()));
}

}  // namespace

TEST_CASE("normal loading validates structure without reading payload checksums") {
  const auto store = built_store("prism_format_valid");
  auto data = read_bytes(store / prism::storage::kStoreDataFilename);
  data[0] ^= 0x1;
  write_bytes(store / prism::storage::kStoreDataFilename, data);

  const auto loaded = prism::storage::load_store_index(store);
  REQUIRE(loaded);
  auto verified = prism::storage::verify_all_records(*loaded);
  REQUIRE_FALSE(verified);
  CHECK(verified.error().code == prism::storage::StoreErrorCode::checksum_mismatch);
}

TEST_CASE("loader rejects truncated, malformed, and wrong-identifier indexes") {
  const auto store = built_store("prism_format_bad_index");
  const auto index_path = store / prism::storage::kStoreIndexFilename;
  auto original = read_bytes(index_path);

  write_bytes(index_path, std::vector<char>(original.begin(), original.begin() + 5));
  auto truncated = prism::storage::load_store_index(store);
  REQUIRE_FALSE(truncated);
  CHECK(truncated.error().code == prism::storage::StoreErrorCode::index_corrupt);

  auto wrong_identifier = original;
  wrong_identifier[4] = 'X';
  write_bytes(index_path, wrong_identifier);
  auto identifier = prism::storage::load_store_index(store);
  REQUIRE_FALSE(identifier);
  CHECK(identifier.error().code == prism::storage::StoreErrorCode::index_corrupt);

  auto malformed = original;
  malformed.back() ^= 0x7f;
  write_bytes(index_path, malformed);
  auto invalid = prism::storage::load_store_index(store);
  REQUIRE_FALSE(invalid);
  CHECK(invalid.error().code == prism::storage::StoreErrorCode::index_corrupt);
}

TEST_CASE("loader rejects unsupported versions and invalid record ordering") {
  const auto store = built_store("prism_format_metadata");
  const auto loaded = prism::storage::load_store_index(store);
  REQUIRE(loaded);
  const auto index_path = store / prism::storage::kStoreIndexFilename;

  write_custom_index(index_path, 2, loaded->data_file_length, loaded->records);
  auto version = prism::storage::load_store_index(store);
  REQUIRE_FALSE(version);
  CHECK(version.error().code ==
        prism::storage::StoreErrorCode::unsupported_format_version);

  auto unsorted = loaded->records;
  std::swap(unsorted[0].record_id, unsorted[1].record_id);
  write_custom_index(index_path, 1, loaded->data_file_length, unsorted);
  auto order = prism::storage::load_store_index(store);
  REQUIRE_FALSE(order);
  CHECK(order.error().code == prism::storage::StoreErrorCode::index_corrupt);

  auto duplicate = loaded->records;
  duplicate[1].record_id = duplicate[0].record_id;
  write_custom_index(index_path, 1, loaded->data_file_length, duplicate);
  auto duplicates = prism::storage::load_store_index(store);
  REQUIRE_FALSE(duplicates);
  CHECK(duplicates.error().code == prism::storage::StoreErrorCode::index_corrupt);
}

TEST_CASE("loader rejects zero, overlapping, beyond-range, and length mismatch metadata") {
  const auto store = built_store("prism_format_ranges");
  const auto loaded = prism::storage::load_store_index(store);
  REQUIRE(loaded);
  const auto index_path = store / prism::storage::kStoreIndexFilename;

  auto zero = loaded->records;
  zero[0].byte_length = 0;
  write_custom_index(index_path, 1, loaded->data_file_length, zero);
  CHECK_FALSE(prism::storage::load_store_index(store));

  auto overlap = loaded->records;
  overlap[1].byte_offset = overlap[0].byte_offset;
  write_custom_index(index_path, 1, loaded->data_file_length, overlap);
  CHECK_FALSE(prism::storage::load_store_index(store));

  auto beyond = loaded->records;
  beyond.back().byte_length += 1;
  write_custom_index(index_path, 1, loaded->data_file_length, beyond);
  CHECK_FALSE(prism::storage::load_store_index(store));

  write_custom_index(index_path, 1, loaded->data_file_length + 1, loaded->records);
  auto mismatch = prism::storage::load_store_index(store);
  REQUIRE_FALSE(mismatch);
  CHECK(mismatch.error().code == prism::storage::StoreErrorCode::data_file_mismatch);
}

TEST_CASE("loader rejects missing files and exact data truncation or append") {
  const auto missing_index = built_store("prism_format_missing_index");
  std::filesystem::remove(missing_index / prism::storage::kStoreIndexFilename);
  auto index = prism::storage::load_store_index(missing_index);
  REQUIRE_FALSE(index);
  CHECK(index.error().code == prism::storage::StoreErrorCode::index_corrupt);

  const auto missing_data = built_store("prism_format_missing_data");
  std::filesystem::remove(missing_data / prism::storage::kStoreDataFilename);
  auto data_missing = prism::storage::load_store_index(missing_data);
  REQUIRE_FALSE(data_missing);
  CHECK(data_missing.error().code ==
        prism::storage::StoreErrorCode::data_file_mismatch);

  const auto truncated = built_store("prism_format_truncated_data");
  auto data = read_bytes(truncated / prism::storage::kStoreDataFilename);
  data.pop_back();
  write_bytes(truncated / prism::storage::kStoreDataFilename, data);
  auto short_data = prism::storage::load_store_index(truncated);
  REQUIRE_FALSE(short_data);
  CHECK(short_data.error().code == prism::storage::StoreErrorCode::data_file_mismatch);

  const auto appended = built_store("prism_format_appended_data");
  auto extra = read_bytes(appended / prism::storage::kStoreDataFilename);
  extra.push_back('x');
  write_bytes(appended / prism::storage::kStoreDataFilename, extra);
  auto long_data = prism::storage::load_store_index(appended);
  REQUIRE_FALSE(long_data);
  CHECK(long_data.error().code == prism::storage::StoreErrorCode::data_file_mismatch);
}

TEST_CASE("metadata validation rejects offset arithmetic overflow") {
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::vector<prism::storage::RecordMetadata> records{
      {1, 0, maximum, 0}, {2, maximum, 1, 0}};
  auto result = prism::storage::validate_store_metadata(
      records, maximum, maximum, "synthetic.index");
  REQUIRE_FALSE(result);
  CHECK(result.error().code ==
        prism::storage::StoreErrorCode::arithmetic_overflow);
  CHECK(result.error().record_id == 2);
  CHECK(result.error().byte_offset == maximum);
}
