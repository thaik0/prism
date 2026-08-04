#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prism/storage/builder.hpp"
#include "prism/storage/tiered_store.hpp"

namespace {

struct StoreFixture {
  std::filesystem::path directory;
  prism::storage::LoadedStoreIndex index;
};

StoreFixture build_fixture(const std::string& name) {
  auto directory = std::filesystem::temp_directory_path() / name;
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
  const auto built = prism::storage::build_store(
      std::filesystem::path(PRISM_TEST_SOURCE_DIR) / "fixtures" /
          "store_manifest.json",
      directory);
  REQUIRE(built);
  auto loaded = prism::storage::load_store_index(directory);
  REQUIRE(loaded);
  return StoreFixture{directory, std::move(*loaded)};
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  const std::vector<char> chars{std::istreambuf_iterator<char>(stream), {}};
  std::vector<std::byte> result;
  result.reserve(chars.size());
  for (const char value : chars) {
    result.push_back(static_cast<std::byte>(value));
  }
  return result;
}

const prism::storage::RecordMetadata& record(
    const StoreFixture& fixture, prism::storage::RecordId id) {
  for (const auto& metadata : fixture.index.records) {
    if (metadata.record_id == id) {
      return metadata;
    }
  }
  throw std::logic_error("fixture record not found");
}

void corrupt_byte(const StoreFixture& fixture, prism::storage::RecordId id) {
  const auto& metadata = record(fixture, id);
  std::fstream stream(fixture.index.data_path,
                      std::ios::binary | std::ios::in | std::ios::out);
  stream.seekg(static_cast<std::streamoff>(metadata.byte_offset));
  char value = 0;
  stream.read(&value, 1);
  value ^= 0x1;
  stream.seekp(static_cast<std::streamoff>(metadata.byte_offset));
  stream.write(&value, 1);
  stream.flush();
}

}  // namespace

TEST_CASE("factory validates capacity, metadata, and unknown records") {
  const auto fixture = build_fixture("prism_engine_factory");
  auto zero = prism::storage::TieredStore::open(fixture.directory, 0);
  REQUIRE_FALSE(zero);
  CHECK(zero.error().code ==
        prism::storage::StoreErrorCode::invalid_configuration);

  auto too_small = prism::storage::TieredStore::open(fixture.directory, 1);
  REQUIRE_FALSE(too_small);
  CHECK(too_small.error().code ==
        prism::storage::StoreErrorCode::invalid_configuration);

  auto store = prism::storage::TieredStore::open(fixture.directory, 40);
  REQUIRE(store);
  CHECK((*store)->metadata(2).value() == record(fixture, 2));
  auto unknown = (*store)->metadata(999);
  REQUIRE_FALSE(unknown);
  CHECK(unknown.error().code == prism::storage::StoreErrorCode::unknown_record);
  CHECK((*store)->residency_snapshot() ==
        prism::storage::ResidencySnapshot{{}, 0, 40});
}

TEST_CASE("slow and fast reads return exact bytes without automatic promotion") {
  const auto fixture = build_fixture("prism_engine_reads");
  auto store = prism::storage::TieredStore::open(fixture.directory, 40);
  REQUIRE(store);
  std::vector<std::byte> destination{std::byte{0xff}};

  auto slow = (*store)->read_into(1, destination);
  REQUIRE(slow);
  CHECK(slow->tier == prism::storage::ReadTier::slow);
  CHECK(destination == read_file(std::filesystem::path(PRISM_TEST_SOURCE_DIR) /
                                 "fixtures" / "payloads" / "record_1.bin"));
  CHECK((*store)->residency_snapshot().resident_record_ids.empty());

  auto promoted = (*store)->promote(1);
  REQUIRE(promoted);
  CHECK_FALSE(promoted->already_resident);
  destination.assign(100, std::byte{0});
  auto fast = (*store)->read_into(1, destination);
  REQUIRE(fast);
  CHECK(fast->tier == prism::storage::ReadTier::fast);
  CHECK(destination.size() == record(fixture, 1).byte_length);

  destination.assign(4, std::byte{1});
  auto unknown = (*store)->read_into(99, destination);
  REQUIRE_FALSE(unknown);
  CHECK(destination.empty());
  CHECK((*store)->residency_snapshot().resident_record_ids ==
        std::vector<prism::storage::RecordId>{1});
}

TEST_CASE("slow read checksum and truncation failures clear destination and preserve state") {
  const auto checksum_fixture = build_fixture("prism_engine_read_checksum");
  auto checksum_store =
      prism::storage::TieredStore::open(checksum_fixture.directory, 40);
  REQUIRE(checksum_store);
  corrupt_byte(checksum_fixture, 2);
  std::vector<std::byte> destination(5, std::byte{1});
  auto checksum = (*checksum_store)->read_into(2, destination);
  REQUIRE_FALSE(checksum);
  CHECK(checksum.error().code ==
        prism::storage::StoreErrorCode::checksum_mismatch);
  CHECK(destination.empty());
  CHECK((*checksum_store)->stats().successful_slow_reads == 0);
  CHECK((*checksum_store)->stats().failure_count(
            prism::storage::StoreErrorCode::checksum_mismatch) == 1);

  const auto short_fixture = build_fixture("prism_engine_read_short");
  auto short_store = prism::storage::TieredStore::open(short_fixture.directory, 40);
  REQUIRE(short_store);
  std::filesystem::resize_file(short_fixture.index.data_path,
                               record(short_fixture, 3).byte_offset + 1);
  destination.assign(5, std::byte{1});
  auto truncated = (*short_store)->read_into(3, destination);
  REQUIRE_FALSE(truncated);
  CHECK(truncated.error().code == prism::storage::StoreErrorCode::truncated_read);
  CHECK(destination.empty());
  CHECK((*short_store)->residency_snapshot().resident_record_ids.empty());
}

TEST_CASE("promotion is verified, idempotent, capacity-strict, and never evicts") {
  const auto fixture = build_fixture("prism_engine_promotion");
  auto store = prism::storage::TieredStore::open(fixture.directory, 27);
  REQUIRE(store);
  REQUIRE((*store)->promote(3));
  REQUIRE((*store)->promote(1));
  auto repeat = (*store)->promote(1);
  REQUIRE(repeat);
  CHECK(repeat->already_resident);
  auto insufficient = (*store)->promote(2);
  REQUIRE_FALSE(insufficient);
  CHECK(insufficient.error().code ==
        prism::storage::StoreErrorCode::insufficient_capacity);
  CHECK((*store)->residency_snapshot().resident_record_ids ==
        std::vector<prism::storage::RecordId>{1, 3});

  auto small = prism::storage::TieredStore::open(fixture.directory, 13);
  REQUIRE(small);
  auto oversized = (*small)->promote(3);
  REQUIRE_FALSE(oversized);
  CHECK(oversized.error().code ==
        prism::storage::StoreErrorCode::oversized_record);

  const auto corrupt_fixture = build_fixture("prism_engine_promotion_corrupt");
  auto corrupt_store =
      prism::storage::TieredStore::open(corrupt_fixture.directory, 40);
  REQUIRE(corrupt_store);
  corrupt_byte(corrupt_fixture, 1);
  auto corrupt = (*corrupt_store)->promote(1);
  REQUIRE_FALSE(corrupt);
  CHECK(corrupt.error().code ==
        prism::storage::StoreErrorCode::checksum_mismatch);
  CHECK((*corrupt_store)->residency_snapshot().resident_record_ids.empty());
  CHECK((*corrupt_store)->stats().committed_promotions == 0);
}

TEST_CASE("eviction is idempotent and leaves authoritative slow bytes unchanged") {
  const auto fixture = build_fixture("prism_engine_eviction");
  const auto before = read_file(fixture.index.data_path);
  auto store = prism::storage::TieredStore::open(fixture.directory, 40);
  REQUIRE(store);
  REQUIRE((*store)->promote(2));
  auto removed = (*store)->evict(2);
  REQUIRE(removed);
  CHECK(removed->was_resident);
  auto absent = (*store)->evict(2);
  REQUIRE(absent);
  CHECK_FALSE(absent->was_resident);
  auto unknown = (*store)->evict(99);
  REQUIRE_FALSE(unknown);
  CHECK(read_file(fixture.index.data_path) == before);
  CHECK((*store)->stats().committed_evictions == 1);
}

TEST_CASE("exact target replacement is deterministic and capacity safe") {
  const auto fixture_a = build_fixture("prism_engine_target_a");
  const auto fixture_b = build_fixture("prism_engine_target_b");
  auto first = prism::storage::TieredStore::open(fixture_a.directory, 40);
  auto second = prism::storage::TieredStore::open(fixture_b.directory, 40);
  REQUIRE(first);
  REQUIRE(second);
  REQUIRE((*first)->promote(2));
  REQUIRE((*second)->promote(2));

  auto result_a = (*first)->apply_target_set({3, 1});
  auto result_b = (*second)->apply_target_set({1, 3});
  REQUIRE(result_a);
  REQUIRE(result_b);
  CHECK(result_a->residency == result_b->residency);
  CHECK(result_a->residency.resident_record_ids ==
        std::vector<prism::storage::RecordId>{1, 3});
  CHECK(result_a->promotion_count == 2);
  CHECK(result_a->eviction_count == 1);
  CHECK(result_a->residency.resident_bytes <=
        result_a->residency.capacity_bytes);
  CHECK((*first)->stats().promotion_source_reads ==
        (*second)->stats().promotion_source_reads);

  auto unchanged = (*first)->apply_target_set({3, 1});
  REQUIRE(unchanged);
  CHECK(unchanged->promotion_count == 0);
  CHECK(unchanged->eviction_count == 0);
  auto empty = (*first)->apply_target_set({});
  REQUIRE(empty);
  CHECK(empty->residency.resident_record_ids.empty());
}

TEST_CASE("target preflight failures preserve exact prior residency") {
  const auto fixture = build_fixture("prism_engine_target_preflight");
  auto store = prism::storage::TieredStore::open(fixture.directory, 27);
  REQUIRE(store);
  REQUIRE((*store)->promote(1));
  const auto before = (*store)->residency_snapshot();
  const auto committed_before = (*store)->stats().committed_promotions;

  for (const auto& target :
       std::vector<std::vector<prism::storage::RecordId>>{{1, 1}, {1, 99},
                                                         {2, 3}}) {
    auto result = (*store)->apply_target_set(target);
    REQUIRE_FALSE(result);
    CHECK((*store)->residency_snapshot() == before);
    CHECK((*store)->stats().committed_promotions == committed_before);
  }

  auto small = prism::storage::TieredStore::open(fixture.directory, 13);
  REQUIRE(small);
  auto oversized = (*small)->apply_target_set({3});
  REQUIRE_FALSE(oversized);
  CHECK((*small)->residency_snapshot().resident_record_ids.empty());
}

TEST_CASE("failed staged target reads discard staging and do not partially evict") {
  const auto fixture = build_fixture("prism_engine_target_atomic");
  auto store = prism::storage::TieredStore::open(fixture.directory, 40);
  REQUIRE(store);
  REQUIRE((*store)->promote(1));
  const auto before = (*store)->residency_snapshot();
  const auto committed_before = (*store)->stats().committed_promotions;
  corrupt_byte(fixture, 3);

  auto result = (*store)->apply_target_set({2, 3});
  REQUIRE_FALSE(result);
  CHECK(result.error().code ==
        prism::storage::StoreErrorCode::checksum_mismatch);
  CHECK((*store)->residency_snapshot() == before);
  const auto stats = (*store)->stats();
  CHECK(stats.committed_promotions == committed_before);
  CHECK(stats.committed_evictions == 0);
  CHECK(stats.failed_target_set_calls == 1);
  CHECK(stats.aborted_staged_bytes == record(fixture, 2).byte_length);
  CHECK(stats.promotion_source_reads == 2);
}

TEST_CASE("truncated staged target read preserves exact prior residency") {
  const auto fixture = build_fixture("prism_engine_target_truncated");
  auto store = prism::storage::TieredStore::open(fixture.directory, 40);
  REQUIRE(store);
  REQUIRE((*store)->promote(1));
  const auto before = (*store)->residency_snapshot();
  const auto committed_before = (*store)->stats().committed_promotions;
  std::filesystem::resize_file(fixture.index.data_path,
                               record(fixture, 3).byte_offset + 1);

  auto result = (*store)->apply_target_set({2, 3});
  REQUIRE_FALSE(result);
  CHECK(result.error().code == prism::storage::StoreErrorCode::truncated_read);
  CHECK((*store)->residency_snapshot() == before);
  const auto stats = (*store)->stats();
  CHECK(stats.committed_promotions == committed_before);
  CHECK(stats.committed_evictions == 0);
  CHECK(stats.aborted_staged_bytes == record(fixture, 2).byte_length);
}

TEST_CASE("logical counters distinguish client, migration, movement, and failures") {
  const auto fixture = build_fixture("prism_engine_stats");
  auto store = prism::storage::TieredStore::open(fixture.directory, 40);
  REQUIRE(store);
  std::vector<std::byte> destination;
  REQUIRE((*store)->read_into(1, destination));
  REQUIRE((*store)->promote(1));
  REQUIRE((*store)->promote(1));
  REQUIRE((*store)->read_into(1, destination));
  REQUIRE((*store)->apply_target_set({2, 3}));
  REQUIRE((*store)->evict(2));
  REQUIRE((*store)->evict(2));
  REQUIRE_FALSE((*store)->read_into(99, destination));

  const auto stats = (*store)->stats();
  CHECK(stats.successful_fast_reads == 1);
  CHECK(stats.successful_fast_read_bytes == record(fixture, 1).byte_length);
  CHECK(stats.successful_slow_reads == 1);
  CHECK(stats.successful_slow_read_bytes == record(fixture, 1).byte_length);
  CHECK(stats.promotion_source_reads == 3);
  CHECK(stats.promotion_source_read_bytes == fixture.index.data_file_length);
  CHECK(stats.committed_promotions == 3);
  CHECK(stats.committed_promotion_bytes == fixture.index.data_file_length);
  CHECK(stats.committed_evictions == 2);
  CHECK(stats.committed_eviction_bytes ==
        record(fixture, 1).byte_length + record(fixture, 2).byte_length);
  CHECK(stats.target_set_calls == 1);
  CHECK(stats.successful_target_set_calls == 1);
  CHECK(stats.failed_target_set_calls == 0);
  CHECK(stats.current_resident_records == 1);
  CHECK(stats.current_resident_bytes == record(fixture, 3).byte_length);
  CHECK(stats.resident_byte_high_water_mark ==
        record(fixture, 2).byte_length + record(fixture, 3).byte_length);
  CHECK(stats.failure_count(prism::storage::StoreErrorCode::unknown_record) == 1);
}
