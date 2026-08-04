#include "prism/storage/builder.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <nlohmann/json.hpp>
#include <zlib.h>

#include "prism/storage/store_format.hpp"
#include "prism/storage/detail/checked_arithmetic.hpp"
#include "prism_store_generated.h"

namespace prism::storage {
namespace {

struct ManifestRecord {
  RecordId record_id;
  std::filesystem::path payload_path;
};

StoreError make_error(StoreErrorCode code, std::string message,
                      std::optional<std::filesystem::path> path = std::nullopt,
                      std::optional<RecordId> record_id = std::nullopt) {
  return StoreError{code, std::move(message), record_id, std::nullopt,
                    std::move(path)};
}

Expected<std::vector<ManifestRecord>> load_manifest(
    const std::filesystem::path& manifest_path) {
  std::ifstream stream(manifest_path);
  if (!stream) {
    return unexpected(make_error(StoreErrorCode::malformed_manifest,
                                 "could not open manifest", manifest_path));
  }
  try {
    nlohmann::json document;
    stream >> document;
    stream >> std::ws;
    if (stream.peek() != std::char_traits<char>::eof()) {
      return unexpected(make_error(StoreErrorCode::malformed_manifest,
                                   "manifest contains trailing content",
                                   manifest_path));
    }
    if (!document.is_object() || document.size() != 1U ||
        !document.contains("records") || !document["records"].is_array() ||
        document["records"].empty()) {
      return unexpected(make_error(
          StoreErrorCode::malformed_manifest,
          "manifest must contain only one nonempty records array", manifest_path));
    }
    std::vector<ManifestRecord> records;
    records.reserve(document["records"].size());
    const auto base_directory = manifest_path.parent_path();
    for (const auto& row : document["records"]) {
      if (!row.is_object() || row.size() != 2U || !row.contains("record_id") ||
          !row.contains("payload_path") ||
          !row["record_id"].is_number_unsigned() ||
          !row["payload_path"].is_string()) {
        return unexpected(make_error(
            StoreErrorCode::malformed_manifest,
            "each record must contain unsigned record_id and string payload_path",
            manifest_path));
      }
      const auto payload_text = row["payload_path"].get<std::string>();
      if (payload_text.empty()) {
        return unexpected(make_error(StoreErrorCode::malformed_manifest,
                                     "payload_path must be nonempty", manifest_path));
      }
      std::filesystem::path payload_path(payload_text);
      if (payload_path.is_relative()) {
        payload_path = base_directory / payload_path;
      }
      records.push_back(
          ManifestRecord{row["record_id"].get<RecordId>(), payload_path});
    }
    std::sort(records.begin(), records.end(),
              [](const ManifestRecord& left, const ManifestRecord& right) {
                return left.record_id < right.record_id;
              });
    for (std::size_t index = 1; index < records.size(); ++index) {
      if (records[index - 1].record_id == records[index].record_id) {
        return unexpected(make_error(StoreErrorCode::duplicate_record,
                                     "manifest contains a duplicate record ID",
                                     manifest_path, records[index].record_id));
      }
    }
    return records;
  } catch (const nlohmann::json::exception&) {
    return unexpected(make_error(StoreErrorCode::malformed_manifest,
                                 "manifest is not valid strict JSON", manifest_path));
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate manifest records", manifest_path));
  }
}

class TemporaryDirectoryGuard {
 public:
  explicit TemporaryDirectoryGuard(std::filesystem::path path)
      : path_(std::move(path)) {}
  TemporaryDirectoryGuard(const TemporaryDirectoryGuard&) = delete;
  TemporaryDirectoryGuard& operator=(const TemporaryDirectoryGuard&) = delete;
  ~TemporaryDirectoryGuard() {
    if (active_) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }
  void release() noexcept { active_ = false; }

 private:
  std::filesystem::path path_;
  bool active_ = true;
};

Expected<bool> prepare_destination(const std::filesystem::path& destination) {
  std::error_code error;
  if (std::filesystem::exists(destination, error)) {
    if (error || !std::filesystem::is_directory(destination, error) ||
        error || !std::filesystem::is_empty(destination, error) || error) {
      return unexpected(make_error(StoreErrorCode::destination_exists,
                                   "destination exists and is not an empty directory",
                                   destination));
    }
    if (!std::filesystem::remove(destination, error) || error) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "could not remove empty destination directory",
                                   destination));
    }
  } else if (error) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not inspect destination", destination));
  }
  return true;
}

Expected<std::uint64_t> payload_size(const ManifestRecord& record) {
  std::error_code error;
  const auto status = std::filesystem::status(record.payload_path, error);
  if (error || !std::filesystem::is_regular_file(status)) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "payload is missing or is not a regular file",
                                 record.payload_path, record.record_id));
  }
  const std::uintmax_t size =
      std::filesystem::file_size(record.payload_path, error);
  if (error || size > std::numeric_limits<std::uint64_t>::max()) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "payload length is not representable",
                                 record.payload_path, record.record_id));
  }
  if (size == 0) {
    return unexpected(make_error(StoreErrorCode::malformed_manifest,
                                 "zero-length payloads are not supported",
                                 record.payload_path, record.record_id));
  }
  return static_cast<std::uint64_t>(size);
}

Expected<RecordMetadata> append_payload(const ManifestRecord& record,
                                        std::uint64_t offset,
                                        std::ofstream& destination) {
  auto size = payload_size(record);
  if (!size) {
    return unexpected(size.error());
  }
  auto next_offset = detail::checked_add_bytes(
      offset, *size, record.record_id, record.payload_path);
  if (!next_offset) {
    return unexpected(next_offset.error());
  }
  std::ifstream source(record.payload_path, std::ios::binary);
  if (!source) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "payload is unreadable", record.payload_path,
                                 record.record_id));
  }
  std::array<char, 64U * 1024U> buffer{};
  std::uint64_t copied = 0;
  uLong checksum = ::crc32(0L, Z_NULL, 0);
  while (copied < *size) {
    const std::uint64_t remaining = *size - copied;
    const std::size_t requested = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    source.read(buffer.data(), static_cast<std::streamsize>(requested));
    const std::streamsize received = source.gcount();
    if (received <= 0) {
      return unexpected(make_error(StoreErrorCode::truncated_read,
                                   "payload became truncated while building",
                                   record.payload_path, record.record_id));
    }
    destination.write(buffer.data(), received);
    if (!destination) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "could not write store.data", std::nullopt,
                                   record.record_id));
    }
    checksum = ::crc32(checksum, reinterpret_cast<const Bytef*>(buffer.data()),
                       static_cast<uInt>(received));
    copied += static_cast<std::uint64_t>(received);
  }
  if (source.bad()) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "payload read failed", record.payload_path,
                                 record.record_id));
  }
  return RecordMetadata{record.record_id, offset, *size,
                        static_cast<std::uint32_t>(checksum)};
}

Expected<bool> write_index(const std::filesystem::path& path,
                           const std::vector<RecordMetadata>& records,
                           std::uint64_t data_length) {
  try {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::Record>> rows;
    rows.reserve(records.size());
    for (const auto& record : records) {
      rows.push_back(fb::CreateRecord(builder, record.record_id,
                                      record.byte_offset, record.byte_length,
                                      record.crc32));
    }
    const auto vector = builder.CreateVector(rows);
    const auto root = fb::CreateStoreIndex(builder, kStoreFormatVersion,
                                           data_length, vector);
    fb::FinishStoreIndexBuffer(builder, root);

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "could not create store.index", path));
    }
    stream.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                 static_cast<std::streamsize>(builder.GetSize()));
    stream.flush();
    if (!stream) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "could not write store.index", path));
    }
    stream.close();
    if (stream.fail()) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "could not close store.index", path));
    }
    return true;
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate store index", path));
  }
}

}  // namespace

Expected<BuildResult> build_store(
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& output_directory) {
  auto manifest = load_manifest(manifest_path);
  if (!manifest) {
    return unexpected(manifest.error());
  }
  auto destination_ready = prepare_destination(output_directory);
  if (!destination_ready) {
    return unexpected(destination_ready.error());
  }

  std::filesystem::path temporary = output_directory;
  temporary += ".tmp";
  std::error_code error;
  if (std::filesystem::exists(temporary, error)) {
    if (error || !std::filesystem::is_directory(temporary, error) || error ||
        !std::filesystem::is_empty(temporary, error) || error) {
      return unexpected(make_error(
          StoreErrorCode::destination_exists,
          "builder temporary path exists and is not an empty directory",
          temporary));
    }
    if (!std::filesystem::remove(temporary, error) || error) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "could not remove empty temporary directory",
                                   temporary));
    }
  } else if (error) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not inspect temporary path", temporary));
  }
  if (!std::filesystem::create_directory(temporary, error) || error) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not create temporary directory", temporary));
  }
  TemporaryDirectoryGuard temporary_guard(temporary);

  const auto data_path = temporary / kStoreDataFilename;
  std::ofstream data_stream(data_path, std::ios::binary | std::ios::trunc);
  if (!data_stream) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not create store.data", data_path));
  }
  std::vector<RecordMetadata> metadata;
  try {
    metadata.reserve(manifest->size());
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate record metadata", temporary));
  }
  std::uint64_t offset = 0;
  for (const auto& record : *manifest) {
    auto appended = append_payload(record, offset, data_stream);
    if (!appended) {
      return unexpected(appended.error());
    }
    metadata.push_back(*appended);
    auto next_offset = detail::checked_add_bytes(
        offset, appended->byte_length, appended->record_id, data_path);
    if (!next_offset) {
      return unexpected(next_offset.error());
    }
    offset = *next_offset;
  }
  data_stream.flush();
  if (!data_stream) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not flush store.data", data_path));
  }
  data_stream.close();
  if (data_stream.fail()) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not close store.data", data_path));
  }

  auto index_written = write_index(temporary / kStoreIndexFilename, metadata, offset);
  if (!index_written) {
    return unexpected(index_written.error());
  }
  auto loaded = load_store_index(temporary);
  if (!loaded) {
    return unexpected(loaded.error());
  }
  auto verified = verify_all_records(*loaded);
  if (!verified) {
    return unexpected(verified.error());
  }

  std::filesystem::rename(temporary, output_directory, error);
  if (error) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not publish completed store",
                                 output_directory));
  }
  temporary_guard.release();
  return BuildResult{static_cast<std::uint64_t>(metadata.size()), offset};
}

}  // namespace prism::storage
