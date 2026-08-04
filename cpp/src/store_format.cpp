#include "prism/storage/store_format.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <system_error>

#include <fcntl.h>
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <unistd.h>
#include <zlib.h>

#include "prism_store_generated.h"

namespace prism::storage {
namespace {

StoreError make_error(StoreErrorCode code, std::string message,
                      std::optional<std::filesystem::path> path = std::nullopt) {
  return StoreError{code, std::move(message), std::nullopt, std::nullopt,
                    std::move(path)};
}

Expected<std::uint64_t> regular_file_size(const std::filesystem::path& path,
                                          StoreErrorCode missing_code) {
  std::error_code error;
  const auto status = std::filesystem::status(path, error);
  if (error || !std::filesystem::is_regular_file(status)) {
    return unexpected(make_error(missing_code, "required regular file is missing",
                                 path));
  }
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size > std::numeric_limits<std::uint64_t>::max()) {
    return unexpected(make_error(StoreErrorCode::io_error,
                                 "could not determine a representable file size",
                                 path));
  }
  return static_cast<std::uint64_t>(size);
}

Expected<std::vector<std::uint8_t>> read_file(
    const std::filesystem::path& path, std::uint64_t length) {
  if (length > std::numeric_limits<std::size_t>::max()) {
    return unexpected(make_error(StoreErrorCode::arithmetic_overflow,
                                 "file length does not fit in memory", path));
  }
  try {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "could not open file for reading", path));
    }
    if (!bytes.empty()) {
      stream.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
      return unexpected(make_error(StoreErrorCode::io_error,
                                   "could not read the complete file", path));
    }
    return bytes;
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate file buffer", path));
  }
}

std::uint32_t payload_crc32(const std::vector<std::byte>& payload) {
  uLong checksum = ::crc32(0L, Z_NULL, 0);
  std::size_t consumed = 0;
  while (consumed < payload.size()) {
    const std::size_t remaining = payload.size() - consumed;
    const std::size_t chunk_size =
        std::min<std::size_t>(remaining, std::numeric_limits<uInt>::max());
    checksum = ::crc32(
        checksum,
        reinterpret_cast<const Bytef*>(payload.data() + consumed),
        static_cast<uInt>(chunk_size));
    consumed += chunk_size;
  }
  return static_cast<std::uint32_t>(checksum);
}

}  // namespace

Expected<LoadedStoreIndex> load_store_index(
    const std::filesystem::path& store_directory) {
  std::error_code error;
  const auto directory_status = std::filesystem::status(store_directory, error);
  if (error || !std::filesystem::is_directory(directory_status)) {
    return unexpected(make_error(StoreErrorCode::invalid_configuration,
                                 "store directory does not exist", store_directory));
  }

  const auto data_path = store_directory / kStoreDataFilename;
  const auto index_path = store_directory / kStoreIndexFilename;
  auto data_length = regular_file_size(data_path, StoreErrorCode::data_file_mismatch);
  if (!data_length) {
    return unexpected(data_length.error());
  }
  auto index_length = regular_file_size(index_path, StoreErrorCode::index_corrupt);
  if (!index_length) {
    return unexpected(index_length.error());
  }
  if (*index_length < sizeof(flatbuffers::uoffset_t) + 4U) {
    return unexpected(make_error(StoreErrorCode::index_corrupt,
                                 "store index is truncated", index_path));
  }
  auto bytes = read_file(index_path, *index_length);
  if (!bytes) {
    return unexpected(bytes.error());
  }
  if (!fb::StoreIndexBufferHasIdentifier(bytes->data())) {
    return unexpected(make_error(StoreErrorCode::index_corrupt,
                                 "store index has an invalid file identifier",
                                 index_path));
  }
  flatbuffers::Verifier verifier(bytes->data(), bytes->size());
  if (!fb::VerifyStoreIndexBuffer(verifier)) {
    return unexpected(make_error(StoreErrorCode::index_corrupt,
                                 "store index failed FlatBuffers verification",
                                 index_path));
  }

  const fb::StoreIndex* root = fb::GetStoreIndex(bytes->data());
  if (root->format_version() != kStoreFormatVersion) {
    return unexpected(make_error(
        StoreErrorCode::unsupported_format_version,
        "store index uses an unsupported format version", index_path));
  }
  if (root->data_file_length() != *data_length) {
    return unexpected(make_error(StoreErrorCode::data_file_mismatch,
                                 "declared data length does not match the file",
                                 data_path));
  }
  const auto* records = root->records();
  if (records == nullptr || records->size() == 0U) {
    return unexpected(make_error(StoreErrorCode::index_corrupt,
                                 "store index contains no records", index_path));
  }

  LoadedStoreIndex result{store_directory, data_path, index_path,
                          root->format_version(), root->data_file_length(), {}};
  try {
    result.records.reserve(records->size());
  } catch (const std::bad_alloc&) {
    return unexpected(make_error(StoreErrorCode::allocation_failure,
                                 "could not allocate record metadata", index_path));
  }

  std::uint64_t expected_offset = 0;
  std::optional<RecordId> previous_id;
  for (const fb::Record* record : *records) {
    if (record == nullptr) {
      return unexpected(make_error(StoreErrorCode::index_corrupt,
                                   "store index contains a null record", index_path));
    }
    const RecordId id = record->record_id();
    const std::uint64_t offset = record->byte_offset();
    const std::uint64_t length = record->byte_length();
    if (previous_id && id <= *previous_id) {
      return unexpected(make_error(StoreErrorCode::index_corrupt,
                                   "record IDs are not strictly increasing",
                                   index_path));
    }
    if (length == 0) {
      return unexpected(make_error(StoreErrorCode::index_corrupt,
                                   "record length must be positive", index_path));
    }
    if (offset != expected_offset) {
      return unexpected(make_error(StoreErrorCode::index_corrupt,
                                   "record byte ranges are not contiguous and ordered",
                                   index_path));
    }
    if (length > std::numeric_limits<std::uint64_t>::max() - offset) {
      return unexpected(make_error(StoreErrorCode::arithmetic_overflow,
                                   "record byte range overflows", index_path));
    }
    const std::uint64_t end = offset + length;
    if (end > root->data_file_length()) {
      return unexpected(make_error(StoreErrorCode::index_corrupt,
                                   "record byte range exceeds the data file",
                                   index_path));
    }
    result.records.push_back(
        RecordMetadata{id, offset, length, record->crc32()});
    expected_offset = end;
    previous_id = id;
  }
  if (expected_offset != root->data_file_length()) {
    return unexpected(make_error(StoreErrorCode::data_file_mismatch,
                                 "record ranges do not cover the data file",
                                 data_path));
  }
  return result;
}

Expected<std::vector<std::byte>> read_record_payload(
    const std::filesystem::path& data_path, const RecordMetadata& metadata,
    bool verify_crc) {
  if (metadata.byte_length > std::numeric_limits<std::size_t>::max() ||
      metadata.byte_offset >
          static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    return unexpected(StoreError{StoreErrorCode::arithmetic_overflow,
                                 "record range cannot be represented by this platform",
                                 metadata.record_id, metadata.byte_offset, data_path});
  }
  std::vector<std::byte> payload;
  try {
    payload.resize(static_cast<std::size_t>(metadata.byte_length));
  } catch (const std::bad_alloc&) {
    return unexpected(StoreError{StoreErrorCode::allocation_failure,
                                 "could not allocate record payload",
                                 metadata.record_id, metadata.byte_offset, data_path});
  }

  const int descriptor = ::open(data_path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    return unexpected(StoreError{StoreErrorCode::io_error,
                                 "could not open the data file",
                                 metadata.record_id, metadata.byte_offset, data_path});
  }
  std::size_t consumed = 0;
  while (consumed < payload.size()) {
    const std::size_t remaining = payload.size() - consumed;
    const std::size_t chunk_size = std::min<std::size_t>(
        remaining, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    const std::uint64_t absolute_offset =
        metadata.byte_offset + static_cast<std::uint64_t>(consumed);
    const ssize_t count = ::pread(
        descriptor, payload.data() + consumed, chunk_size,
        static_cast<off_t>(absolute_offset));
    if (count < 0) {
      const int saved_errno = errno;
      ::close(descriptor);
      (void)saved_errno;
      return unexpected(StoreError{StoreErrorCode::io_error,
                                   "offset-based data read failed",
                                   metadata.record_id, absolute_offset, data_path});
    }
    if (count == 0) {
      ::close(descriptor);
      return unexpected(StoreError{StoreErrorCode::truncated_read,
                                   "data file ended before the record was complete",
                                   metadata.record_id, absolute_offset, data_path});
    }
    consumed += static_cast<std::size_t>(count);
  }
  if (::close(descriptor) != 0) {
    return unexpected(StoreError{StoreErrorCode::io_error,
                                 "could not close the data file after reading",
                                 metadata.record_id, metadata.byte_offset, data_path});
  }
  if (verify_crc && payload_crc32(payload) != metadata.crc32) {
    payload.clear();
    return unexpected(StoreError{StoreErrorCode::checksum_mismatch,
                                 "record checksum does not match metadata",
                                 metadata.record_id, metadata.byte_offset, data_path});
  }
  return payload;
}

Expected<std::uint64_t> verify_all_records(const LoadedStoreIndex& index) {
  std::uint64_t verified = 0;
  for (const auto& metadata : index.records) {
    auto payload = read_record_payload(index.data_path, metadata, true);
    if (!payload) {
      return unexpected(payload.error());
    }
    ++verified;
  }
  return verified;
}

}  // namespace prism::storage
