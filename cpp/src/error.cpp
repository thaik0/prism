#include "prism/storage/error.hpp"

namespace prism::storage {

const char* to_string(StoreErrorCode code) noexcept {
  switch (code) {
    case StoreErrorCode::unknown_record:
      return "unknown_record";
    case StoreErrorCode::duplicate_record:
      return "duplicate_record";
    case StoreErrorCode::invalid_target_set:
      return "invalid_target_set";
    case StoreErrorCode::insufficient_capacity:
      return "insufficient_capacity";
    case StoreErrorCode::oversized_record:
      return "oversized_record";
    case StoreErrorCode::index_corrupt:
      return "index_corrupt";
    case StoreErrorCode::unsupported_format_version:
      return "unsupported_format_version";
    case StoreErrorCode::data_file_mismatch:
      return "data_file_mismatch";
    case StoreErrorCode::truncated_read:
      return "truncated_read";
    case StoreErrorCode::checksum_mismatch:
      return "checksum_mismatch";
    case StoreErrorCode::io_error:
      return "io_error";
    case StoreErrorCode::allocation_failure:
      return "allocation_failure";
    case StoreErrorCode::arithmetic_overflow:
      return "arithmetic_overflow";
    case StoreErrorCode::invalid_configuration:
      return "invalid_configuration";
    case StoreErrorCode::destination_exists:
      return "destination_exists";
    case StoreErrorCode::malformed_manifest:
      return "malformed_manifest";
  }
  return "unknown_error_code";
}

}  // namespace prism::storage
