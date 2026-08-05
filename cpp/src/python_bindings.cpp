#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "prism/storage/builder.hpp"
#include "prism/storage/error.hpp"
#include "prism/storage/store_format.hpp"
#include "prism/storage/tiered_store.hpp"

namespace py = pybind11;

namespace {

using prism::storage::BuildRecord;
using prism::storage::Expected;
using prism::storage::RecordId;
using prism::storage::RecordMetadata;
using prism::storage::ResidencySnapshot;
using prism::storage::StoreError;
using prism::storage::StoreErrorCode;
using prism::storage::StoreStats;
using prism::storage::TieredStore;

[[noreturn]] void raise_native_error(const StoreError& error,
                                     const py::object& error_type) {
  py::object instance = error_type(py::str(error.message));
  instance.attr("code") = py::str(prism::storage::to_string(error.code));
  instance.attr("message") = py::str(error.message);
  instance.attr("record_id") = py::none();
  instance.attr("offset") = py::none();
  instance.attr("path") = py::none();
  if (error.record_id.has_value()) {
    instance.attr("record_id") = py::cast(*error.record_id);
  }
  if (error.byte_offset.has_value()) {
    instance.attr("offset") = py::cast(*error.byte_offset);
  }
  if (error.path.has_value()) {
    instance.attr("path") = py::cast(error.path->string());
  }
  PyErr_SetObject(error_type.ptr(), instance.ptr());
  throw py::error_already_set();
}

template <typename T>
T unwrap(Expected<T>&& result, const py::object& error_type) {
  if (!result) {
    raise_native_error(result.error(), error_type);
  }
  return std::move(*result);
}

std::uint64_t require_unsigned_integer(py::handle value,
                                       const char* field_name) {
  if (PyBool_Check(value.ptr()) != 0 || PyIndex_Check(value.ptr()) == 0) {
    throw py::type_error(std::string(field_name) +
                         " must be a non-boolean integer");
  }
  py::object index =
      py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
  if (!index) {
    throw py::error_already_set();
  }
  const unsigned long long converted = PyLong_AsUnsignedLongLong(index.ptr());
  if (PyErr_Occurred() != nullptr) {
    throw py::error_already_set();
  }
  static_assert(sizeof(unsigned long long) >= sizeof(std::uint64_t));
  if constexpr (sizeof(unsigned long long) > sizeof(std::uint64_t)) {
    if (converted > std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error(std::string(field_name) +
                                " exceeds uint64 range");
    }
  }
  return static_cast<std::uint64_t>(converted);
}

std::filesystem::path require_path(py::handle value, const char* field_name) {
  py::object path =
      py::reinterpret_steal<py::object>(PyOS_FSPath(value.ptr()));
  if (!path) {
    PyErr_Clear();
    throw py::type_error(std::string(field_name) +
                         " must be str or os.PathLike");
  }
  if (PyUnicode_Check(path.ptr()) != 0) {
    return std::filesystem::u8path(py::cast<std::string>(path));
  }
  if (PyBytes_Check(path.ptr()) != 0) {
    return std::filesystem::path(py::cast<std::string>(path));
  }
  throw py::type_error(std::string(field_name) +
                       " must resolve to str or bytes");
}

py::sequence require_sequence(py::handle value, const char* field_name) {
  if (PySequence_Check(value.ptr()) == 0 ||
      PyUnicode_Check(value.ptr()) != 0 || PyBytes_Check(value.ptr()) != 0 ||
      PyByteArray_Check(value.ptr()) != 0) {
    throw py::type_error(std::string(field_name) +
                         " must be a finite sequence");
  }
  return py::reinterpret_borrow<py::sequence>(value);
}

std::vector<RecordId> require_record_ids(py::handle value) {
  const py::sequence sequence = require_sequence(value, "record_ids");
  std::vector<RecordId> result;
  result.reserve(static_cast<std::size_t>(sequence.size()));
  for (const py::handle item : sequence) {
    result.push_back(require_unsigned_integer(item, "record_id"));
  }
  return result;
}

std::vector<BuildRecord> require_build_records(py::handle value) {
  const py::sequence sequence = require_sequence(value, "records");
  std::vector<BuildRecord> result;
  result.reserve(static_cast<std::size_t>(sequence.size()));
  for (const py::handle item : sequence) {
    if (PyTuple_Check(item.ptr()) == 0) {
      throw py::type_error("each record must be a (record_id, bytes) tuple");
    }
    const py::tuple row = py::reinterpret_borrow<py::tuple>(item);
    if (row.size() != 2) {
      throw py::value_error("each record tuple must contain exactly two items");
    }
    const RecordId record_id =
        require_unsigned_integer(row[0], "record_id");
    if (PyBytes_Check(row[1].ptr()) == 0) {
      throw py::type_error("record payload must be immutable bytes");
    }
    const std::string payload = py::cast<std::string>(row[1]);
    std::vector<std::byte> native_payload;
    native_payload.reserve(payload.size());
    for (const char byte : payload) {
      native_payload.push_back(
          static_cast<std::byte>(static_cast<unsigned char>(byte)));
    }
    result.push_back(BuildRecord{record_id, std::move(native_payload)});
  }
  return result;
}

py::dict snapshot_dict(const ResidencySnapshot& snapshot) {
  py::dict result;
  result["resident_record_ids"] = py::cast(snapshot.resident_record_ids);
  result["resident_bytes"] = snapshot.resident_bytes;
  result["capacity_bytes"] = snapshot.capacity_bytes;
  return result;
}

py::dict metadata_dict(const RecordMetadata& metadata) {
  py::dict result;
  result["record_id"] = metadata.record_id;
  result["byte_offset"] = metadata.byte_offset;
  result["byte_length"] = metadata.byte_length;
  result["crc32"] = metadata.crc32;
  return result;
}

py::dict stats_dict(const StoreStats& stats) {
  py::dict failures;
  for (std::size_t index = 0;
       index < prism::storage::store_error_code_count(); ++index) {
    const auto code = static_cast<StoreErrorCode>(index);
    failures[py::str(prism::storage::to_string(code))] =
        stats.failure_count(code);
  }

  py::dict result;
  result["successful_fast_reads"] = stats.successful_fast_reads;
  result["successful_fast_read_bytes"] = stats.successful_fast_read_bytes;
  result["successful_slow_reads"] = stats.successful_slow_reads;
  result["successful_slow_read_bytes"] = stats.successful_slow_read_bytes;
  result["promotion_source_reads"] = stats.promotion_source_reads;
  result["promotion_source_read_bytes"] = stats.promotion_source_read_bytes;
  result["committed_promotions"] = stats.committed_promotions;
  result["committed_promotion_bytes"] = stats.committed_promotion_bytes;
  result["committed_evictions"] = stats.committed_evictions;
  result["committed_eviction_bytes"] = stats.committed_eviction_bytes;
  result["target_set_calls"] = stats.target_set_calls;
  result["successful_target_set_calls"] = stats.successful_target_set_calls;
  result["failed_target_set_calls"] = stats.failed_target_set_calls;
  result["aborted_staged_bytes"] = stats.aborted_staged_bytes;
  result["current_resident_records"] = stats.current_resident_records;
  result["current_resident_bytes"] = stats.current_resident_bytes;
  result["resident_byte_high_water_mark"] =
      stats.resident_byte_high_water_mark;
  result["failures_by_code"] = std::move(failures);
  return result;
}

std::vector<RecordId> sorted_difference(const std::vector<RecordId>& left,
                                        const std::vector<RecordId>& right) {
  std::vector<RecordId> result;
  std::set_difference(left.begin(), left.end(), right.begin(), right.end(),
                      std::back_inserter(result));
  return result;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "Private synchronous bindings for the Prism storage engine";
  module.attr("STORE_FORMAT_VERSION") = prism::storage::kStoreFormatVersion;

  py::object error_type = py::reinterpret_steal<py::object>(PyErr_NewException(
      "prism._native.NativeStoreError", PyExc_Exception, nullptr));
  if (!error_type) {
    throw py::error_already_set();
  }
  module.attr("NativeStoreError") = error_type;

  module.def(
      "build_store",
      [error_type](py::handle records, py::handle output_directory) {
        auto native_records = require_build_records(records);
        const auto destination =
            require_path(output_directory, "output_directory");
        const auto built = unwrap(
            prism::storage::build_store(std::move(native_records), destination),
            error_type);
        py::dict result;
        result["format_version"] = prism::storage::kStoreFormatVersion;
        result["record_count"] = built.record_count;
        result["data_file_bytes"] = built.data_bytes;
        return result;
      },
      py::arg("records"), py::arg("output_dir"));

  py::class_<TieredStore, std::unique_ptr<TieredStore>>(module, "TieredStore")
      .def_static(
          "open",
          [error_type](py::handle store_directory, py::handle capacity) {
            const auto path = require_path(store_directory, "store_directory");
            const auto capacity_bytes =
                require_unsigned_integer(capacity, "fast_capacity_bytes");
            return unwrap(TieredStore::open(path, capacity_bytes), error_type);
          },
          py::arg("store_dir"), py::arg("fast_capacity_bytes"))
      .def(
          "read",
          [error_type](TieredStore& store, py::handle record_id) {
            const auto id = require_unsigned_integer(record_id, "record_id");
            std::vector<std::byte> payload;
            const auto info =
                unwrap(store.read_into(id, payload), error_type);
            py::dict result;
            result["payload"] = py::bytes(
                reinterpret_cast<const char*>(payload.data()),
                static_cast<py::ssize_t>(payload.size()));
            result["tier"] =
                info.tier == prism::storage::ReadTier::fast ? "fast" : "slow";
            result["byte_count"] = info.bytes;
            return result;
          },
          py::arg("record_id"))
      .def(
          "promote",
          [error_type](TieredStore& store, py::handle record_id) {
            const auto id = require_unsigned_integer(record_id, "record_id");
            const auto promoted = unwrap(store.promote(id), error_type);
            py::dict result;
            result["moved"] = !promoted.already_resident;
            result["record_id"] = id;
            result["bytes_moved"] = promoted.bytes_moved;
            return result;
          },
          py::arg("record_id"))
      .def(
          "evict",
          [error_type](TieredStore& store, py::handle record_id) {
            const auto id = require_unsigned_integer(record_id, "record_id");
            const auto evicted = unwrap(store.evict(id), error_type);
            py::dict result;
            result["moved"] = evicted.was_resident;
            result["record_id"] = id;
            result["bytes_moved"] = evicted.bytes_freed;
            return result;
          },
          py::arg("record_id"))
      .def(
          "apply_target_set",
          [error_type](TieredStore& store, py::handle record_ids) {
            auto target = require_record_ids(record_ids);
            const auto before = store.residency_snapshot();
            const auto applied =
                unwrap(store.apply_target_set(target), error_type);
            py::dict result;
            result["incoming_record_ids"] = sorted_difference(
                applied.residency.resident_record_ids,
                before.resident_record_ids);
            result["outgoing_record_ids"] = sorted_difference(
                before.resident_record_ids,
                applied.residency.resident_record_ids);
            result["promotion_count"] = applied.promotion_count;
            result["promotion_bytes"] = applied.promotion_bytes;
            result["eviction_count"] = applied.eviction_count;
            result["eviction_bytes"] = applied.eviction_bytes;
            result["target_changed"] =
                before.resident_record_ids !=
                applied.residency.resident_record_ids;
            result["residency"] = snapshot_dict(applied.residency);
            return result;
          },
          py::arg("record_ids"))
      .def("snapshot", [](const TieredStore& store) {
        return snapshot_dict(store.residency_snapshot());
      })
      .def("stats", [](const TieredStore& store) {
        return stats_dict(store.stats());
      })
      .def(
          "record_metadata",
          [error_type](const TieredStore& store, py::handle record_id) {
            const auto id = require_unsigned_integer(record_id, "record_id");
            return metadata_dict(unwrap(store.metadata(id), error_type));
          },
          py::arg("record_id"));
}
