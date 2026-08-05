#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>

#include <CLI/CLI.hpp>

#include "prism/storage/error.hpp"
#include "prism/storage/tooling.hpp"

int main(int argc, char** argv) {
  CLI::App app{"Inspect and optionally checksum a Prism immutable store"};
  std::filesystem::path store_directory;
  bool verify_all = false;
  std::uint64_t capacity_bytes = 0;
  auto* capacity_option = app.add_option(
      "--capacity-bytes", capacity_bytes, "Evaluate fast-tier capacity feasibility");
  app.add_option("--store-dir", store_directory, "Store directory")->required();
  app.add_flag("--verify-all", verify_all, "Read and checksum every record");
  CLI11_PARSE(app, argc, argv);

  const std::optional<std::uint64_t> capacity =
      capacity_option->count() > 0 ? std::optional<std::uint64_t>(capacity_bytes)
                                   : std::nullopt;
  auto report = prism::storage::inspect_store_json(store_directory, verify_all,
                                                    capacity);
  if (!report) {
    std::cerr << prism::storage::to_string(report.error().code) << ": "
              << report.error().message << '\n';
    return EXIT_FAILURE;
  }
  std::cout << *report;
  return EXIT_SUCCESS;
}
