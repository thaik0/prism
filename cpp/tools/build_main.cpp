#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "prism/storage/builder.hpp"
#include "prism/storage/error.hpp"

int main(int argc, char** argv) {
  CLI::App app{"Build a deterministic Prism immutable record store"};
  std::filesystem::path manifest;
  std::filesystem::path output_directory;
  bool json_output = false;
  app.add_option("--manifest", manifest, "Input JSON manifest")->required();
  app.add_option("--output-dir", output_directory, "Final store directory")
      ->required();
  app.add_flag("--json", json_output, "Print a machine-readable result");
  CLI11_PARSE(app, argc, argv);

  auto result = prism::storage::build_store(manifest, output_directory);
  if (!result) {
    if (json_output) {
      nlohmann::ordered_json report = {
          {"success", false},
          {"error_code", prism::storage::to_string(result.error().code)},
          {"message", result.error().message},
      };
      std::cout << report.dump(2) << '\n';
    } else {
      std::cerr << prism::storage::to_string(result.error().code) << ": "
                << result.error().message << '\n';
    }
    return EXIT_FAILURE;
  }
  if (json_output) {
    nlohmann::ordered_json report = {
        {"success", true},
        {"record_count", result->record_count},
        {"data_bytes", result->data_bytes},
    };
    std::cout << report.dump(2) << '\n';
  } else {
    std::cout << "Built Prism store: records=" << result->record_count
              << ", data_bytes=" << result->data_bytes << '\n';
  }
  return EXIT_SUCCESS;
}
