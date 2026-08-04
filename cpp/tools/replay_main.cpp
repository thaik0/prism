#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <CLI/CLI.hpp>

#include "prism/storage/error.hpp"
#include "prism/storage/tooling.hpp"

int main(int argc, char** argv) {
  CLI::App app{"Deterministically replay storage operations against a Prism store"};
  std::filesystem::path store_directory;
  std::filesystem::path trace_path;
  std::filesystem::path output_path;
  std::uint64_t capacity_bytes = 0;
  app.add_option("--store-dir", store_directory, "Store directory")->required();
  app.add_option("--capacity-bytes", capacity_bytes, "Fast-tier byte capacity")
      ->required();
  app.add_option("--trace", trace_path, "Input JSON Lines trace")->required();
  app.add_option("--output", output_path, "Deterministic JSON report")->required();
  CLI11_PARSE(app, argc, argv);

  auto report = prism::storage::replay_trace_json(
      store_directory, capacity_bytes, trace_path);
  if (!report) {
    std::cerr << prism::storage::to_string(report.error().code) << ": "
              << report.error().message << '\n';
    return EXIT_FAILURE;
  }
  auto written =
      prism::storage::write_deterministic_report(output_path, report->json);
  if (!written) {
    std::cerr << prism::storage::to_string(written.error().code) << ": "
              << written.error().message << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "Replayed operations; expected_outcome_mismatches="
            << report->expected_outcome_mismatch_count << '\n';
  return report->expected_outcome_mismatch_count == 0 ? EXIT_SUCCESS : 2;
}
