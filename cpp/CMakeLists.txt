cmake_minimum_required(VERSION 3.24)

project(prism_storage VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)

option(PRISM_BUILD_TESTS "Build Prism native tests" ON)
option(PRISM_BUILD_PYTHON_BINDINGS "Build the private Python extension" OFF)
option(PRISM_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(PRISM_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(PRISM_WARNINGS_AS_ERRORS "Treat Prism warnings as errors" OFF)

if(PRISM_ENABLE_ASAN AND PRISM_ENABLE_UBSAN)
  message(FATAL_ERROR "Enable ASan and UBSan in separate build directories")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_GRPCTEST OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EXPECTED_BUILD_PACKAGE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  flatbuffers
  URL https://github.com/google/flatbuffers/archive/refs/tags/v24.3.25.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  cli11
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  tl_expected
  URL https://github.com/TartanLlama/expected/archive/refs/tags/v1.1.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(flatbuffers nlohmann_json cli11 tl_expected)
find_package(ZLIB REQUIRED)

set(PRISM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(PRISM_GENERATED_HEADER "${PRISM_GENERATED_DIR}/prism_store_generated.h")
add_custom_command(
  OUTPUT "${PRISM_GENERATED_HEADER}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${PRISM_GENERATED_DIR}"
  COMMAND "$<TARGET_FILE:flatc>" --cpp --scoped-enums
          -o "${PRISM_GENERATED_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/schema/prism_store.fbs"
  DEPENDS flatc "${CMAKE_CURRENT_SOURCE_DIR}/schema/prism_store.fbs"
  VERBATIM
)
add_custom_target(prism_flatbuffers_generated DEPENDS "${PRISM_GENERATED_HEADER}")

function(prism_configure_owned_target target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(PRISM_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
    )
    if(PRISM_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(PRISM_ENABLE_ASAN)
      target_compile_options(${target} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address)
    endif()
    if(PRISM_ENABLE_UBSAN)
      target_compile_options(${target} PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=undefined)
    endif()
  endif()
endfunction()

add_library(prism_storage
  src/builder.cpp
  src/error.cpp
  src/store_format.cpp
  src/tiered_store.cpp
  src/tooling.cpp
  "${PRISM_GENERATED_HEADER}"
)
add_dependencies(prism_storage prism_flatbuffers_generated)
set_target_properties(prism_storage PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(prism_storage
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  SYSTEM PRIVATE "${PRISM_GENERATED_DIR}"
)
target_link_libraries(prism_storage
  PUBLIC tl::expected
  PRIVATE flatbuffers nlohmann_json::nlohmann_json ZLIB::ZLIB
)
prism_configure_owned_target(prism_storage)

if(PRISM_BUILD_PYTHON_BINDINGS)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 3.0.4 CONFIG REQUIRED)
  pybind11_add_module(prism_native MODULE src/python_bindings.cpp)
  target_link_libraries(prism_native PRIVATE prism_storage)
  set_target_properties(prism_native PROPERTIES OUTPUT_NAME "_native")
  prism_configure_owned_target(prism_native)
  install(TARGETS prism_native LIBRARY DESTINATION prism)
endif()

add_executable(prism_store_build tools/build_main.cpp)
target_link_libraries(prism_store_build PRIVATE prism_storage CLI11::CLI11 nlohmann_json::nlohmann_json)
prism_configure_owned_target(prism_store_build)

add_executable(prism_store_inspect tools/inspect_main.cpp)
target_link_libraries(prism_store_inspect PRIVATE prism_storage CLI11::CLI11)
prism_configure_owned_target(prism_store_inspect)

add_executable(prism_store_replay tools/replay_main.cpp)
target_link_libraries(prism_store_replay PRIVATE prism_storage CLI11::CLI11)
prism_configure_owned_target(prism_store_replay)

if(PRISM_BUILD_TESTS)
  FetchContent_Declare(
    catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.6.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(catch2)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras" "${catch2_SOURCE_DIR}/contrib")
  include(Catch)

  add_executable(prism_storage_tests
    tests/test_builder.cpp
    tests/test_tiered_store.cpp
    tests/test_tooling.cpp
    tests/test_store_format.cpp
  )
  add_dependencies(prism_storage_tests prism_flatbuffers_generated)
  target_include_directories(prism_storage_tests SYSTEM PRIVATE "${PRISM_GENERATED_DIR}")
  target_compile_definitions(prism_storage_tests PRIVATE
    PRISM_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests"
  )
  target_link_libraries(prism_storage_tests PRIVATE prism_storage Catch2::Catch2WithMain flatbuffers ZLIB::ZLIB nlohmann_json::nlohmann_json)
  prism_configure_owned_target(prism_storage_tests)
  catch_discover_tests(prism_storage_tests)
endif()
