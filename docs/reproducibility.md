# Reproducing Prism

## Supported environment

Prism's Python and native paths support POSIX macOS and Linux. Use:

- Python 3.11 or newer;
- CMake 3.24 or newer;
- a C++17 compiler;
- zlib development headers and library;
- NumPy, SciPy, scikit-learn, and pytest through the Python environment.

The LLMServingSim integration additionally requires Git submodules and Docker.
Its runtime is pinned to an amd64 container image and was verified under Docker
Desktop on an arm64 macOS host. Windows is not supported by the native engine.

## Installation and Python verification

From the repository root:

```bash
python3 -m pip install -e .
python3 -c "import prism; import prism._native; import prism.native; print(prism.__version__)"
python3 -m pytest -q
python3 -m compileall -q src tests
python3 -m pip check
```

The editable install builds the private native extension and may fetch pinned C++
dependency archives into the ignored `build` tree on first use.

## Native Debug and sanitizer suites

Debug with warnings treated as errors:

```bash
cmake -S cpp -B build/cpp-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPRISM_BUILD_TESTS=ON \
  -DPRISM_WARNINGS_AS_ERRORS=ON
cmake --build build/cpp-debug --parallel
ctest --test-dir build/cpp-debug --output-on-failure
```

AddressSanitizer:

```bash
cmake -S cpp -B build/cpp-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPRISM_BUILD_TESTS=ON \
  -DPRISM_WARNINGS_AS_ERRORS=ON \
  -DPRISM_ENABLE_ASAN=ON
cmake --build build/cpp-asan --parallel
ctest --test-dir build/cpp-asan --output-on-failure
```

UndefinedBehaviorSanitizer:

```bash
cmake -S cpp -B build/cpp-ubsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPRISM_BUILD_TESTS=ON \
  -DPRISM_WARNINGS_AS_ERRORS=ON \
  -DPRISM_ENABLE_UBSAN=ON
cmake --build build/cpp-ubsan --parallel
ctest --test-dir build/cpp-ubsan --output-on-failure
```

ASan and UBSan must use separate build directories. The project rejects enabling
both in one configuration.

## Representative controlled pipeline

Each output directory must be absent or empty. The following commands reproduce
the accepted small structure run and the dedicated 1,000-window predictor run:

```bash
PYTHONPATH=src python3 -m prism.workload.cli \
  --config configs/milestone1_representative.json \
  --output-dir /tmp/prism_repro_m1

PYTHONPATH=src python3 -m prism.workload.validate \
  --run-dir /tmp/prism_repro_m1 \
  --require-demonstrations

PYTHONPATH=src python3 -m prism.structure.cli \
  --run-dir /tmp/prism_repro_m1 \
  --config configs/milestone2_representative.json \
  --output-dir /tmp/prism_repro_m2

PYTHONPATH=src python3 -m prism.workload.cli \
  --config configs/milestone3_predictor_workload.json \
  --output-dir /tmp/prism_repro_m3_source

PYTHONPATH=src python3 -m prism.workload.validate \
  --run-dir /tmp/prism_repro_m3_source \
  --require-demonstrations \
  --require-intensity-signal

PYTHONPATH=src python3 -m prism.predictor.cli \
  --run-dir /tmp/prism_repro_m3_source \
  --structure-config configs/milestone2_representative.json \
  --config configs/milestone3_predictor.json \
  --output-dir /tmp/prism_repro_m3

PYTHONPATH=src python3 -m prism.simulation.cli \
  --run-dir /tmp/prism_repro_m3_source \
  --predictor-run-dir /tmp/prism_repro_m3 \
  --config configs/milestone4_simulation.json \
  --output-dir /tmp/prism_repro_m4
```

## Frozen scientific experiments

These commands are deterministic but substantially more expensive than the test
suite. The committed final report relies on their accepted checked-in result
documents; rerunning them is not required for ordinary verification.

```bash
PYTHONPATH=src python3 -m prism.experiments.cli \
  --manifest configs/milestone5_experiments.json \
  --output-dir /tmp/prism_repro_m5

PYTHONPATH=src python3 -m prism.experiments.cli \
  --manifest configs/milestone55_actionability.json \
  --output-dir /tmp/prism_repro_m55
```

The first command runs 36 Milestone 1--4 pipelines. The second runs 27. Use
`--resume` with the same output path to hash-validate and reuse completed runs.

## Native tools and Python/C++ parity

After the Debug build:

```bash
build/cpp-debug/prism_store_build \
  --manifest cpp/tests/fixtures/store_manifest.json \
  --output-dir /tmp/prism_repro_store

build/cpp-debug/prism_store_inspect \
  --store-dir /tmp/prism_repro_store \
  --capacity-bytes 40 \
  --verify-all

build/cpp-debug/prism_store_replay \
  --store-dir /tmp/prism_repro_store \
  --capacity-bytes 40 \
  --trace cpp/tests/fixtures/replay.jsonl \
  --output /tmp/prism_repro_replay.json

python3 -m prism.native.cli \
  --fixture \
  --output-dir /tmp/prism_repro_native_fixture
```

The accepted full parity run uses the source and predictor paths produced above:

```bash
python3 -m prism.native.cli \
  --run-dir /tmp/prism_repro_m3_source \
  --predictor-run-dir /tmp/prism_repro_m3 \
  --simulation-config configs/milestone4_simulation.json \
  --output-dir /tmp/prism_repro_native
```

## Pinned LLMServingSim integration

Initialize the recursively pinned upstream and run the bundled ten-request smoke
trace:

```bash
git submodule update --init --recursive third_party/LLMServingSim

PYTHONPATH=src python3 -m prism.llm_sim.cli \
  --config configs/milestone8_llmservingsim.json \
  --output-dir /tmp/prism_repro_m8_tiny \
  --workers 6 \
  --tiny
```

Remove `--tiny` to run the accepted 300-request, six-policy experiment. Each
policy runs from a clean simulator state. The first invocation builds the pinned
container and upstream simulator; host Docker configuration may require the
environment-specific settings described in
[the Milestone 8 execution plan](execution_plan_milestone8.md).

## Artifact policy

Generated workload, learner, predictor, simulation, parity, and LLM simulator
roots are written to the chosen output directories. They include canonical JSON,
JSONL, NPZ, native store, and simulator files described by the corresponding
module documentation.

Large generated experiment roots, build trees, Docker layers, wheels, native
stores, raw simulator logs, and temporary payloads are intentionally not
committed. Committed evidence consists of source/configuration, deterministic
tests, exact upstream pins, and the compact result reports. Keep generated roots
outside the repository, normally under `/tmp`, to avoid accidental release
artifacts.
