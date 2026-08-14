# Prism Cloud Phase 1 Execution Plan

## Status and boundary

**Status:** Complete

This plan adds only a portable Linux batch boundary around the completed Prism
repository. It does not change a workload, learner, predictor, projection,
placement policy, native-engine semantic, evaluation gate, or scientific
conclusion. AWS and later cloud infrastructure are outside this phase.

## Repository evidence inspected

- `README.md`, the technical plan, decisions, milestones, reproducibility guide,
  native-engine and Python/C++ integration documents, and final experiment
  reports;
- `pyproject.toml`, `cpp/CMakeLists.txt`, pinned CMake dependencies, public CLIs,
  canonical persistence code, representative configurations, Python tests, and
  all four native CTest sources;
- branch `cloud/milestone1`, the `origin` remote, the `v1.0.0` closeout commit,
  recent history, and the pre-existing untracked state inside the
  `third_party/LLMServingSim` submodule.

The pre-edit baseline was 29/29 native CTest cases and 25 focused native Python
tests. The existing submodule state was not changed.

## Implementation plan

1. Use the digest-pinned official `python:3.12.10-slim-bookworm` base. Debian
   Bookworm supplies CMake 3.25, GCC 12.2 with C++17, and zlib, matching Prism's
   documented Python 3.11+/CMake 3.24+ POSIX support.
2. In an unprivileged build/test stage, build and test standalone Debug, ASan,
   and UBSan native configurations. Build the existing scikit-build-core wheel,
   which compiles the private pybind11 extension against the same
   `prism_storage` target. Install that wheel, smoke the extension, and run the
   Linux Python regression suite.
3. Copy only application wheels into a smaller runtime stage. Install only
   Python runtime dependencies plus `libstdc++6` and `zlib1g`, and run as fixed
   uid/gid 10001 without a home directory.
4. Add one version-1 container spec that selects one experiment from the
   existing frozen Milestone 5 manifest. The runner calls existing experiment
   orchestration and then the existing representative native parity harness.
5. Select accepted `baseline__seed_1729`: it exercises seeded workload
   generation, NMF structure learning, activation/intensity prediction,
   projection, placement evaluation, canonical artifacts, the pybind extension,
   and all four certified native policy paths in one bounded run.
6. Require exact complete-tree identity for repeated Linux runs. Across native
   macOS and Linux, require byte identity for configs, observable events,
   summaries, native store bytes/index, and native operations; exact discrete
   NPZ arrays; and absolute `1e-9` equivalence for numerical arrays/reports.

## Assumptions and risks

- Building needs network access to Debian, Python, and the pinned CMake source
  archives. The runtime needs no network.
- A host bind-mounted output must be writable by uid 10001.
- BLAS, libc, and compiler differences can change last-bit floating values.
  Cross-platform numerical comparison therefore uses Prism's existing
  `1e-9` evaluation tolerance, while same-container determinism remains exact.
- The image was verified on Linux/aarch64 under Docker Desktop. The pinned base
  and Python wheels are multi-architecture, but Linux/amd64 was not separately
  executed in this phase.

## Explicit non-goals

- AWS, S3, ECR, Batch, ECS, CloudWatch, IAM, Terraform, Kubernetes, or CI
  deployment;
- Docker socket access from the workload container;
- LLMServingSim execution inside the Phase 1 runtime;
- new models, policies, experiments, scientific gates, or research claims;
- concurrency, asynchronous I/O, production serving, or Phase 2 work.

## Completion checkpoints

- [x] Linux multi-stage native/pybind build and test boundary.
- [x] Versioned runner, deterministic manifest, parity verifier, and focused
  failure/security tests.
- [x] Host regression, two runtime runs, repeat determinism, native/container
  parity, failure-mode checks, security inspection, and documentation.
- [x] Three bounded milestone commits pushed to `origin/cloud/milestone1`.
