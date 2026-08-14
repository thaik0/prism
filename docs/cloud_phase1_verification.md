# Prism Cloud Phase 1 Verification Record

## Environment and inspected assumptions

Verification ran on August 14, 2026 from branch `cloud/milestone1`, based on the
`v1.0.0` closeout commit and `origin` remote. The host used Python 3.12.10,
CMake 4.4.2, and Apple Clang 21. Docker Desktop 29.6.1 supplied Linux/aarch64.

Prism requires Python 3.11+, CMake 3.24+, C++17, zlib, NumPy, SciPy, and
scikit-learn. Native CMake dependencies are existing pinned archives:
FlatBuffers 24.3.25, nlohmann/json 3.11.3, CLI11 2.4.2, tl::expected 1.1.0,
and test-only Catch2 3.6.0. The wheel path remains scikit-build-core 0.11.6 plus
pybind11 3.0.4 and links the existing `prism_storage` target.

The only Linux-specific correction was environmental: the unreadable-file
native test must not run as root because root can bypass its permission model.
All native build/test steps now run as an unprivileged build user. No C++ or
research behavior changed.

## Build-stage results

The final image command was equivalent to:

```bash
docker build --no-cache \
  --build-arg PRISM_GIT_REVISION="$(git rev-parse HEAD)" \
  -t prism:phase1 .
```

Results:

- Debian Bookworm, GCC 12.2.0, CMake 3.25.1, zlib 1.2.13;
- Debug CTest: 29/29 passed;
- AddressSanitizer CTest: 29/29 passed;
- UndefinedBehaviorSanitizer CTest: 29/29 passed;
- `import prism._native`: passed;
- forced native parity smoke: passed for all four policies with zero mismatches;
- installed-wheel Linux Python suite: 283 passed, one existing opt-in
  LLMServingSim Docker integration test skipped;
- `pip check`: no broken requirements.

Host pre-edit native baselines also passed: 29/29 CTest and 25 focused native
Python tests. The final host suite was 283 passed, one opt-in test skipped.

## Runtime and representative output

The runtime inspected as uid/gid `10001:10001`, approximately 179 MB, with only
the installed wheel/runtime dependencies and linked `libz`, `libstdc++`,
`libgcc_s`, `libc`, and `libm`. The accepted `baseline__seed_1729` run completed
twice from a read-only repository mount into separate output mounts.

Each successful run produced 29 hashed canonical artifacts plus
`run_manifest.json`. Representative evidence included:

- full experiment status `completed`;
- all existing simulation scientific gates passed;
- predictive-greedy test combined cost `57764.0` and hit rate
  `0.5407038748666904`;
- 130,349 Python/native operations across four policies;
- zero native mismatches, invalidations, unexpected exceptions, or capacity
  violations;
- native payload/store verification and overall parity passed.

This is the existing accepted experiment behavior, not a new result or claim.

## Determinism and native/container parity

Two independent Linux runtime roots passed `--mode repeat` with all 30 files
byte-identical.

The host-native and Linux roots passed `--mode cross-platform`:

- 8 byte-identical classified artifacts;
- 6 NPZ numerical artifacts with exact shapes and discrete arrays and all
  floating values within absolute `1e-9`;
- 6 semantic JSON reports with exact IDs/gates/status and numerical values
  within absolute `1e-9`.

The observable event stream, resolved configs, summary, native store data/index,
and native operation log were byte-identical. Hidden-ground-truth probability
fields differed only at roughly `1e-16`, so that artifact is semantically
numerical rather than byte-classified. The two Linux roots retained complete
byte identity, including hidden truth.

## Failure and isolation results

Deliberate runtime checks produced:

| Check | Exit | Result |
|---|---:|---|
| Missing experiment spec | 2 | rejected as missing/non-regular input |
| Invalid experiment ID | 2 | rejected outside the accepted manifest |
| Output path outside `/output` | 2 | rejected before execution |
| Read-only output mount | 2 | write failure propagated |
| Mutation attempt under read-only `/input` | 1 | kernel returned read-only filesystem |
| Invalid structure config during real Prism run | 2 | failed stage propagated; `run_status.json` retained `StructureLearnerConfigError` |

The successful runner never modified mounted input. Output artifacts appeared
only under the selected `/output` mount.

## Security and image inspection

- Runtime config user is `10001:10001`; no privileged mode or Docker socket is
  used.
- No repository source, tests, compiler, CMake, build directory, `.git`, `.aws`,
  `.ssh`, `.env`, private key, or user credential was copied into the runtime.
- Standard CA certificate files are present, as required by the Python base;
  these are public trust roots, not credentials.
- The macOS workspace path and user name were absent from installed Prism files
  and generated deterministic manifests.
- `/input` and `/output` are the only runner boundaries; the runtime has no user
  home dependency and contains no host-specific absolute path.

## Scope confirmation and remaining limitation

No AWS or Phase 2 infrastructure was added. No Prism research function was
duplicated or modified. The representative run uses only existing Prism public
or documented interfaces.

Linux/aarch64 is fully verified. The digest-pinned base and Python dependencies
publish Linux/amd64 variants, but amd64 execution was not separately tested.
Package and C++ source versions are pinned where meaningful; Debian package
security revisions still follow the pinned Bookworm base repositories at build
time.
