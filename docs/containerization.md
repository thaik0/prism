# Linux Container Batch Contract

## Purpose

Prism Cloud Phase 1 packages the existing Python pipeline, C++17 storage engine,
and private pybind11 extension as a portable Linux batch workload. The container
is orchestration only: `prism.experiments` still owns the controlled pipeline,
and `prism.native` still owns deterministic native-store execution and parity.

## Image architecture

`Dockerfile` has two stages:

- `build-test` installs GCC 12.2, CMake 3.25, Ninja, and zlib headers. As an
  unprivileged build user it runs Debug, ASan, and UBSan CTest suites. It then
  builds and installs the existing scikit-build-core wheel, imports
  `prism._native`, runs the forced native parity fixture, and runs Python tests.
- `runtime` starts again from the slim Python base, installs only the Prism
  wheel and its runtime Python dependencies plus `libstdc++6` and `zlib1g`, and
  runs as uid/gid 10001. It contains no source tree, compiler, CMake, tests,
  Docker client/socket, or build cache.

The base image is digest-pinned. NumPy, SciPy, scikit-learn, and the build-stage
pytest version are pinned in `container/constraints.txt`. C++ archive versions
remain pinned by the existing `cpp/CMakeLists.txt` contract.

## Build

```bash
docker build --no-cache \
  --build-arg PRISM_GIT_REVISION="$(git rev-parse HEAD)" \
  -t prism:phase1 .
```

If the local Docker credential helper is unavailable, use a deliberately empty
temporary `DOCKER_CONFIG`; this changes only the client authentication context
for public pulls.

## Versioned input spec

`container/phase1-experiment.json` is schema version 1:

```json
{
  "experiment_id": "baseline__seed_1729",
  "experiment_manifest": "configs/milestone5_experiments.json",
  "kind": "accepted_milestone5_native_parity",
  "schema_version": 1
}
```

The manifest path must be relative and remain within `/input`. The experiment ID
must already exist in the strict frozen Milestone 5 manifest. The spec cannot
define algorithms, stages, policies, arbitrary commands, or output paths.

## Batch invocation

Create an empty host output writable by the runtime uid, then mount the
repository input read-only:

```bash
mkdir -p /tmp/prism-phase1-output
chmod 0777 /tmp/prism-phase1-output

docker run --rm \
  --mount type=bind,src="$PWD",dst=/input,readonly \
  --mount type=bind,src=/tmp/prism-phase1-output,dst=/output \
  prism:phase1 \
  prism-container-run \
  --spec /input/container/phase1-experiment.json \
  --output-dir /output
```

`PRISM_INPUT_ROOT=/input` and `PRISM_OUTPUT_ROOT=/output` are fixed in the
runtime. Symlink-resolved inputs must remain inside `/input`; the output must be
empty and remain inside `/output`. No Docker socket or privileged mode is used.
Missing or invalid inputs, occupied or unwritable output, escaped paths, failed
Prism stages, and native parity failures return nonzero status.

The successful output is:

```text
/output/
  experiment/       existing Milestone 5 one-run artifact tree
  native/           existing four-entry native parity artifact tree
  run_manifest.json deterministic Phase 1 provenance and artifact hashes
```

## Deterministic manifest

`run_manifest.json` records:

- schema version, Prism package version, and build-supplied Git revision;
- experiment kind/ID, exact spec hash, and relevant input configuration hashes;
- Python, OS, machine, NumPy, SciPy, and scikit-learn identification;
- native build type and compiler identification;
- SHA-256 for every canonical artifact below the output root.

It deliberately contains no timestamp, UUID, username, home or host path,
container ID, hostname, credential, or absolute mounted path. The manifest is
written only after the pipeline and native parity gates pass.

## Determinism and parity

Run `prism-container-verify` against two output roots:

```bash
prism-container-verify \
  --left /tmp/prism-phase1-output-1 \
  --right /tmp/prism-phase1-output-2 \
  --mode repeat
```

`repeat` requires identical path sets and bytes for every artifact and manifest.

For native macOS versus Linux container output, use `--mode cross-platform`.
The contract is:

- byte-identical resolved workload/simulation configs, workload config,
  observable events, summary, native `store.data`, native `store.index`, and
  native operation JSONL;
- identical NPZ array names, shapes, and all discrete/string/boolean values;
- floating NPZ arrays and semantic report numbers within absolute `1e-9`, with
  gate/status/policy/ID values exact;
- identical package/source/spec identity and independently validated artifact
  hashes in both manifests.

Hidden-ground-truth JSON is numerical rather than byte-gated because verified
macOS/Linux math-library differences changed only last-bit probability values;
the observable event stream and summary remained byte-identical.

## Representative scope and limitations

The accepted baseline seed is representative because it covers the complete
controlled learned pipeline and 130,349 existing Python/native operation
comparisons without creating a new scientific experiment. It does not execute
the much larger Milestone 5 sweep or LLMServingSim.

The runtime remains a research batch image: synchronous native calls hold the
GIL, ordinary slow-tier reads use the OS page cache, and there is no cloud
service, concurrent serving path, mutable storage, or performance claim.
