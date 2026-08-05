# Milestone 6 Native Two-Tier Storage Engine

## Purpose and boundary

Milestone 6 supplies the real storage substrate for Prism's supported thesis:
learned latent-demand structure can feed stable, cost-aware deterministic
placement. Milestone 5.5 did not establish useful dynamic placement caused by the
current fast forecast. This engine therefore executes caller-supplied placement
decisions but contains no policy, ranking, admission, victim selection, forecast,
or ML logic.

Records are immutable and indivisible. `store.data` is the authoritative
file-backed slow tier. The fast tier owns complete payload copies in ordinary
C++ memory under one strict byte capacity. The implementation is standalone
C++17 and has no Python boundary.

## Build and dependencies

The native build uses CMake and CTest. First configure downloads explicit release
archives into the ignored build tree:

| Dependency | Version | Purpose |
|---|---:|---|
| FlatBuffers | `v24.3.25` | Versioned, verified binary index and generated accessors |
| nlohmann/json | `v3.11.3` | Manifest and JSON Lines replay parsing/reporting |
| CLI11 | `v2.4.2` | Native command-line parsing |
| tl::expected | `v1.1.0` | C++17 structured result values |
| Catch2 | `v3.6.0` | Native test framework only |
| zlib | platform package | Trusted CRC32 implementation |

Each version is pinned in `cpp/CMakeLists.txt`; no dependency source is vendored.
Warnings are applied to Prism-owned targets and can be made fatal without making
third-party warnings fatal.

```bash
cmake -S cpp -B build/cpp-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPRISM_BUILD_TESTS=ON \
  -DPRISM_WARNINGS_AS_ERRORS=ON
cmake --build build/cpp-debug --parallel
ctest --test-dir build/cpp-debug --output-on-failure
```

Supported targets are POSIX macOS and Linux with `pread`. Windows is not
supported by this milestone.

## Durable store format

One store is exactly:

```text
store/
    store.data
    store.index
```

`store.data` concatenates payload bytes in ascending record-ID order, without a
header, compression, or embedded metadata. `store.index` is a FlatBuffer using
file identifier `PRSM` and format version `1`. Its root contains:

```text
format_version:uint32
data_file_length:uint64
records:[Record]

Record:
    record_id:uint64
    byte_offset:uint64
    byte_length:uint64
    crc32:uint32
```

Records must have strictly increasing unique IDs, positive lengths, checked
contiguous nonoverlapping ranges starting at zero, and complete coverage of the
declared data length. The declared length must equal the actual file length.
CRC32 covers each complete payload and detects accidental corruption; it is not
cryptographic authentication.

Normal opening verifies the directory, both regular files, file identifier,
FlatBuffers structure, version, ordering, arithmetic, ranges, exact file length,
positive capacity, and that at least one record fits. It deliberately does not
checksum all payloads. Full checksum verification is explicit inspection work.

## Deterministic builder and publication

The builder accepts one strict JSON manifest:

```json
{
  "records": [
    {"record_id": 1, "payload_path": "payloads/record_1.bin"}
  ]
}
```

Relative payload paths resolve from the manifest directory. The builder rejects
unknown manifest shapes, duplicate IDs, missing/nonregular/unreadable files,
zero-length payloads, and unrepresentable arithmetic. Manifest order does not
affect output. The trusted zlib CRC is calculated while each payload is copied.

For output `/path/store`, construction occurs only in `/path/store.tmp`. A
nonempty final or temporary path is preserved and rejected. An existing empty
path may be removed. After both files are flushed and closed, the builder reopens
the temporary store and verifies every payload checksum, then publishes it with
one sibling directory rename. A builder-created temporary directory is removed
after failure. This gives atomic final visibility on the same supported
filesystem; it is not a crash-durable transaction or recovery journal.

```bash
build/cpp-debug/prism_store_build \
  --manifest cpp/tests/fixtures/store_manifest.json \
  --output-dir /tmp/prism_store
```

`--json` emits a stable machine-readable success or error summary. Identical
payloads and dependency versions produce byte-identical `store.data` and
`store.index` files.

## Reads and residency

The factory returns either a fully validated `TieredStore` or `StoreError`. It
opens `store.data` read-only and keeps one descriptor for explicit offset reads.
Metadata lookup returns immutable values.

`read_into` clears the caller-owned destination before work. A resident read
copies the owned fast buffer and reports `fast`. A nonresident read uses `pread`,
detects short reads, verifies CRC32, reports `slow`, and never promotes. Any
failure leaves the destination empty, changes no residency, increments no
successful-read counter, and records one stable failure code.

The fast tier is a sorted map of record ID to engine-owned
`std::vector<std::byte>`. Complete records only are installed; duplicate or
partial residency is impossible. Snapshots return ascending IDs and internally
consistent bytes/capacity without exposing buffers.

## Promotion, eviction, and exact targets

Individual promotion is idempotent for a resident record, rejects an oversized
record or insufficient remaining capacity, never selects a victim, and stages a
complete checksum-verified slow read before installation. Individual eviction is
idempotent for a known absent record, removes only the memory copy, and never
writes the authoritative file.

`apply_target_set` makes the supplied IDs the complete exact residency set. It:

1. sorts without deduplicating and rejects duplicates;
2. validates IDs, individual sizes, checked totals, and capacity;
3. computes ascending incoming and outgoing IDs;
4. reads and verifies every incoming payload into temporary buffers;
5. constructs the complete next state before touching current residency; and
6. commits with a no-throw container swap, then updates movement counters.

Unknown IDs, invalid targets, allocation failure, short reads, I/O failure, or
checksum mismatch before the swap leave prior residency and committed movement
counters exactly unchanged. Successfully staged bytes discarded by an aborted
transition are counted separately. The engine performs no implicit target
expansion, victim selection, or policy-specific behavior.

## Structured errors and logical statistics

Expected runtime failures use `tl::expected` and a `StoreError` containing a
stable code, concise message, and optional record ID, byte offset, and path.
Codes cover unknown/duplicate records, invalid targets, capacity and oversized
records, corrupt/versioned indexes, data mismatch, truncation, checksum and I/O
failure, allocation and arithmetic failure, invalid configuration, occupied
destinations, and malformed manifest/trace input.

`StoreStats` contains only deterministic logical values:

- successful fast/slow client reads and bytes;
- successful promotion-source reads and bytes, including staging later aborted;
- committed promotion/eviction counts and bytes;
- target calls, successful/failed calls, and aborted staged bytes;
- current resident records/bytes and byte high-water mark; and
- a count for every stable error code.

Idempotent promotion/eviction does not count movement. Failed work does not count
as a successful read or committed movement.

## Inspection and replay

Inspection reports format/version, record count, data bytes, min/median/max
record size, structural validity, optional capacity feasibility, and optional
full checksum results:

```bash
build/cpp-debug/prism_store_inspect \
  --store-dir /tmp/prism_store \
  --capacity-bytes 40 \
  --verify-all
```

Replay consumes strict JSON Lines with increasing unique sequence numbers and
operations `read`, `promote`, `evict`, `apply_target_set`, and `snapshot`.
Unknown fields and unsupported operations are rejected. An optional `expected`
object may specify `success`, `error_code`, `read_tier`, `bytes`, and
`resident_record_ids`.

```bash
build/cpp-debug/prism_store_replay \
  --store-dir /tmp/prism_store \
  --capacity-bytes 40 \
  --trace cpp/tests/fixtures/replay.jsonl \
  --output /tmp/replay.json
```

Every operation report includes success/error, read tier/bytes/payload CRC where
applicable, committed movement deltas, post-operation residency, and cumulative
statistics. The final report includes residency, counters, the capacity
invariant, and expected-outcome mismatches. Identical store and trace inputs
produce byte-identical reports.

## Verification and limitations

Separate builds support `PRISM_ENABLE_ASAN` and `PRISM_ENABLE_UBSAN`:

```bash
cmake -S cpp -B build/cpp-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPRISM_BUILD_TESTS=ON \
  -DPRISM_WARNINGS_AS_ERRORS=ON \
  -DPRISM_ENABLE_ASAN=ON
cmake --build build/cpp-asan --parallel
ctest --test-dir build/cpp-asan --output-on-failure

cmake -S cpp -B build/cpp-ubsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPRISM_BUILD_TESTS=ON \
  -DPRISM_WARNINGS_AS_ERRORS=ON \
  -DPRISM_ENABLE_UBSAN=ON
cmake --build build/cpp-ubsan --parallel
ctest --test-dir build/cpp-ubsan --output-on-failure
```

The native suite covers deterministic builds, format corruption, exact reads,
checksum and truncation handling, capacity and residency invariants, idempotence,
staged failure atomicity, exact counters, strict replay, and repeated artifact
bytes.

No wall-clock timing is collected or placed in canonical artifacts. Ordinary
file reads are influenced by the operating-system page cache, and this milestone
makes no RAM/SSD latency or performance-superiority claim.

Milestone 6 itself has no Python binding, predictor invocation, or policy. The
subsequent synchronous binding and parity layer is documented in
`docs/python_cpp_integration.md`; it does not change this engine's semantics.
The engine still has no mutable record, dirty state, writeback, concurrency,
asynchronous I/O, memory mapping, direct I/O, compression, encryption, network
storage, database, GPU behavior, LLM integration, or crash-recovery mechanism.
