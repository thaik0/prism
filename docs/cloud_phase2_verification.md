# Prism Cloud Phase 2 Verification Record

## Status

**Status:** Valid real AWS execution operationally verified; cross-host parity
contract corrected and locally verified.

The valid Batch run succeeded and its cloud artifacts were independently
validated. The original whole-tree byte comparison against a fresh local run
exposed a last-bit numerical portability boundary. This record distinguishes
that boundary from AWS operational correctness and does not represent local
tests as cloud evidence.

## Repository and local verification

- Branch: `cloud/milestone2`
- Phase 1 base: commit `b9e38d8`
- Accepted experiment: `baseline__seed_1729`
- Contract: `prism-cloud-v1`
- Runtime target: Fargate/Linux/ARM64
- Selected resources: 1 vCPU, 2 GiB, one attempt
- Static credentials in source/image: forbidden

The focused suite covers deterministic inputs, manifest hashes, safe keys and
paths, immutable ECR resolution, locked Batch requests, Fargate/ARM64
parameters, separate roles, completion validation, input failure,
completion-last publication, incomplete rejection, remote artifact hashes,
safe download, and identical repeated download.

The full local Python suite passed with 297 tests and the one
existing opt-in LLMServingSim Docker test skipped. Phase 1 runner and semantic
verifier tests remain unchanged and green.

The Linux/ARM64 packaging build ran with Docker 29.6.1. Its build stage passed:

- Debug native CTest: 29/29;
- AddressSanitizer native CTest: 29/29;
- UndefinedBehaviorSanitizer native CTest: 29/29;
- native parity smoke: four policies, zero mismatches;
- installed-wheel Python suite at that checkpoint: 296 passed, one skipped;
- `pip check`: no broken requirements.

The initial local runtime image was Linux/aarch64, ran as `10001:10001`, and was
212,044,152 bytes by Docker config size. It imported boto3 1.43.72 and exposed
both `prism-cloud` and `prism-cloud-bootstrap`. A deliberate bootstrap invocation
without its allowlisted environment exited 2. Runtime inspection found no
source tree, `.git`, or conventional static AWS credential directory.

Two representative runs in this image each produced 29 Phase 1 artifacts plus
`run_manifest.json`. Repeat verification passed all 30 files byte-for-byte.
Representative accepted evidence remained unchanged:

- experiment `baseline__seed_1729` completed;
- Predictive Greedy combined cost `57764.0` and hit rate
  `0.5407038748666904`;
- 130,349 Python/native operations, zero mismatches, and overall parity passed.

No repository credential pattern was found outside the pinned third-party tree.
The image contains boto3 but no access key, secret, session token, credential
file, or account identifier.

## Real AWS valid-run evidence

Run `pcv1-739ccbafb9f94363a2647012` supplied the following real acceptance
evidence:

- AWS Batch reached `SUCCEEDED`;
- the runtime reported `aarch64` and CloudWatch logs were available;
- the workload published 29 hashed Phase 1 artifacts plus `run_manifest.json`;
- Prism result validation passed; and
- `prism-cloud download` verified every downloaded S3 artifact.

The inputs and workload artifacts matched a fresh local Linux/ARM64 run of the
same Phase 2 image. `--mode repeat` correctly rejected whole-tree byte identity.
The first differences were `structure/learned_structure.npz` and
`structure/recovery_report.json`: for example, activation cosine similarity was
`0.9853218619186884` locally versus `0.9853218619186886` on AWS, and
reconstruction error was `217.38123663101337` versus
`217.38123663101345`. Differences at roughly `1e-16` to `1e-13` propagated into
predictor and simulation floating artifacts and their hashes. No scientific
gate or discrete result was reported to change.

## Determinism-boundary diagnosis

Two independent local runs of the exact image ID `fa19e2cd...` were identical
across all 30 files. The image contained NumPy 2.3.2, SciPy 1.16.1,
scikit-learn 1.9.0, and OpenBLAS 0.3.30. On the local host OpenBLAS selected the
`neoversen1` kernel and 12 threads.

Changing only the BLAS thread count reproduced last-bit divergence beginning in
NMF and propagating downstream. Forced generic ARM kernels produced additional
last-bit variants; forcing a Neoverse V1 kernel was unsupported by the local
CPU. No tested thread count reproduced the exact AWS pair. This isolates the
cause to host-selected floating-point kernel/reduction execution rather than
input mismatch, global random state, or cloud orchestration. Thread pinning
alone is therefore not a demonstrated cross-host byte-identity fix.

The corrected `cross-host` contract preserves same-host `repeat` byte identity,
requires identical declared image runtime and scientific identity, and verifies
all 29 manifest artifacts. In the controlled thread-topology comparison it
reported 13 byte-identical artifacts, 6 numerical NPZ artifacts, 10 semantic
JSON artifacts, and maximum absolute drift `5.665276581190426e-11`, passing the
existing `1e-9` Phase 1 tolerance. The new mode still needs to be run against
the retained local/AWS roots; it does not require another Batch submission.

Invalid-input execution, cost evidence, and teardown remain separate Phase 2
acceptance items unless already recorded outside this repository. No AWS
resource was modified or recreated during this diagnosis.
