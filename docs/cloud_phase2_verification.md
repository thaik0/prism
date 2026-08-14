# Prism Cloud Phase 2 Verification Record

## Status

**Status:** Local implementation verified; real AWS acceptance blocked.

This record is intentionally incomplete until valid and invalid real AWS Batch
executions run. Local tests are not represented as cloud acceptance evidence.

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

## Real AWS acceptance evidence to record

- sanitized caller identity and Region command;
- ECR repository and immutable image digest;
- bucket/prefix, execution role, Prism job role, compute environment, queue,
  public subnet IDs (sanitized), no-inbound security group, and job-definition
  revision;
- valid Prism run ID and Batch job ID;
- Batch `SUCCEEDED`, Linux/ARM64 evidence, CloudWatch stream, final completion
  key, downloaded artifact count, and S3 hash validation;
- exact repeat verification against accepted local Linux output;
- invalid-input job ID, `FAILED`, nonzero exit, log evidence, and confirmation
  that no successful completion manifest exists;
- optional repeated valid job parity if cost remains immaterial;
- runtime-derived cost estimate with dated Region pricing source;
- teardown status and any intentionally retained resources.

## External blocker and unperformed acceptance

On August 14, 2026, a read-only boto3 session/STS probe reported no credentials
and no configured Region. AWS CLI was also absent before the temporary boto3
probe. Consequently:

- no ECR repository, S3 bucket, IAM role, security group, compute environment,
  queue, or job definition was created;
- no image was pushed and no immutable ECR digest exists;
- no valid or invalid Batch job ID, CloudWatch stream, S3 completion key,
  cloud download, cloud/local parity result, runtime-derived cost, or teardown
  evidence exists;
- teardown is not applicable because no AWS resources were created.

These are required Phase 2 acceptance items and remain incomplete. No mock ID or
local Docker result is substituted for real AWS evidence. Resuming acceptance
requires a Region and usable credentials through the standard boto3 provider
chain, then following the exact commands in `docs/cloud_phase2.md`.
