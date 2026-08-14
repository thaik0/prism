# Prism Cloud Phase 2 Execution Plan

## Status and boundary

**Status:** Local implementation complete; real AWS acceptance blocked by
unavailable credentials

This milestone adds one narrow AWS Batch adapter around the accepted Cloud
Phase 1 Linux/ARM64 runner. It does not change the selected experiment, its
scientific pipeline, native semantics, deterministic artifacts, or parity
rules. Infrastructure automation and deployment CI remain outside this phase.

## Repository contracts inspected

- `AGENTS.md`, `README.md`, the technical plan, decisions, reproducibility
  guide, and Milestone 5 results;
- the Cloud Phase 1 execution plan, verification record, container contract,
  `Dockerfile`, accepted experiment spec, Python package entry points, runner,
  verifier, and focused tests;
- branch `cloud/milestone2`, the `origin` remote, recent Phase 1 history, and the
  pre-existing untracked state inside the `third_party/LLMServingSim` submodule.

The frozen entry point is `prism-container-run` with
`container/phase1-experiment.json`, selecting `baseline__seed_1729`. A valid
Phase 1 root contains `experiment/`, `native/`, and a final
`run_manifest.json`; the manifest hashes all 29 scientific artifacts. Repeat
Linux verification is byte-exact, while accepted native/Linux verification
uses the existing exact/discrete/numerical semantic classifications.

## Implementation plan

1. Define the strict `prism-cloud-v1` input and completion contracts, safe S3
   key/path handling, deterministic bundle hashing, and stable run identity.
2. Add a boto3-backed local `prism-cloud` CLI for submission, direct Batch
   status, recent CloudWatch logs, and verified canonical-artifact download.
3. Add a fixed cloud bootstrap entry point that downloads and validates the
   minimal five-file Phase 1 input, invokes the existing runner, validates its
   output, uploads artifacts, and publishes the completion manifest last.
4. Add focused fake/stub-based tests and preserve all existing Phase 1 tests.
5. Document manual least-privilege AWS setup, digest-pinned ECR publication,
   representative operation, failure checks, cost estimation, and teardown.
6. Run one valid and one invalid real AWS Batch job when the normal local AWS
   credential provider chain and account resources permit it; compare the valid
   download with the accepted local Linux output using the existing verifier.

## Assumptions and risks

- One AWS Region, S3 bucket, Batch queue, ARM64/Fargate job definition, and ECR
  repository are explicitly configured by the operator.
- The bounded representative job starts at 1 vCPU and 2 GiB, a supported
  Fargate pair; actual execution is the acceptance check for sufficiency.
- Public subnets with assigned public IPs provide outbound ECR/S3/CloudWatch
  access, and the selected security group has no inbound rules.
- The local CLI and running task use the standard boto3 credential provider
  chain. No credential material is accepted as Prism configuration.
- Real execution depends on account credentials, service permissions, quotas,
  ARM64 image publication, and AWS network availability. Any unavailable
  prerequisite will be reported rather than simulated as acceptance evidence.

## Explicit non-goals

- Terraform, CDK, CloudFormation, deployment CI, or automatic ECR publishing;
- retries, resume, schedulers, databases, queues beyond AWS Batch, services, or
  multi-user behavior;
- new experiments, model/predictor/controller changes, LLMServingSim cloud
  execution, or changes to deterministic scientific provenance;
- multi-region operation or production observability.

## Completion checkpoints

- [x] Versioned contracts, deterministic bundle, AWS adapter, and local CLI.
- [x] Bootstrap, completion validation, safe download, and failure tests.
- [x] Manual AWS documentation and local regression verification.
- [ ] Valid and invalid real Batch acceptance evidence, or an explicit external
  prerequisite blocker. The blocker is recorded: the standard boto3 chain has
  no credentials or Region in the execution environment.
