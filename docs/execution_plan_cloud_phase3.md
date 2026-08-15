# Prism Cloud Phase 3 Execution Plan

## Status and boundary

**Status:** In progress

This milestone automates the accepted Cloud Phase 2 AWS architecture with one
Terraform environment and GitHub Actions OIDC. It does not change the Phase 1
container, the `prism-cloud-v1` S3/Batch contract, the accepted representative
experiment, or any Prism scientific behavior.

## Inspected repository and live AWS state

The repository inspection covered `AGENTS.md`, `README.md`, the technical plan,
all Cloud Phase 1/2 plans and verification records, setup/teardown guidance,
the container build and runner, `src/prism/cloud`, Phase 2 IAM and Batch JSON,
tests, packaging, ignore rules, GitHub workflows, and current Git state.

The live `164998084998` account in `us-east-1` no longer contains the Phase 2
experiment bucket, ECR repository, Batch compute environment, queue, active job
definition, runtime roles, or dedicated security group. The retained
`/aws/batch/job` CloudWatch log group is empty. The existing default VPC has six
public subnets, and the account retains the AWS Batch service-linked role. No
GitHub Actions OIDC provider exists.

## Ownership and migration decisions

- Import `/aws/batch/job`; preserving the accepted shared Batch log-group name
  avoids a create collision and retains its existing never-expire setting.
- Deliberately recreate the deleted ECR repository, experiment bucket, Batch
  environment, queue, job definition, execution/job roles, and no-ingress
  security group. There is no retained resource or evidence store to preserve.
- Reference the existing default VPC, public subnets, and AWS Batch
  service-linked role. Prism does not own them.
- Create and own the account's GitHub OIDC provider because none exists. If the
  account later adopts a shared provider, it must be migrated deliberately
  rather than duplicated.

## Terraform and state approach

`infra/terraform/bootstrap` is a narrow, local-state configuration that creates
only the private, encrypted, versioned state bucket. `infra/terraform/main` uses
that bucket with native S3 lockfiles; DynamoDB is not used. Bootstrap and the
first main apply use the developer's existing SSO identity. State, lock files,
plans, credentials, and `.terraform/` are ignored.

The first main apply may omit `image_uri`, creating the foundation without a
Batch job definition. After publication records an immutable ECR digest, a
reviewed apply supplies `<repository>@sha256:<digest>` and Terraform registers
the promoted job-definition revision. Publication never registers or mutates a
job definition.

## OIDC trust and workflow layout

OIDC trust is limited to repository `thaik0/prism` using GitHub's customized
immutable owner/repository IDs, audience `sts.amazonaws.com`, and exact GitHub
deployment-environment subjects:

- `prism-cloud-deploy` for immutable ECR publication and manual smoke;
- `prism-cloud-apply` for Terraform plan/apply.

The deployment role is limited to the Prism ECR repository, accepted Batch
queue/job-definition family, the experiment bucket prefix, and Batch logs. The
separate Terraform role can manage the named Prism resource set and state
prefix, but is not granted `AdministratorAccess`.

Four workflows remain separate: credential-free PR CI, image publication,
manual Terraform plan/apply, and manual cloud smoke. Smoke verifies the active
Terraform-managed image digest before invoking the existing `prism-cloud`
submit/wait/download interface and the established cross-host parity verifier.

## Assumptions, risks, and explicit non-goals

- Region is `us-east-1`; the accepted default-VPC public subnets remain suitable
  for Fargate tasks with assigned public IPs.
- Linux/ARM64, 1 vCPU, 2 GiB, one attempt, separate runtime roles, `awslogs`,
  and completion-manifest-last behavior remain fixed.
- GitHub environment protection must restrict deployment to the accepted ref
  and protect apply with required review before real OIDC acceptance.
- Provider/API drift and IAM resource-level support are validated by real plan,
  apply, OIDC assumption, publication, and smoke evidence.
- No multi-environment module hierarchy, Kubernetes, service tier, database,
  new scientific work, automatic per-commit Batch run, or Phase 4 feature is in
  scope.

## Checkpoints

- [ ] Terraform foundation, bootstrap, import, plan, apply, and no-drift plan.
- [ ] OIDC provider, constrained roles, environment protection, and assumption
  proof.
- [ ] PR CI, image publish, Terraform plan/apply, and manual smoke workflows.
- [ ] Real publication, promotion, Batch smoke, parity/evidence, documentation,
  final tests, and clean pushed branch.
