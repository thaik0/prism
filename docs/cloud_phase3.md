# Prism Cloud Phase 3 Operations

## Architecture and ownership

Terraform is the source of truth for the accepted single-account,
single-Region Cloud Phase 2 execution architecture. The main stack owns the
`prism-cloud` ECR repository, experiment bucket, `/aws/batch/job` log group,
Fargate compute environment, queue, ARM64 job definition, execution/job roles,
dedicated no-ingress security group, GitHub OIDC provider, deployment role, and
Terraform role.

The account's default VPC, six public subnets, and AWS Batch service-linked role
are referenced and remain account-owned. Image publication is intentionally not
infrastructure ownership: it pushes a Git-SHA tag and records a digest. Only a
reviewed Terraform apply promotes that digest into a job-definition revision.

## Bootstrap and remote state

Terraform state records the binding between configuration and real AWS resource
IDs. Remote state makes that binding available to GitHub Actions; versioning
provides recovery history; native S3 lockfiles prevent two writers from
changing it concurrently. DynamoDB is not used.

The narrow bootstrap stack has local state and creates only the state bucket:

```bash
terraform -chdir=infra/terraform/bootstrap init
terraform -chdir=infra/terraform/bootstrap plan \
  -var=aws_profile=prism-admin -out=/tmp/prism-bootstrap.tfplan
terraform -chdir=infra/terraform/bootstrap apply /tmp/prism-bootstrap.tfplan
```

Verify private access, encryption, and versioning directly:

```bash
aws --profile prism-admin s3api get-public-access-block \
  --bucket prism-terraform-state-164998084998-us-east-1
aws --profile prism-admin s3api get-bucket-encryption \
  --bucket prism-terraform-state-164998084998-us-east-1
aws --profile prism-admin s3api get-bucket-versioning \
  --bucket prism-terraform-state-164998084998-us-east-1
```

Neither bootstrap nor main state is committed. If bootstrap state is lost, do
not recreate the bucket: initialize the bootstrap stack and import the existing
bucket and its four configuration resources.

## Phase 2 migration

The live pre-migration inventory found that all Phase 2 resources except the
empty log group had already been deleted. The exact migration therefore was:

```bash
terraform -chdir=infra/terraform/main init \
  -backend-config=profile=prism-admin
terraform -chdir=infra/terraform/main import \
  -var=aws_profile=prism-admin \
  aws_cloudwatch_log_group.batch /aws/batch/job
terraform -chdir=infra/terraform/main plan \
  -var=aws_profile=prism-admin -out=/tmp/prism-foundation.tfplan
terraform -chdir=infra/terraform/main apply /tmp/prism-foundation.tfplan
terraform -chdir=infra/terraform/main plan \
  -detailed-exitcode -var=aws_profile=prism-admin
```

The first apply deliberately omitted `image_uri`; all foundation resources
were created without inventing an unapproved image. The imported log group kept
its accepted never-expire retention and received only Terraform ownership tags.

## OIDC trust and permissions

Trust and permission policies solve different problems:

- each role's trust policy accepts only GitHub's provider, audience
  `sts.amazonaws.com`, repository `thaik0/prism`, and its exact environment
  subject;
- each role's permissions policy limits what an accepted workflow can do.

`prism-cloud-deploy` can publish to the Prism ECR repository and run/inspect the
accepted smoke path. `prism-cloud-apply` can manage the named Prism resource set
and the remote-state prefix. Runtime execution and job roles remain separate.
No role uses `AdministratorAccess`, and GitHub stores no AWS access key.

Both GitHub environments allow only `main` and `cloud/milestone3`. The current
GitHub plan does not support required reviewers for this private repository, so
Terraform apply also requires manual dispatch, the `apply` operation, and a
typed `APPLY` confirmation. Enable required reviewers if the repository plan
later supports them.

## Normal delivery workflow

1. Pull requests run `.github/workflows/cloud-ci.yml` with no AWS credentials.
   It runs Python and native regression tests, the Linux/ARM64 container build,
   and Terraform format/init/validate.
2. Manually dispatch `Publish Prism ARM64 image`. Tests run before OIDC. The
   workflow pushes `sha-<git-sha>` to ECR and records the resulting digest and
   full promotion URI in the job summary. The ECR repository rejects tag
   mutation.
3. Manually dispatch `Prism Cloud Terraform` with operation `plan` and the full
   recorded `<repository>@sha256:<digest>`. Review the uploaded textual plan.
4. Dispatch it again with operation `apply`, the same digest, and confirmation
   `APPLY`. The saved plan is applied and an immediate no-drift plan must pass.
5. Manually dispatch `Prism Cloud smoke` with the promoted URI and embedded Git
   SHA. It verifies the active job definition first, then uses `prism-cloud` to
   submit, wait, validate the final completion manifest and every S3 hash,
   download artifacts, collect logs, and compare an exact-image local run with
   `prism-container-verify --mode cross-host`.

Publication never calls `RegisterJobDefinition`; smoke never changes
infrastructure; pull requests never receive an AWS token. `latest` is not an
accepted provenance reference.

## Local plan, promotion, and recovery

For a local reviewed promotion using SSO:

```bash
export TF_VAR_image_uri="164998084998.dkr.ecr.us-east-1.amazonaws.com/prism-cloud@sha256:REPLACE"
terraform -chdir=infra/terraform/main plan \
  -var=aws_profile=prism-admin -out=/tmp/prism-promotion.tfplan
terraform -chdir=infra/terraform/main apply /tmp/prism-promotion.tfplan
terraform -chdir=infra/terraform/main plan \
  -detailed-exitcode -var=aws_profile=prism-admin
```

If apply is interrupted, inspect the GitHub log and current lock before doing
anything. Never bypass an active lock. Once the writer is confirmed dead, use
`terraform force-unlock LOCK_ID`, review a fresh plan, and continue. A failed
Batch smoke does not require Terraform recovery: inspect `prism-cloud status`
and `prism-cloud logs`, preserve the failed attempt, fix the image or inputs,
publish a new immutable digest, and promote it through a new reviewed plan.

S3 bucket deletion is intentionally blocked while evidence or versions remain,
and ECR deletion is blocked while images remain. These are safety properties,
not teardown bugs.

## Costs and teardown

Idle costs are S3 state/experiment bytes and requests, ECR image storage, and
CloudWatch log storage; the Fargate compute environment and Batch queue do not
run or bill compute while idle. A smoke run adds Fargate vCPU/memory time plus
small ECR, S3, and CloudWatch request/transfer/ingestion charges. Record the job
duration and current `us-east-1` Fargate ARM rates in the verification report;
pricing is an estimate, not a billing guarantee.

Before teardown, preserve required evidence and the promoted image URI. Then:

1. empty all experiment-bucket object versions and ECR images deliberately;
2. run `terraform destroy` for the main stack with the current
   `TF_VAR_image_uri` supplied;
3. confirm Batch's asynchronous queue/environment deletion completes;
4. retain the state bucket for recovery, or empty all state versions and delete
   it only through the bootstrap stack as a separate final decision.

Destroying the main stack removes OIDC access. The local SSO identity is
therefore the documented break-glass path for a complete teardown.
