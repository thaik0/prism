# AWS Batch Cloud Execution

## Scope and contract

Cloud Phase 2 submits exactly one accepted Phase 1 representative experiment
to AWS Batch on Fargate/Linux/ARM64. S3 carries a deterministic five-file input
bundle and canonical output artifacts; CloudWatch receives stdout/stderr. The
container still calls the unchanged Phase 1 runner and verifier. It does not
accept a repository upload, arbitrary command, output path, algorithm, policy,
or experiment ID.

The contract is `prism-cloud-v1`:

```text
prism-cloud/v1/runs/<stable-run-id>/
  input/
    configs/milestone2_representative.json
    configs/milestone3_predictor.json
    configs/milestone3_predictor_workload.json
    configs/milestone5_experiments.json
    container/phase1-experiment.json
    input_manifest.json
  submission.json
  attempts/<batch-job-id>/
    artifacts/<complete Phase 1 output tree>
    failure_manifest.json       # optional diagnostic; never a valid result
    completion_manifest.json    # successful commit marker, uploaded last
```

The stable run ID hashes the input-manifest hash, immutable image URI, Region,
bucket, queue, and exact job-definition revision. A repeated submission may
therefore have the same Prism run ID but always receives a distinct Batch job ID
and attempt prefix. `submission.json` maps the run ID to the most recently
submitted attempt; AWS Batch and S3 remain authoritative.

A result is valid only when Batch is `SUCCEEDED`, the completion manifest is
strictly valid, and every listed S3 artifact has the recorded size and SHA-256.
The CLI downloads no other objects. Batch IDs and AWS metadata remain in the
completion manifest and never enter deterministic Phase 1 scientific files.

## Local prerequisites and authentication

Install Prism and the declared boto3 dependency, Docker with ARM64 build
support, and AWS CLI v2. Authentication uses only the normal AWS CLI/boto3
credential provider chain; do not place access keys in this repository, the
image, Batch environment variables, or command history.

```bash
python3 -m pip install -e .
aws sts get-caller-identity
aws configure get region
```

The manual flow uses shell variables only to keep account-specific values out
of committed files. Replace example names as needed:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export PRISM_BUCKET="prism-cloud-${AWS_ACCOUNT_ID}-${AWS_REGION}"
export PRISM_ECR_REPOSITORY=prism-cloud
export PRISM_COMPUTE_ENV=prism-cloud-fargate
export PRISM_JOB_QUEUE=prism-cloud
export PRISM_JOB_DEFINITION=prism-cloud-v1-arm64
```

## ECR: manual build, login, tag, push, and digest

Create one private repository. No `latest` tag is used or accepted by the
submission CLI.

```bash
aws ecr create-repository --region "$AWS_REGION" \
  --repository-name "$PRISM_ECR_REPOSITORY" \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256

export PRISM_GIT_REVISION="$(git rev-parse HEAD)"
export PRISM_IMAGE_TAG="phase2-${PRISM_GIT_REVISION}"
export PRISM_ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PRISM_ECR_REPOSITORY}"

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker buildx build --platform linux/arm64 --load --no-cache \
  --build-arg PRISM_GIT_REVISION="$PRISM_GIT_REVISION" \
  -t "prism:phase2-${PRISM_GIT_REVISION}" .
docker tag "prism:phase2-${PRISM_GIT_REVISION}" \
  "${PRISM_ECR_URI}:${PRISM_IMAGE_TAG}"
docker push "${PRISM_ECR_URI}:${PRISM_IMAGE_TAG}"

export PRISM_IMAGE_DIGEST="$(aws ecr describe-images --region "$AWS_REGION" \
  --repository-name "$PRISM_ECR_REPOSITORY" \
  --image-ids imageTag="$PRISM_IMAGE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export PRISM_IMAGE_URI="${PRISM_ECR_URI}@${PRISM_IMAGE_DIGEST}"
```

The image is the accepted Phase 1 runtime plus the small cloud package and
boto3 required for S3 access. The Phase 1 command, spec, scientific pipeline,
and canonical run-manifest content are unchanged.

## S3 and separate IAM roles

Create the bucket with public access blocked and default encryption. For
Regions other than `us-east-1`, add
`--create-bucket-configuration LocationConstraint="$AWS_REGION"` to the create
command.

```bash
aws s3api create-bucket --region "$AWS_REGION" --bucket "$PRISM_BUCKET"
aws s3api put-public-access-block --region "$AWS_REGION" --bucket "$PRISM_BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --region "$AWS_REGION" --bucket "$PRISM_BUCKET" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

Create two roles using the ECS task trust policy. The execution role gets only
the standard ECS task execution policy for private ECR pulls and awslogs. The
Prism job role gets only direct object access under the configured Prism Cloud
input and attempt prefixes; it has no account-wide S3 access.

```bash
aws iam create-role --role-name prism-cloud-execution \
  --assume-role-policy-document file://cloud/aws/ecs-task-trust.json
aws iam attach-role-policy --role-name prism-cloud-execution \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

aws iam create-role --role-name prism-cloud-job \
  --assume-role-policy-document file://cloud/aws/ecs-task-trust.json
sed "s/BUCKET_NAME/${PRISM_BUCKET}/g" cloud/aws/prism-job-policy.json \
  > /tmp/prism-cloud-job-policy.json
aws iam put-role-policy --role-name prism-cloud-job \
  --policy-name prism-cloud-prefix-access \
  --policy-document file:///tmp/prism-cloud-job-policy.json
```

Local submission credentials separately need ECR `DescribeImages`, Batch
`DescribeJobDefinitions`, `SubmitJob`, and `DescribeJobs`, CloudWatch Logs
`GetLogEvents`, plus object read/write access to this bucket's
`prism-cloud/v1/` prefix. `iam:PassRole` is needed only while manually
registering the job definition.

## Fargate network, compute environment, queue, and job definition

Use existing public subnets. Create a dedicated security group with no inbound
rules; its default outbound rule permits ECR, S3, and CloudWatch access. Do not
create a NAT gateway solely for this bounded experiment.

```bash
export PRISM_VPC_ID="vpc-REPLACE"
export PRISM_SUBNETS='["subnet-REPLACE_A","subnet-REPLACE_B"]'
export PRISM_SECURITY_GROUP_ID="$(aws ec2 create-security-group \
  --region "$AWS_REGION" --vpc-id "$PRISM_VPC_ID" \
  --group-name prism-cloud-batch --description 'Prism Batch outbound only' \
  --query GroupId --output text)"

aws batch create-compute-environment --region "$AWS_REGION" \
  --compute-environment-name "$PRISM_COMPUTE_ENV" \
  --type MANAGED --state ENABLED \
  --compute-resources "type=FARGATE,maxvCpus=4,subnets=${PRISM_SUBNETS},securityGroupIds=[${PRISM_SECURITY_GROUP_ID}]"

aws batch create-job-queue --region "$AWS_REGION" \
  --job-queue-name "$PRISM_JOB_QUEUE" --state ENABLED --priority 1 \
  --compute-environment-order order=1,computeEnvironment="$PRISM_COMPUTE_ENV"
```

Copy [the job-definition template](../cloud/aws/job-definition.json) to `/tmp`
and replace `ACCOUNT_ID`, `REGION`, and `IMAGE_DIGEST`, then register it:

```bash
sed -e "s/ACCOUNT_ID/${AWS_ACCOUNT_ID}/g" \
    -e "s/REGION/${AWS_REGION}/g" \
    -e "s/IMAGE_DIGEST/${PRISM_IMAGE_DIGEST#sha256:}/g" \
    cloud/aws/job-definition.json > /tmp/prism-cloud-job-definition.json
aws batch register-job-definition --region "$AWS_REGION" \
  --cli-input-json file:///tmp/prism-cloud-job-definition.json
```

The selected `1 vCPU / 2 GiB` pair is supported by Fargate and is intended for
the bounded single-seed representative workload, not the 36-run sweep. Actual
peak-memory/runtime evidence from acceptance determines whether it is
sufficient. Increasing it requires a new job-definition revision and therefore
a new stable run ID.

The template follows the official AWS Batch
[RegisterJobDefinition API](https://docs.aws.amazon.com/batch/latest/APIReference/API_RegisterJobDefinition.html),
[Fargate job-definition guidance](https://docs.aws.amazon.com/batch/latest/userguide/create-job-definition-Fargate.html),
and [supported Fargate CPU/memory pairs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-tasks-services.html).

## Submit, inspect, download, and verify

The job definition, rather than user input, fixes the image and
`prism-cloud-bootstrap` command. Submission verifies that its active revision
matches the resolved digest and locked Fargate/ARM64/role/log contract.

```bash
export PRISM_CLOUD_BUCKET="$PRISM_BUCKET"
export PRISM_CLOUD_IMAGE="${PRISM_ECR_URI}:${PRISM_IMAGE_TAG}"
export PRISM_CLOUD_JOB_QUEUE="$PRISM_JOB_QUEUE"
export PRISM_CLOUD_JOB_DEFINITION="$PRISM_JOB_DEFINITION"

prism-cloud submit --spec container/phase1-experiment.json
prism-cloud status RUN_ID
prism-cloud logs RUN_ID
prism-cloud download RUN_ID --output-dir /tmp/prism-cloud-output
```

`status` prints native Batch state separately from Prism validity. A successful
Batch state with a missing completion manifest is still incomplete. Download
refuses failed/incomplete results, malformed manifests, missing or corrupt
objects, unsafe paths, unexpected destination files, and non-identical
overwrite. A repeated download reuses only byte-identical files.

Compare the verified download to an accepted local Linux output:

```bash
prism-container-verify --left /tmp/prism-phase1-local \
  --right /tmp/prism-cloud-output --mode repeat
```

Both use Linux/ARM64 and the same image digest, so acceptance requires complete
byte identity. Use `cross-platform` only for the separately documented native
macOS/Linux comparison.

## Deliberate missing-input failure

Submit the fixed definition with a valid-format run ID whose S3 input does not
exist. This tests a real nonzero path without an arbitrary command override:

```bash
export INVALID_RUN_ID=pcv1-000000000000000000000000
aws batch submit-job --region "$AWS_REGION" \
  --job-name prism-invalid-input --job-queue "$PRISM_JOB_QUEUE" \
  --job-definition "$PRISM_JOB_DEFINITION" --retry-strategy attempts=1 \
  --container-overrides "environment=[
    {name=PRISM_CLOUD_BUCKET,value=${PRISM_BUCKET}},
    {name=PRISM_CLOUD_IMAGE_URI,value=${PRISM_IMAGE_URI}},
    {name=PRISM_CLOUD_REGION,value=${AWS_REGION}},
    {name=PRISM_CLOUD_RUN_ID,value=${INVALID_RUN_ID}}]"
```

Expected: Batch `FAILED`, nonzero exit, an input error in `/aws/batch/job`, and
no valid completion manifest below that attempt.

## Cost estimate

Use the Batch job's `startedAt`/`stoppedAt` duration and current 1-vCPU, 2-GiB
Fargate rates in the execution Region. Add stored S3 bytes, request count,
CloudWatch ingestion/storage, and ECR storage; report secondary items even when
they round below one cent. Record the pricing source and retrieval date in the
verification report. This is an estimate, not a billing guarantee.

## Teardown

Preserve sanitized evidence first. Then disable/delete the queue and compute
environment, delete the no-inbound security group, empty/delete the bucket,
remove ECR, and remove both roles. Batch deletions are asynchronous; wait for
each resource to become `DELETED` before deleting its dependency.

```bash
aws batch update-job-queue --region "$AWS_REGION" \
  --job-queue "$PRISM_JOB_QUEUE" --state DISABLED
aws batch delete-job-queue --region "$AWS_REGION" --job-queue "$PRISM_JOB_QUEUE"
aws batch update-compute-environment --region "$AWS_REGION" \
  --compute-environment "$PRISM_COMPUTE_ENV" --state DISABLED
aws batch delete-compute-environment --region "$AWS_REGION" \
  --compute-environment "$PRISM_COMPUTE_ENV"
aws batch deregister-job-definition --region "$AWS_REGION" \
  --job-definition "$PRISM_JOB_DEFINITION"
aws ec2 delete-security-group --region "$AWS_REGION" \
  --group-id "$PRISM_SECURITY_GROUP_ID"

aws s3 rm "s3://${PRISM_BUCKET}/prism-cloud/v1/" --recursive
aws s3api delete-bucket --region "$AWS_REGION" --bucket "$PRISM_BUCKET"
aws ecr delete-repository --region "$AWS_REGION" \
  --repository-name "$PRISM_ECR_REPOSITORY" --force

aws iam delete-role-policy --role-name prism-cloud-job \
  --policy-name prism-cloud-prefix-access
aws iam delete-role --role-name prism-cloud-job
aws iam detach-role-policy --role-name prism-cloud-execution \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam delete-role --role-name prism-cloud-execution
```

No infrastructure framework, deployment CI, automatic publishing,
retry/resume, database, service, or production observability is part of this
phase.
