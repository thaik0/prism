data "tls_certificate" "github" {
  url = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github.certificates[length(data.tls_certificate.github.certificates) - 1].sha1_fingerprint]
}

data "aws_iam_policy_document" "github_deploy_trust" {
  statement {
    sid     = "GitHubEnvironmentOnly"
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:${var.github_deploy_environment}"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name                 = "prism-cloud-github-deploy"
  assume_role_policy   = data.aws_iam_policy_document.github_deploy_trust.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "github_deploy" {
  statement {
    sid       = "EcrLogin"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "PublishPrismImage"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:ListImages",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.prism.arn]
  }

  statement {
    sid    = "SubmitAcceptedBatchJob"
    effect = "Allow"
    actions = [
      "batch:SubmitJob",
    ]
    resources = [
      aws_batch_job_queue.prism.arn,
      "arn:aws:batch:${var.aws_region}:${local.account_id}:job-definition/prism-cloud-v1-arm64:*",
    ]
  }

  statement {
    sid    = "InspectAcceptedBatchJob"
    effect = "Allow"
    actions = [
      "batch:DescribeJobDefinitions",
      "batch:DescribeJobs",
      "batch:DescribeJobQueues",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "ListPrismCloudPrefix"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.experiments.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["prism-cloud/v1/*"]
    }
  }

  statement {
    sid       = "ReadWritePrismCloudRuns"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.experiments.arn}/prism-cloud/v1/*"]
  }

  statement {
    sid    = "InspectPrismBatchLogs"
    effect = "Allow"
    actions = [
      "logs:DescribeLogStreams",
      "logs:GetLogEvents",
    ]
    resources = [
      aws_cloudwatch_log_group.batch.arn,
      "${aws_cloudwatch_log_group.batch.arn}:log-stream:*",
    ]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "prism-cloud-github-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy.json
}

data "aws_iam_policy_document" "github_terraform_trust" {
  statement {
    sid     = "GitHubEnvironmentOnly"
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:${var.github_apply_environment}"]
    }
  }
}

resource "aws_iam_role" "terraform" {
  name                 = "prism-cloud-github-terraform"
  assume_role_policy   = data.aws_iam_policy_document.github_terraform_trust.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "github_terraform" {
  statement {
    sid    = "RemoteStateBucket"
    effect = "Allow"
    actions = [
      "s3:GetBucketLocation",
      "s3:GetBucketVersioning",
      "s3:ListBucket",
    ]
    resources = ["arn:aws:s3:::${var.state_bucket_name}"]
  }

  statement {
    sid    = "RemoteStateObjectsAndLock"
    effect = "Allow"
    actions = [
      "s3:DeleteObject",
      "s3:GetObject",
      "s3:PutObject",
    ]
    resources = ["arn:aws:s3:::${var.state_bucket_name}/prism/cloud-phase3/*"]
  }

  statement {
    sid    = "ManageExperimentBucket"
    effect = "Allow"
    actions = [
      "s3:CreateBucket",
      "s3:DeleteBucket",
      "s3:DeleteBucketPolicy",
      "s3:GetBucketLifecycleConfiguration",
      "s3:GetBucketLocation",
      "s3:GetBucketOwnershipControls",
      "s3:GetBucketPolicy",
      "s3:GetBucketPublicAccessBlock",
      "s3:GetBucketTagging",
      "s3:GetBucketVersioning",
      "s3:GetEncryptionConfiguration",
      "s3:ListBucket",
      "s3:PutBucketLifecycleConfiguration",
      "s3:PutBucketOwnershipControls",
      "s3:PutBucketPublicAccessBlock",
      "s3:PutBucketTagging",
      "s3:PutBucketVersioning",
      "s3:PutEncryptionConfiguration",
    ]
    resources = [aws_s3_bucket.experiments.arn]
  }

  statement {
    sid    = "DeleteExperimentObjectsDuringExplicitDestroy"
    effect = "Allow"
    actions = [
      "s3:DeleteObject",
      "s3:DeleteObjectVersion",
      "s3:GetObject",
      "s3:ListBucketVersions",
    ]
    resources = [
      aws_s3_bucket.experiments.arn,
      "${aws_s3_bucket.experiments.arn}/*",
    ]
  }

  statement {
    sid    = "ManagePrismEcr"
    effect = "Allow"
    actions = [
      "ecr:CreateRepository",
      "ecr:DeleteLifecyclePolicy",
      "ecr:DeleteRepository",
      "ecr:DescribeRepositories",
      "ecr:GetLifecyclePolicy",
      "ecr:ListTagsForResource",
      "ecr:PutImageScanningConfiguration",
      "ecr:PutImageTagMutability",
      "ecr:PutLifecyclePolicy",
      "ecr:TagResource",
      "ecr:UntagResource",
    ]
    resources = [aws_ecr_repository.prism.arn]
  }

  statement {
    sid    = "ManagePrismLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:DeleteLogGroup",
      "logs:DescribeLogGroups",
      "logs:ListTagsForResource",
      "logs:TagResource",
      "logs:UntagResource",
    ]
    resources = [aws_cloudwatch_log_group.batch.arn]
  }

  statement {
    sid    = "ManagePrismBatch"
    effect = "Allow"
    actions = [
      "batch:CreateComputeEnvironment",
      "batch:CreateJobQueue",
      "batch:DeleteComputeEnvironment",
      "batch:DeleteJobQueue",
      "batch:DeregisterJobDefinition",
      "batch:DescribeComputeEnvironments",
      "batch:DescribeJobDefinitions",
      "batch:DescribeJobQueues",
      "batch:ListTagsForResource",
      "batch:RegisterJobDefinition",
      "batch:TagResource",
      "batch:UntagResource",
      "batch:UpdateComputeEnvironment",
      "batch:UpdateJobQueue",
    ]
    resources = [
      aws_batch_compute_environment.prism.arn,
      aws_batch_job_queue.prism.arn,
      "arn:aws:batch:${var.aws_region}:${local.account_id}:job-definition/prism-cloud-v1-arm64:*",
    ]
  }

  statement {
    sid    = "ReadNetworkConfiguration"
    effect = "Allow"
    actions = [
      "ec2:DescribeSecurityGroups",
      "ec2:DescribeSubnets",
      "ec2:DescribeVpcs",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "ManagePrismSecurityGroup"
    effect = "Allow"
    actions = [
      "ec2:AuthorizeSecurityGroupEgress",
      "ec2:CreateSecurityGroup",
      "ec2:CreateTags",
      "ec2:DeleteSecurityGroup",
      "ec2:RevokeSecurityGroupEgress",
    ]
    resources = [
      "arn:aws:ec2:${var.aws_region}:${local.account_id}:security-group/*",
      "arn:aws:ec2:${var.aws_region}:${local.account_id}:vpc/${var.vpc_id}",
    ]
  }

  statement {
    sid    = "ManagePrismRoles"
    effect = "Allow"
    actions = [
      "iam:AttachRolePolicy",
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:DeleteRolePolicy",
      "iam:DetachRolePolicy",
      "iam:GetRole",
      "iam:GetRolePolicy",
      "iam:ListAttachedRolePolicies",
      "iam:ListRolePolicies",
      "iam:ListRoleTags",
      "iam:PutRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:UpdateAssumeRolePolicy",
    ]
    resources = [
      aws_iam_role.execution.arn,
      aws_iam_role.job.arn,
      aws_iam_role.github_deploy.arn,
      aws_iam_role.terraform.arn,
    ]
  }

  statement {
    sid       = "PassRuntimeRolesToEcsTasks"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.job.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }

  statement {
    sid    = "ManageGitHubOidcProvider"
    effect = "Allow"
    actions = [
      "iam:AddClientIDToOpenIDConnectProvider",
      "iam:CreateOpenIDConnectProvider",
      "iam:DeleteOpenIDConnectProvider",
      "iam:GetOpenIDConnectProvider",
      "iam:ListOpenIDConnectProviderTags",
      "iam:RemoveClientIDFromOpenIDConnectProvider",
      "iam:TagOpenIDConnectProvider",
      "iam:UntagOpenIDConnectProvider",
      "iam:UpdateOpenIDConnectProviderThumbprint",
    ]
    resources = ["arn:aws:iam::${local.account_id}:oidc-provider/token.actions.githubusercontent.com"]
  }
}

resource "aws_iam_role_policy" "terraform" {
  name   = "prism-cloud-terraform-management"
  role   = aws_iam_role.terraform.id
  policy = data.aws_iam_policy_document.github_terraform.json
}
