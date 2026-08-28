variable "aws_region" {
  description = "AWS Region for the single Prism Cloud environment."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Optional local AWS profile; CI uses OIDC environment credentials."
  type        = string
  default     = null
  nullable    = true
}

variable "vpc_id" {
  description = "Existing VPC used by the accepted Fargate Batch environment."
  type        = string
  default     = "vpc-0542615b3b30a8c71"
}

variable "subnet_ids" {
  description = "Existing public subnets used by Fargate; Prism does not own them."
  type        = list(string)
  default = [
    "subnet-0192440095fbc3698",
    "subnet-030604af6c20278ff",
    "subnet-048c1aaf9d02a1ac9",
    "subnet-0094870c7422782c5",
    "subnet-0f1d1cdedd5716857",
    "subnet-053f93be5adb0d543",
  ]

  validation {
    condition     = length(var.subnet_ids) >= 2 && length(distinct(var.subnet_ids)) == length(var.subnet_ids)
    error_message = "At least two distinct public subnet IDs are required."
  }
}

variable "experiment_bucket_name" {
  description = "S3 bucket for the prism-cloud-v1 input/output contract."
  type        = string
  default     = "prism-cloud-164998084998-us-east-1"
}

variable "state_bucket_name" {
  description = "Bootstrap-created remote-state bucket."
  type        = string
  default     = "prism-terraform-state-164998084998-us-east-1"
}

variable "image_uri" {
  description = "Approved immutable Prism ECR image URI. Null creates the foundation without a job definition."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition = var.image_uri == null || can(regex(
      "^[0-9]{12}\\.dkr\\.ecr\\.[a-z0-9-]+\\.amazonaws\\.com/[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$",
      var.image_uri,
    ))
    error_message = "image_uri must be null or a private ECR URI pinned by sha256 digest."
  }
}

variable "github_oidc_repository_subject" {
  description = "GitHub-customized OIDC owner/repository subject, including immutable IDs."
  type        = string
  default     = "thaik0@206290841/prism@1322114194"
}

variable "github_deploy_environment" {
  description = "Exact GitHub environment allowed to assume the deployment role."
  type        = string
  default     = "prism-cloud-deploy"
}

variable "github_apply_environment" {
  description = "Exact GitHub environment allowed to assume the Terraform role."
  type        = string
  default     = "prism-cloud-apply"
}
