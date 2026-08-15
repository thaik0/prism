variable "aws_region" {
  description = "AWS Region for the single Prism Cloud environment."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Optional local AWS profile used only for bootstrap."
  type        = string
  default     = null
  nullable    = true
}

variable "state_bucket_name" {
  description = "Globally unique S3 bucket used only for Prism Terraform state."
  type        = string
  default     = "prism-terraform-state-164998084998-us-east-1"
}
