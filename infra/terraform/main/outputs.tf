output "aws_account_id" {
  value = local.account_id
}

output "aws_region" {
  value = data.aws_region.current.region
}

output "batch_compute_environment_arn" {
  value = aws_batch_compute_environment.prism.arn
}

output "batch_job_definition_arn" {
  value = try(aws_batch_job_definition.prism[0].arn, null)
}

output "batch_job_queue_arn" {
  value = aws_batch_job_queue.prism.arn
}

output "deployment_role_arn" {
  value = aws_iam_role.github_deploy.arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.prism.repository_url
}

output "experiment_bucket_name" {
  value = aws_s3_bucket.experiments.id
}

output "promoted_image_uri" {
  value = var.image_uri
}

output "terraform_role_arn" {
  value = aws_iam_role.terraform.arn
}
