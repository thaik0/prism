output "state_bucket_name" {
  description = "S3 bucket configured by the main stack backend."
  value       = aws_s3_bucket.terraform_state.id
}
