resource "aws_ecr_repository" "prism" {
  name                 = "prism-cloud"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "prism" {
  repository = aws_ecr_repository.prism.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Remove untagged build remnants after seven days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 7
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_s3_bucket" "experiments" {
  bucket        = var.experiment_bucket_name
  force_destroy = false
}

resource "aws_s3_bucket_ownership_controls" "experiments" {
  bucket = aws_s3_bucket.experiments.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "experiments" {
  bucket = aws_s3_bucket.experiments.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "experiments" {
  bucket = aws_s3_bucket.experiments.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "experiments" {
  bucket = aws_s3_bucket.experiments.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "experiments" {
  bucket = aws_s3_bucket.experiments.id

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [aws_s3_bucket_versioning.experiments]
}

resource "aws_cloudwatch_log_group" "batch" {
  name = "/aws/batch/job"
}
