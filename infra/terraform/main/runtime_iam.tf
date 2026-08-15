data "aws_iam_policy_document" "ecs_task_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "prism-cloud-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_trust.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "job" {
  name               = "prism-cloud-job"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_trust.json
}

data "aws_iam_policy_document" "job" {
  statement {
    sid       = "ReadPrismCloudInputs"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.experiments.arn}/prism-cloud/v1/runs/*/input/*"]
  }

  statement {
    sid       = "ReadWritePrismCloudAttempts"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.experiments.arn}/prism-cloud/v1/runs/*/attempts/*"]
  }
}

resource "aws_iam_role_policy" "job" {
  name   = "prism-cloud-prefix-access"
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.job.json
}
