resource "aws_security_group" "batch" {
  name        = "prism-cloud-batch"
  description = "Prism Batch outbound only"
  vpc_id      = data.aws_vpc.selected.id

  egress {
    description      = "Fargate access to ECR, S3, and CloudWatch"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = []
  }
}

resource "aws_batch_compute_environment" "prism" {
  name  = "prism-cloud-fargate"
  type  = "MANAGED"
  state = "ENABLED"

  compute_resources {
    max_vcpus          = 4
    security_group_ids = [aws_security_group.batch.id]
    subnets            = var.subnet_ids
    type               = "FARGATE"
  }

  lifecycle {
    create_before_destroy = false
  }
}

resource "aws_batch_job_queue" "prism" {
  name     = "prism-cloud"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.prism.arn
  }
}

resource "aws_batch_job_definition" "prism" {
  count = var.image_uri == null ? 0 : 1

  name                  = "prism-cloud-v1-arm64"
  type                  = "container"
  platform_capabilities = ["FARGATE"]
  propagate_tags        = true

  container_properties = jsonencode({
    image            = var.image_uri
    command          = ["prism-cloud-bootstrap"]
    executionRoleArn = aws_iam_role.execution.arn
    jobRoleArn       = aws_iam_role.job.arn
    resourceRequirements = [
      { type = "VCPU", value = "1" },
      { type = "MEMORY", value = "2048" },
    ]
    networkConfiguration = { assignPublicIp = "ENABLED" }
    logConfiguration     = { logDriver = "awslogs" }
    runtimePlatform = {
      operatingSystemFamily = "LINUX"
      cpuArchitecture       = "ARM64"
    }
  })

  retry_strategy {
    attempts = 1
  }
}
