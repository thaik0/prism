terraform {
  backend "s3" {
    bucket       = "prism-terraform-state-164998084998-us-east-1"
    key          = "prism/cloud-phase3/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}
