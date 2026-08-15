data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_vpc" "selected" {
  id = var.vpc_id
}

data "aws_subnet" "selected" {
  for_each = toset(var.subnet_ids)
  id       = each.value
}

check "subnets_match_vpc_and_are_public" {
  assert {
    condition = alltrue([
      for subnet in data.aws_subnet.selected :
      subnet.vpc_id == data.aws_vpc.selected.id && subnet.map_public_ip_on_launch
    ])
    error_message = "Every configured subnet must belong to the selected VPC and map public IPs on launch."
  }
}
