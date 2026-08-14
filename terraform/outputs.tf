# Outputs.
#
# In this foundation PR the outputs describe the *context* Terraform is
# operating in (region, account, naming, tags) rather than provisioned
# infrastructure, since no resources exist yet. Later PRs add outputs for the
# VPC, subnets, IAM roles, and EKS connection details.
#
# The account ID is marked `sensitive` so it is not printed to the console or CI
# logs; it is still available to tooling via `terraform output -raw`.

output "aws_region" {
  description = "AWS region Terraform is operating in (resolved from the active provider configuration)."
  value       = data.aws_region.current.name
}

output "aws_account_id" {
  description = "AWS account ID resolved from the active credentials. Marked sensitive to keep the account identifier out of logs."
  value       = data.aws_caller_identity.current.account_id
  sensitive   = true
}

output "name_prefix" {
  description = "Canonical \"<project>-<environment>\" prefix that later PRs apply to resource names."
  value       = local.name_prefix
}

output "common_tags" {
  description = "The common tag set applied to every resource via the provider's default_tags."
  value       = local.common_tags
}

# --- Network (Sprint 6, PR 2) -------------------------------------------------
# These describe the provisioned VPC and are the hand-off contract for the EKS
# PR: the cluster and node group consume the VPC and subnet IDs directly, and
# EKS control-plane/node-group placement uses the private subnets.

output "vpc_id" {
  description = "ID of the VPC hosting the platform. Consumed by the EKS cluster and node group in a later PR."
  value       = aws_vpc.this.id
}

output "vpc_cidr_block" {
  description = "IPv4 CIDR block of the VPC (useful for security-group and peering rules in later PRs)."
  value       = aws_vpc.this.cidr_block
}

output "availability_zones" {
  description = "Availability Zones the subnets are spread across (discovered at plan time, not hard-coded)."
  value       = local.azs
}

output "public_subnet_ids" {
  description = "IDs of the public subnets (NAT, internet gateway, and future public load balancers). Tagged kubernetes.io/role/elb."
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets where EKS worker nodes run. This is the primary input to the EKS node group in a later PR. Tagged kubernetes.io/role/internal-elb."
  value       = aws_subnet.private[*].id
}

output "internet_gateway_id" {
  description = "ID of the VPC's internet gateway."
  value       = aws_internet_gateway.this.id
}

output "nat_gateway_ids" {
  description = "IDs of the NAT gateway(s) providing outbound egress for the private subnets (empty if NAT is disabled)."
  value       = aws_nat_gateway.this[*].id
}

output "nat_public_ips" {
  description = "Public Elastic IP(s) of the NAT gateway(s) — the stable egress address(es) for private-subnet traffic (useful for allow-listing). Empty if NAT is disabled."
  value       = aws_eip.nat[*].public_ip
}

# --- IAM (Sprint 6, PR 3) -----------------------------------------------------
# The EKS PR consumes these role identities: the cluster role is passed to the
# EKS cluster, the node role to the managed node group's instance profile.
#
# Role *names* are non-sensitive. Role *ARNs* embed the AWS account ID, which
# this project deliberately treats as sensitive (see the aws_account_id output),
# so the ARN outputs are marked `sensitive` to keep the account ID out of logs —
# retrieve them explicitly with `terraform output -raw <name>` when wiring EKS.

output "eks_cluster_role_name" {
  description = "Name of the EKS control-plane IAM role (assumed by eks.amazonaws.com)."
  value       = aws_iam_role.eks_cluster.name
}

output "eks_cluster_role_arn" {
  description = "ARN of the EKS control-plane IAM role, passed to the EKS cluster in a later PR. Sensitive: the ARN contains the AWS account ID."
  value       = aws_iam_role.eks_cluster.arn
  sensitive   = true
}

output "eks_node_role_name" {
  description = "Name of the EKS worker-node IAM role (assumed by ec2.amazonaws.com)."
  value       = aws_iam_role.eks_node.name
}

output "eks_node_role_arn" {
  description = "ARN of the EKS worker-node IAM role, passed to the managed node group in a later PR. Sensitive: the ARN contains the AWS account ID."
  value       = aws_iam_role.eks_node.arn
  sensitive   = true
}
