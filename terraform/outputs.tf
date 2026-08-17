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

# --- EKS platform (Sprint 6, PR 4) --------------------------------------------
# Connection/inspection details for the provisioned cluster. All are
# non-sensitive: the cluster name, endpoint URL, version, and the cluster
# security-group ID are not secrets, and the API endpoint is access-controlled
# by IAM plus the public-CIDR allow-list — knowing the URL grants nothing. No
# kubeconfig, token, or certificate is emitted here; operators fetch short-lived
# credentials with `aws eks update-kubeconfig` (see configure_kubectl below).

output "eks_cluster_name" {
  description = "Name of the EKS cluster. Used by `aws eks update-kubeconfig` and by the Kubernetes workload PR."
  value       = aws_eks_cluster.this.name
}

output "eks_cluster_endpoint" {
  description = "HTTPS endpoint of the EKS Kubernetes API server. Not a secret: access is gated by IAM and the public-access CIDR allow-list."
  value       = aws_eks_cluster.this.endpoint
}

output "eks_cluster_version" {
  description = "Kubernetes minor version running on the EKS control plane (the explicitly pinned version)."
  value       = aws_eks_cluster.this.version
}

output "eks_cluster_security_group_id" {
  description = "ID of the cluster security group EKS created and manages for control-plane/node communication."
  value       = aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
}

output "eks_cluster_oidc_issuer_url" {
  description = "OIDC issuer URL of the cluster. The anchor for future IAM Roles for Service Accounts (IRSA), including the deferred CNI-via-IRSA hardening (ADR-016/-017)."
  value       = aws_eks_cluster.this.identity[0].oidc[0].issuer
}

output "eks_node_group_name" {
  description = "Name of the managed node group backing the cluster's worker capacity."
  value       = aws_eks_node_group.this.node_group_name
}

output "configure_kubectl" {
  description = "Ready-to-run command that writes a kubeconfig entry for this cluster using the caller's AWS credentials (fetches a short-lived token; nothing sensitive is stored in Terraform state by this output)."
  value       = "aws eks update-kubeconfig --region ${data.aws_region.current.name} --name ${aws_eks_cluster.this.name}"
}

# --- EKS access management (Sprint 7, PR 3) -----------------------------------
# Describe the explicit access model (closes H-03) for inspection/audit. These
# confirm the secure posture — access-entry auth, no creator-admin bootstrap —
# and enumerate which access-entry KEYS and policies are configured. The map is
# keyed by the operator's chosen labels and exposes the scoped POLICY per entry,
# not the principal ARNs (those embed the AWS account ID; retrieve them from state
# or the AWS console when auditing).

output "eks_authentication_mode" {
  description = "EKS cluster authentication mode in effect (\"API\" = access entries only; \"API_AND_CONFIG_MAP\" = access entries plus legacy aws-auth). Confirms the access-entry model that replaced creator-admin bootstrap (H-03)."
  value       = aws_eks_cluster.this.access_config[0].authentication_mode
}

output "eks_bootstrap_creator_admin_permissions" {
  description = "Whether the cluster grants the creating principal implicit cluster-admin. Expected false (finding H-03): access is granted explicitly via access entries, not to whoever ran apply."
  value       = aws_eks_cluster.this.access_config[0].bootstrap_cluster_creator_admin_permissions
}

output "eks_access_entry_policies" {
  description = "Map of configured access-entry label -> scoped EKS access policy granted to that principal. Non-sensitive: shows the shape of granted access (which policy, which scope) without exposing principal ARNs. Empty if no access entries are configured."
  value = {
    for k, e in var.cluster_access_entries : k => {
      policy       = e.policy
      access_scope = e.access_scope
      namespaces   = e.access_scope == "namespace" ? e.namespaces : []
    }
  }
}

# --- Container registry / ECR (Sprint 7, PR 1) --------------------------------
# The registry URI/ARN embed the AWS account ID, which this project treats as
# sensitive (see the aws_account_id output). They are therefore marked `sensitive`
# so the account identifier stays out of console/CI logs; retrieve them explicitly
# with `terraform output -raw <name>` when pushing the image or pointing the
# Kustomize overlay at the registry. The repository *name* is non-sensitive.

output "ecr_repository_name" {
  description = "Name of the Terraform-managed ECR repository holding the workload image."
  value       = aws_ecr_repository.this.name
}

output "ecr_repository_url" {
  description = "Registry URI of the ECR repository (<account>.dkr.ecr.<region>.amazonaws.com/<name>). Tag/push the image here and set it on the k8s AWS overlay. Sensitive: the URI contains the AWS account ID."
  value       = aws_ecr_repository.this.repository_url
  sensitive   = true
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository. Sensitive: the ARN contains the AWS account ID."
  value       = aws_ecr_repository.this.arn
  sensitive   = true
}
