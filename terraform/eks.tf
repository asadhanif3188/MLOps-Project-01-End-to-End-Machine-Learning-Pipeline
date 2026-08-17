# Amazon EKS platform (Sprint 6, PR 4).
#
# The managed Kubernetes platform the existing MLOps batch Job runs on: an EKS
# control plane, one small managed node group, and only the core EKS addons.
# This PR consumes what the earlier PRs built — the VPC/subnets (PR 2) and the
# two IAM roles (PR 3) — and adds no networking or IAM of its own beyond the
# cluster-managed security group EKS creates automatically.
#
# Design rationale (cluster size, node sizing, Kubernetes version, endpoint and
# security posture, cost, operational limits) lives in ADR-017 and
# terraform/README.md § EKS platform, not repeated in these comments. Common
# Project/Environment/Owner tags are applied to every resource automatically via
# the provider's default_tags (providers.tf); only the resource-specific Name
# tag (and workload labels) are set below.
#
# Deliberately NOT here (Sprint 6 non-goals): no GPUs, no Cluster Autoscaler /
# autoscaling, no service mesh, no ingress controller, no observability stack,
# no extra AWS services, and no application/workload resources — the Kubernetes
# Job stays in Kustomize (separation of concerns), provisioned in a later PR.

# --- EKS control plane --------------------------------------------------------
# Purpose: the managed Kubernetes API/control plane. It assumes the dedicated
# control-plane role from PR 3 (aws_iam_role.eks_cluster) to manage cluster AWS
# resources (cluster ENIs, the cluster security group) on our behalf.
#
# Networking: the control plane places its cross-account ENIs across both the
# private and public subnets from PR 2 so it can reach nodes (which live in the
# private subnets) and so public/internal load-balancer subnet discovery works
# in later workloads.
#
# Endpoint/security: SECURE BY DEFAULT (closes finding H-02). The API server is
# PRIVATE-ONLY out of the box — endpoint_private_access defaults true and
# endpoint_public_access defaults false, so nothing is exposed to the internet
# without a deliberate choice. Public access is an explicit opt-in that REQUIRES
# a scoped public_access_cidrs allow-list: an unrestricted 0.0.0.0/0 (any /0) is
# rejected by the variable's validation, and the preconditions below reject both
# "public on with an empty CIDR list" (EKS would silently fall back to 0.0.0.0/0)
# and "both endpoints off" (an unreachable API). Reaching a private-only endpoint
# requires in-VPC access (bastion/VPN/SSM/in-VPC runner); for a workstation
# validation run, opt into public access scoped to your own IP. See ADR-022 and
# terraform/README.md § EKS platform.
#
# Access model: EXPLICIT ACCESS ENTRIES (closes finding H-03). authentication_mode
# defaults to API (access entries only — no aws-auth ConfigMap backdoor) and
# bootstrap_cluster_creator_admin_permissions is FALSE, so the principal that runs
# `apply` gets NO implicit cluster-admin. Human/automation access is granted only
# by the aws_eks_access_entry + aws_eks_access_policy_association resources below,
# driven by var.cluster_access_entries — declared, scoped, and independent of who
# created the cluster. The managed node group's access entry is created by EKS
# automatically and is deliberately NOT declared here. See ADR-023 and
# terraform/README.md § EKS access management.
resource "aws_eks_cluster" "this" {
  name     = "${local.name_prefix}-eks"
  version  = var.kubernetes_version
  role_arn = aws_iam_role.eks_cluster.arn

  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    endpoint_private_access = var.cluster_endpoint_private_access
    endpoint_public_access  = var.cluster_endpoint_public_access
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
  }

  # Secure-by-default guardrails, enforced at plan time (H-02). These are
  # executable invariants, not documentation: they fail `plan`/`apply` (and the
  # offline `terraform test` suite) before an insecure cluster can be created.
  lifecycle {
    precondition {
      condition     = var.cluster_endpoint_private_access || var.cluster_endpoint_public_access
      error_message = "EKS API access is fully disabled: at least one of cluster_endpoint_private_access or cluster_endpoint_public_access must be true, otherwise the Kubernetes API server is unreachable."
    }

    precondition {
      condition     = !var.cluster_endpoint_public_access || length(var.cluster_endpoint_public_access_cidrs) > 0
      error_message = "cluster_endpoint_public_access is true but cluster_endpoint_public_access_cidrs is empty. EKS treats an empty public CIDR list as 0.0.0.0/0 (open to the entire internet); to opt into public access you MUST set an explicit, scoped operator IP/CIDR allow-list."
    }
  }

  access_config {
    authentication_mode                         = var.cluster_authentication_mode
    bootstrap_cluster_creator_admin_permissions = var.cluster_bootstrap_creator_admin_permissions
  }

  enabled_cluster_log_types = var.cluster_enabled_log_types

  tags = {
    Name = "${local.name_prefix}-eks"
  }

  # The control-plane role must carry AmazonEKSClusterPolicy before the cluster
  # is created, otherwise EKS cannot manage its AWS resources.
  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policy]
}

# --- Managed node group -------------------------------------------------------
# Purpose: the EC2 worker capacity that runs pods. AWS manages the launch
# template, instance lifecycle, and node bootstrap; the nodes assume the
# dedicated node role from PR 3 (aws_iam_role.eks_node).
#
# Placement: private subnets only — nodes have no public IPs and reach the
# internet outbound through the PR 2 NAT gateway (image/dataset/package pulls),
# with no inbound exposure. Sizing is intentionally small (see variables): a
# fixed pair of t3.medium on-demand nodes, enough for the batch Job plus EKS
# system pods. min = max = desired, and no Cluster Autoscaler is installed, so
# the group does not scale on its own.
resource "aws_eks_node_group" "this" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "${local.name_prefix}-ng"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id

  scaling_config {
    desired_size = var.node_desired_size
    min_size     = var.node_min_size
    max_size     = var.node_max_size
  }

  instance_types = var.node_instance_types
  capacity_type  = var.node_capacity_type
  disk_size      = var.node_disk_size

  # Amazon Linux 2023 is the current EKS-optimized AMI family (Amazon Linux 2 is
  # deprecated for recent Kubernetes versions). No custom AMI, no SSH remote
  # access block is configured — nodes are managed, not logged into.
  ami_type = "AL2023_x86_64_STANDARD"

  update_config {
    max_unavailable = 1
  }

  labels = {
    role = "mlops-workload"
  }

  tags = {
    Name = "${local.name_prefix}-ng"
  }

  # Nodes must have their three managed policies (worker, CNI, ECR read-only)
  # before they launch, or the kubelet/CNI cannot register the node.
  depends_on = [aws_iam_role_policy_attachment.eks_node]
}

# --- Core EKS addons ----------------------------------------------------------
# Only the three addons EKS itself needs to be a functioning cluster:
#   - vpc-cni    : pod networking (VPC-native pod IPs). Runs with the node role's
#                  AmazonEKS_CNI_Policy from PR 3 — no separate IRSA role in this
#                  PR (a documented follow-up; see ADR-016/-017).
#   - coredns    : in-cluster DNS. Scheduled on the worker nodes, so it depends
#                  on the node group existing.
#   - kube-proxy : service networking on each node.
# No optional addons (EBS CSI, EFS, ALB controller, observability agents, etc.):
# the batch Job uses none of them, and adding unused addons would violate the
# sprint's least-footprint / no-observability-stack non-goals.
#
# addon_version is intentionally omitted: EKS then installs the default addon
# version for the pinned cluster version, which is deterministic per Kubernetes
# version and keeps the addons in lock-step with the control plane. OVERWRITE
# lets the managed addon take over the self-managed defaults EKS installs at
# cluster creation.
locals {
  eks_core_addons = toset(["vpc-cni", "coredns", "kube-proxy"])
}

resource "aws_eks_addon" "this" {
  for_each = local.eks_core_addons

  cluster_name = aws_eks_cluster.this.name
  addon_name   = each.value

  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  tags = {
    Name = "${local.name_prefix}-addon-${each.value}"
  }

  # coredns needs schedulable nodes; gating all three on the node group keeps
  # addon reconciliation deterministic.
  depends_on = [aws_eks_node_group.this]
}

# --- EKS access entries (Sprint 7, PR 3) --------------------------------------
# Explicit cluster access, replacing the removed cluster-creator-admin bootstrap
# (closes finding H-03). Each map entry in var.cluster_access_entries becomes an
# access entry (registers the IAM principal with the cluster) plus a policy
# association (grants it a scoped AWS-managed EKS access policy). Nothing here is
# tied to the creating principal, and no personal ARN is committed — the map is
# populated from a git-ignored terraform.tfvars.
#
# Why Terraform/CI needs NO admin entry of its own: this root module manages only
# AWS-API resources (there is no kubernetes/helm provider), so Terraform never
# calls the Kubernetes API. It therefore does not need — and is not granted — a
# cluster access entry. Access entries exist purely for the humans/automation that
# run kubectl against the cluster.
#
# The managed node group's access entry (type EC2_LINUX for the node role) is
# created automatically by EKS for managed node groups; declaring it here would
# collide with that AWS-managed entry, so it is intentionally omitted.
resource "aws_eks_access_entry" "this" {
  for_each = var.cluster_access_entries

  cluster_name  = aws_eks_cluster.this.name
  principal_arn = each.value.principal_arn
  type          = "STANDARD"

  tags = {
    Name = "${local.name_prefix}-access-${each.key}"
  }
}

# Associates the scoped AWS-managed EKS access policy with each principal. The
# policy ARN is an EKS cluster-access-policy ARN (service "eks", not "iam"); the
# partition is resolved (data.aws_partition, from iam.tf) so it is correct in any
# AWS partition. access_scope narrows an entry to specific namespaces when
# requested; cluster scope is the default for an operator that manages the whole
# validation cluster.
resource "aws_eks_access_policy_association" "this" {
  for_each = var.cluster_access_entries

  cluster_name  = aws_eks_cluster.this.name
  principal_arn = each.value.principal_arn
  policy_arn    = "arn:${data.aws_partition.current.partition}:eks::aws:cluster-access-policy/${each.value.policy}"

  access_scope {
    type       = each.value.access_scope
    namespaces = each.value.access_scope == "namespace" ? each.value.namespaces : null
  }

  # The principal must be registered as an access entry before a policy can be
  # associated with it.
  depends_on = [aws_eks_access_entry.this]
}

# --- Usability & least-privilege notes ----------------------------------------
# Two related guardrails are handled by SECURE DEFAULTS + DOCS + VALIDATION rather
# than by resource preconditions/checks, deliberately:
#
#   - Not creating an unusable cluster. With creator-admin bootstrap off (H-03), a
#     cluster with an EMPTY cluster_access_entries has no HUMAN administrator — but
#     it is NOT broken: the managed node group still gets its EKS-created access
#     entry and joins, and addons run. An empty map is the safe, secure default;
#     the operator adds their own principal in a git-ignored terraform.tfvars
#     before validating (mirroring the empty public-CIDR default of H-02). This is
#     documented at the variable, in terraform.tfvars.example, and in
#     terraform/README.md § EKS access management. A precondition/check is NOT used
#     because "no entries yet" is a legitimate intermediate state and would either
#     block the private-by-default apply or break the credential-free `terraform
#     test` gate (check-block failures fail test runs).
#
#   - Avoiding cluster-admin for convenience. The DEFAULT policy for an entry is
#     AmazonEKSAdminPolicy (scoped admin, NOT system:masters); AmazonEKSClusterAdminPolicy
#     is accepted by validation but called out in the docs as a last resort. The
#     variable validation restricts policies to the AWS-managed EKS access-policy
#     set, so an arbitrary broad IAM policy cannot be associated.
