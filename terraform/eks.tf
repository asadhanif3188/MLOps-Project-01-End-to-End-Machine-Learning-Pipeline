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
# Endpoint/security: both private and public API access are enabled by default
# so an operator can validate the cluster with kubectl while in-VPC traffic
# stays private; the public endpoint's source range is restricted via
# cluster_endpoint_public_access_cidrs (open only for first-run validation — see
# the variable's guidance). Access uses EKS access entries (API_AND_CONFIG_MAP)
# and bootstraps the creating principal as cluster admin so kubectl works without
# hand-editing the aws-auth ConfigMap.
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

  access_config {
    authentication_mode                         = "API_AND_CONFIG_MAP"
    bootstrap_cluster_creator_admin_permissions = true
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
