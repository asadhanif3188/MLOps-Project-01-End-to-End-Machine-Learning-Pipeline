# AWS IAM foundation for EKS (Sprint 6, PR 3).
#
# Two dedicated IAM roles the managed EKS cluster (a later PR) needs before it
# can be created: a control-plane role EKS itself assumes, and a worker-node
# role the EC2 instances in the managed node group assume. Nothing here creates
# EKS, EC2, or any application resource — only the roles, their trust
# relationships, and the AWS-managed policy attachments EKS requires.
#
# Design principles (rationale in ADR-016 and terraform/README.md § IAM, not
# repeated in these comments):
#   - Two roles, each dedicated to one purpose (cluster vs. node) — no shared or
#     multi-purpose role.
#   - Trust is scoped to exactly the AWS service that must assume each role
#     (eks.amazonaws.com / ec2.amazonaws.com), nothing wider.
#   - Permissions come from the AWS-managed policies EKS documents as required.
#     This PR authors NO inline/custom policy and NO wildcard grant of its own;
#     any wildcards live inside AWS-owned policies, are maintained by AWS, and
#     are called out in the docs.
#   - No AdministratorAccess, no PowerUserAccess, no static credentials, no
#     access/secret keys, no instance profile secrets — none are created here.
#
# Common Project/Environment/Owner tags are applied to every resource
# automatically via the provider's default_tags (providers.tf); only the
# resource-specific Name tag is set below.

# Partition is resolved (not hard-coded to "aws") so the managed-policy ARNs are
# correct in any partition (aws, aws-us-gov, aws-cn) without editing this file.
data "aws_partition" "current" {}

# --- EKS cluster (control-plane) role -----------------------------------------
# Purpose: the identity the EKS *managed control plane* assumes to manage AWS
# resources on the cluster's behalf — e.g. creating the cluster elastic network
# interfaces in the subnets and managing the cluster security group.
#
# Trust relationship: only the EKS service principal (eks.amazonaws.com) may
# assume it, via sts:AssumeRole. No account, user, or role principal is trusted.
resource "aws_iam_role" "eks_cluster" {
  name        = "${local.name_prefix}-eks-cluster-role"
  description = "EKS control-plane role assumed by eks.amazonaws.com to manage cluster AWS resources."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "eks.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "${local.name_prefix}-eks-cluster-role"
  }
}

# AmazonEKSClusterPolicy is the single AWS-managed policy EKS requires on the
# control-plane role. It grants the control plane the EC2/ELB/CloudWatch actions
# it needs to run the cluster. It is AWS-owned and AWS-maintained.
#
# Intentionally NOT attached: AmazonEKSVPCResourceController. That policy is only
# needed for "security groups for pods", which this batch-Job workload does not
# use. Omitting it keeps the control-plane role to the minimum EKS mandates.
resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKSClusterPolicy"
}

# --- EKS worker-node role -----------------------------------------------------
# Purpose: the instance-profile identity each EC2 worker node (in the managed
# node group) assumes so the kubelet can register with the cluster, the VPC CNI
# can wire pod networking, and the container runtime can pull images from ECR.
#
# Trust relationship: only the EC2 service principal (ec2.amazonaws.com) may
# assume it — worker nodes are EC2 instances.
resource "aws_iam_role" "eks_node" {
  name        = "${local.name_prefix}-eks-node-role"
  description = "EKS worker-node role assumed by ec2.amazonaws.com for the managed node group instances."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "${local.name_prefix}-eks-node-role"
  }
}

# The three AWS-managed policies AWS documents as required for a managed node
# group. Each is AWS-owned and AWS-maintained; this PR attaches, it does not
# author them.
locals {
  # Keyed (not a list) so the attachments are stable if the set ever changes.
  eks_node_managed_policies = {
    # Lets the kubelet describe the cluster and EC2/Auto Scaling resources needed
    # to join the node to the cluster.
    worker_node = "AmazonEKSWorkerNodePolicy"

    # Required by the Amazon VPC CNI plugin to attach ENIs and assign pod IPs.
    # This is the broadest of the three (it includes some ec2:*NetworkInterface
    # and ec2:Describe* actions AWS scopes inside the managed policy); attaching
    # it to the node role is the standard EKS setup. Moving it to a dedicated
    # IRSA role is a hardening option deferred to the EKS PR (see ADR-016).
    cni = "AmazonEKS_CNI_Policy"

    # READ-ONLY pull access to Amazon ECR so the container runtime can fetch
    # images. Read-only by design — nodes cannot push or mutate registries.
    ecr_read_only = "AmazonEC2ContainerRegistryReadOnly"
  }
}

resource "aws_iam_role_policy_attachment" "eks_node" {
  for_each = local.eks_node_managed_policies

  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/${each.value}"
}
