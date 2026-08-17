# Terraform contract tests for the VPC CNI identity model (Sprint 7, PR 4).
#
# OFFLINE, credential-free `terraform test` runs: `mock_provider "aws"` replaces
# the real provider so no AWS API is ever called, and every run uses
# `command = plan`, so nothing is created. The suite pins the CNI-ISOLATION
# contract that closes Sprint 6 finding M-01: the Amazon VPC CNI's AWS
# permissions no longer sit on the worker-node instance profile, but on a
# dedicated role assumed ONLY by the aws-node service account via EKS Pod
# Identity. It is executable proof — not merely documentation — that the node
# role and the CNI application permissions are separated, in the same no-AWS
# spirit as the rest of the Terraform CI gate (ADR-019, ADR-024). It complements
# eks_access_control.tftest.hcl (H-03) and eks_api_security.tftest.hcl (H-02).
#
# Run locally with:  terraform test   (from terraform/)

# Pin the provider-context data sources to realistic values. The default random
# mocks would otherwise feed an invalid partition into the IAM/EKS policy ARNs and
# an empty AZ list into the network slice(), failing the plan on unrelated resources.
mock_provider "aws" {
  mock_data "aws_partition" {
    defaults = {
      partition = "aws"
    }
  }

  mock_data "aws_availability_zones" {
    defaults = {
      names = ["us-east-1a", "us-east-1b", "us-east-1c"]
    }
  }
}

# The CNI permission is NO LONGER on the node role. This is the core M-01
# regression guard: if a future change re-attaches AmazonEKS_CNI_Policy to the
# node instance profile (or otherwise puts CNI permissions back on the node), one
# of these assertions fails.
run "cni_policy_is_not_on_the_node_role" {
  command = plan

  assert {
    condition     = length(aws_iam_role_policy_attachment.eks_node) == 2
    error_message = "The node role must carry exactly two managed policies (worker-node + ECR read-only). AmazonEKS_CNI_Policy must NOT be attached to the node role (M-01)."
  }

  assert {
    condition = alltrue([
      for k, a in aws_iam_role_policy_attachment.eks_node :
      !strcontains(a.policy_arn, "AmazonEKS_CNI_Policy")
    ])
    error_message = "No node-role policy attachment may reference AmazonEKS_CNI_Policy — CNI permissions are isolated to the dedicated VPC CNI role (M-01)."
  }
}

# The node role KEEPS the permissions the node itself needs. Removing CNI must not
# collaterally strip the kubelet/runtime permissions other components depend on.
# AmazonEKSWorkerNodePolicy is doubly load-bearing: besides letting the kubelet
# join, it carries eks-auth:AssumeRoleForPodIdentity, which the pod-identity agent
# needs (under the node profile) to deliver the CNI credentials to aws-node — so
# swapping it for a narrower "join-only" policy would break Pod Identity. This
# by-name assertion fails such a swap in CI (see ADR-024).
run "node_role_keeps_its_own_permissions" {
  command = plan

  assert {
    condition     = strcontains(aws_iam_role_policy_attachment.eks_node["worker_node"].policy_arn, "AmazonEKSWorkerNodePolicy")
    error_message = "The node role must retain AmazonEKSWorkerNodePolicy so the kubelet can join the cluster AND the pod-identity agent can call eks-auth:AssumeRoleForPodIdentity to serve the VPC CNI credentials."
  }

  assert {
    condition     = strcontains(aws_iam_role_policy_attachment.eks_node["ecr_read_only"].policy_arn, "AmazonEC2ContainerRegistryReadOnly")
    error_message = "The node role must retain AmazonEC2ContainerRegistryReadOnly so the container runtime can pull images."
  }
}

# A dedicated VPC CNI role exists and carries exactly the CNI permission — the
# permission was moved, not dropped.
run "dedicated_cni_role_carries_the_cni_policy" {
  command = plan

  assert {
    condition     = aws_iam_role_policy_attachment.vpc_cni.policy_arn == "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
    error_message = "The dedicated VPC CNI role must carry AmazonEKS_CNI_Policy (the same AWS-managed policy moved off the node role), with an ARN built from the resolved partition."
  }
}

# The CNI role is assumable ONLY by EKS Pod Identity (pods.eks.amazonaws.com),
# with the sts:AssumeRole + sts:TagSession actions Pod Identity requires — not by
# EC2, a human, or an account principal.
run "cni_role_trusts_only_pod_identity" {
  command = plan

  assert {
    condition     = strcontains(aws_iam_role.vpc_cni.assume_role_policy, "pods.eks.amazonaws.com")
    error_message = "The VPC CNI role trust policy must trust the EKS Pod Identity service principal pods.eks.amazonaws.com."
  }

  assert {
    condition     = strcontains(aws_iam_role.vpc_cni.assume_role_policy, "sts:TagSession")
    error_message = "The VPC CNI role trust policy must allow sts:TagSession (required by EKS Pod Identity in addition to sts:AssumeRole)."
  }

  assert {
    condition     = !strcontains(aws_iam_role.vpc_cni.assume_role_policy, "ec2.amazonaws.com")
    error_message = "The VPC CNI role must NOT trust ec2.amazonaws.com — it is assumed by the aws-node pod via Pod Identity, not by EC2 instances."
  }
}

# The Pod Identity association binds the aws-node service account (kube-system) to
# the CNI role — the mechanism that delivers CNI credentials to the pod instead of
# the node instance profile.
run "pod_identity_binds_aws_node_service_account" {
  command = plan

  assert {
    condition     = aws_eks_pod_identity_association.vpc_cni.namespace == "kube-system"
    error_message = "The VPC CNI Pod Identity association must target the kube-system namespace where aws-node runs."
  }

  assert {
    condition     = aws_eks_pod_identity_association.vpc_cni.service_account == "aws-node"
    error_message = "The VPC CNI Pod Identity association must target the aws-node service account (the VPC CNI's service account)."
  }
}

# The eks-pod-identity-agent addon is installed — Pod Identity credentials are not
# delivered without it. This is the on-cluster half of the mechanism. Its
# conflict-resolution must let the managed addon take over the self-managed default
# EKS may install, consistent with the core addons.
run "pod_identity_agent_addon_is_installed" {
  command = plan

  assert {
    condition     = aws_eks_addon.pod_identity_agent.addon_name == "eks-pod-identity-agent"
    error_message = "The eks-pod-identity-agent addon must be installed so the VPC CNI can obtain credentials via Pod Identity."
  }

  assert {
    condition     = aws_eks_addon.pod_identity_agent.resolve_conflicts_on_create == "OVERWRITE" && aws_eks_addon.pod_identity_agent.resolve_conflicts_on_update == "OVERWRITE"
    error_message = "The eks-pod-identity-agent addon must resolve conflicts with OVERWRITE so the managed addon takes over any self-managed default."
  }
}
