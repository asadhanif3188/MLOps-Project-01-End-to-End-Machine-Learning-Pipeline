# AWS KMS — customer-managed key for EKS Kubernetes Secret envelope encryption
# (Sprint 7, PR 5 — closes Sprint 6 finding M-02).
#
# Problem (M-02): by default EKS encrypts etcd at rest with an AWS-OWNED key, so
# Kubernetes Secret objects get no *customer-managed* envelope-encryption layer.
# A dedicated CMK gives us a second, independently controlled encryption layer
# (KMS envelope encryption): the API server encrypts each Secret's data with a
# data key that this CMK wraps, and access to the CMK is auditable in CloudTrail
# and revocable by us. This file creates that key; eks.tf wires it into the
# cluster's encryption_config so Secrets are ACTUALLY encrypted with it — not
# merely "a key that exists next to the cluster".
#
# Design rationale (why a CMK, key-policy scoping, rotation, the ephemeral-cluster
# deletion window, and the "encryption is one-way once enabled" caveat) lives in
# ADR-025 and terraform/README.md § Secrets encryption, not repeated in full here.
# Common Project/Environment/Owner tags are applied to every resource
# automatically via the provider's default_tags (providers.tf); only the
# resource-specific Name tag is set below.

# The CMK itself. Symmetric (the only key type EKS envelope encryption supports),
# with automatic annual key rotation ENABLED — AWS rotates the backing key
# material yearly with no re-encryption or config change required, so long-lived
# keys do not accumulate unbounded exposure to a single key version. Rotation is a
# no-cost, no-downtime hardening and is always on here (not a variable): there is
# no reason to run this key without it.
#
# deletion_window_in_days is the ONLY tunable (var.kms_key_deletion_window_days):
# it governs how long the key lingers in PendingDeletion after `terraform destroy`
# schedules its removal. It defaults to the 7-day minimum because this is a
# short-lived, single-operator validation cluster (ADR-020) — a torn-down
# environment should not leave a 30-day pending-deletion key (and its small
# monthly charge) hanging around. A persistent/production key would use a longer
# window as an accidental-deletion safety net.
resource "aws_kms_key" "eks_secrets" {
  description             = "Customer-managed CMK for EKS Kubernetes Secret envelope encryption (M-02)."
  deletion_window_in_days = var.kms_key_deletion_window_days
  enable_key_rotation     = true

  # Key policy — the authoritative access control for a KMS key (a key policy that
  # grants a principal is sufficient on its own in-account; it is why no separate
  # identity-based IAM policy is attached to the cluster role for KMS, keeping the
  # grant scoped to exactly this one key). Three statements, least-privilege:
  #
  #   1. EnableIAMRootAdministration — the AWS-canonical "prevent lock-out"
  #      statement present in every default KMS key policy. It delegates access
  #      control for this key to IAM within THIS account (account root principal).
  #      This is the one justified wildcard action (kms:*): without it the key can
  #      become unmanageable (no principal could ever grant, rotate, or schedule
  #      deletion), which AWS explicitly warns against. It does NOT grant any other
  #      account or service use of the key — only same-account IAM administration.
  #
  #   2. AllowEKSClusterRoleToUseTheKey — lets the EKS control-plane role (the
  #      identity EKS assumes for this cluster) perform exactly the cryptographic
  #      operations the KMS secrets provider needs: Encrypt/Decrypt the data keys,
  #      DescribeKey, and ListGrants. No kms:* and no wildcard action (e.g. no
  #      ReEncrypt*, which EKS does not need) — an explicit, minimal action list,
  #      scoped to this key (Resource "*" in a key policy means "the key this policy
  #      is attached to", not "all keys").
  #
  #   3. AllowEKSClusterRoleToCreateGrants — EKS creates a grant on the CMK so the
  #      managed control plane can use it; this is the documented requirement for
  #      EKS secrets encryption. Constrained with the kms:GrantIsForAWSResource
  #      condition so the role can ONLY create grants for AWS services, not
  #      arbitrary grantees.
  #
  # No principal is a bare "*", no cross-account access, and no broad kms:* is
  # granted to the workload/cluster role. See ADR-025 for the full justification.
  policy = jsonencode({
    Version = "2012-10-17"
    Id      = "${local.name_prefix}-eks-secrets-key-policy"
    Statement = [
      {
        Sid       = "EnableIAMRootAdministration"
        Effect    = "Allow"
        Principal = { AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid       = "AllowEKSClusterRoleToUseTheKey"
        Effect    = "Allow"
        Principal = { AWS = aws_iam_role.eks_cluster.arn }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:ListGrants",
        ]
        Resource = "*"
      },
      {
        Sid       = "AllowEKSClusterRoleToCreateGrants"
        Effect    = "Allow"
        Principal = { AWS = aws_iam_role.eks_cluster.arn }
        Action    = "kms:CreateGrant"
        Resource  = "*"
        Condition = {
          Bool = { "kms:GrantIsForAWSResource" = "true" }
        }
      },
    ]
  })

  tags = {
    Name = "${local.name_prefix}-eks-secrets"
  }
}

# A human-readable alias for the key so operators (and CloudTrail/console views)
# can identify it as the EKS-secrets CMK without memorising the key ID. Purely for
# operability; it grants no access.
resource "aws_kms_alias" "eks_secrets" {
  name          = "alias/${local.name_prefix}-eks-secrets"
  target_key_id = aws_kms_key.eks_secrets.key_id
}
