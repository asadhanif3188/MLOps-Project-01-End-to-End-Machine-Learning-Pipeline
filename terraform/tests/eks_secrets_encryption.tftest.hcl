# Terraform contract tests for EKS Kubernetes Secret envelope encryption
# (Sprint 7, PR 5).
#
# OFFLINE, credential-free `terraform test` runs: `mock_provider "aws"` replaces
# the real provider so no AWS API is ever called, and every run uses
# `command = plan`, so nothing is created. The suite pins the ENCRYPTION contract
# that closes Sprint 6 finding M-02: Kubernetes Secrets are envelope-encrypted with
# a dedicated customer-managed KMS key, the key is rotated and least-privilege, and
# — critically — the CLUSTER is actually configured to use it (not merely that a
# key exists). It is executable proof, in the same no-AWS spirit as the rest of the
# Terraform CI gate (ADR-019, ADR-025). It complements eks_api_security (H-02),
# eks_access_control (H-03), and eks_cni_identity (M-01).
#
# Run locally with:  terraform test   (from terraform/)

# Pin the provider-context data sources to realistic values. The default random
# mocks would otherwise feed an invalid partition into the KMS/policy ARNs and an
# empty AZ list into the network slice(), failing the plan on unrelated resources.
# aws_caller_identity is pinned so the key-policy root ARN is a well-formed account
# root principal.
mock_provider "aws" {
  mock_data "aws_partition" {
    defaults = {
      partition = "aws"
    }
  }

  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
    }
  }

  mock_data "aws_availability_zones" {
    defaults = {
      names = ["us-east-1a", "us-east-1b", "us-east-1c"]
    }
  }

  # The least-privilege run block below uses `command = apply` (see its comment).
  # During a mocked apply the provider would otherwise invent a random string for
  # each computed ARN, which EKS's role_arn / key_arn format validation rejects.
  # Pin well-formed ARNs so the mocked apply is representative and the key policy
  # JSON (which interpolates the cluster role ARN) is fully known.
  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/mock-eks-role"
    }
  }

  mock_resource "aws_kms_key" {
    defaults = {
      arn = "arn:aws:kms:us-east-1:123456789012:key/00000000-0000-0000-0000-000000000000"
    }
  }

  # The mocked apply also evaluates every module output. eks_cluster_oidc_issuer_url
  # indexes the cluster's computed `identity` block, which the mock leaves empty by
  # default; provide a representative value so output evaluation does not fail.
  mock_resource "aws_eks_cluster" {
    defaults = {
      identity = [{ oidc = [{ issuer = "https://oidc.eks.us-east-1.amazonaws.com/id/MOCK" }] }]
    }
  }
}

# The CLUSTER is configured to envelope-encrypt Secrets. This is the core M-02
# guard and the distinction the task calls out: "KMS key exists" is NOT the same
# as "EKS Secrets are configured to use it". If a future change drops the
# encryption_config block (or stops encrypting "secrets"), these assertions fail.
run "cluster_encrypts_secrets_with_a_cmk" {
  command = plan

  assert {
    condition     = length(aws_eks_cluster.this.encryption_config) == 1
    error_message = "The EKS cluster must declare an encryption_config block so Kubernetes Secrets are envelope-encrypted with a customer-managed KMS key (M-02)."
  }

  assert {
    condition     = contains(aws_eks_cluster.this.encryption_config[0].resources, "secrets")
    error_message = "The EKS encryption_config must cover the \"secrets\" resource — that is the Kubernetes object type M-02 requires to be envelope-encrypted."
  }

  # A KMS provider key is wired into the encryption_config (the block is present).
  # The exact key ARN is a computed value not known until apply, so it cannot be
  # compared here under `command = plan`; the reference itself — encryption_config's
  # provider.key_arn = aws_kms_key.eks_secrets.arn — is what `terraform validate`
  # and the graph confirm, and the eks_secrets_encryption_key_arn output surfaces it
  # for live post-apply verification (see terraform/README.md § Secrets encryption).
  assert {
    condition     = length(aws_eks_cluster.this.encryption_config[0].provider) == 1
    error_message = "The EKS encryption_config must specify a KMS provider key (the dedicated EKS-secrets CMK), proving the key is associated with the cluster, not just created alongside it."
  }
}

# The key is a genuine hardening: rotation ON and a valid deletion window. A key
# without rotation would regress the posture even though encryption is "enabled".
run "cmk_is_rotated_and_has_a_valid_deletion_window" {
  command = plan

  assert {
    condition     = aws_kms_key.eks_secrets.enable_key_rotation == true
    error_message = "The EKS-secrets KMS key must have automatic key rotation enabled — a no-cost hardening that bounds exposure to any single key version."
  }

  assert {
    condition     = aws_kms_key.eks_secrets.deletion_window_in_days >= 7 && aws_kms_key.eks_secrets.deletion_window_in_days <= 30
    error_message = "The EKS-secrets KMS key deletion window must be within the AWS-permitted 7-30 day range."
  }
}

# The key alias is present and correctly named — the operable identifier used in
# the console/CloudTrail and by the evidence outputs.
run "cmk_has_the_expected_alias" {
  command = plan

  assert {
    condition     = aws_kms_alias.eks_secrets.name == "alias/mlops-pipeline-dev-eks-secrets"
    error_message = "The EKS-secrets KMS alias must be alias/<project>-<environment>-eks-secrets (mlops-pipeline-dev by default)."
  }
}

# The key policy is LEAST-PRIVILEGE: it grants the EKS cluster role exactly the
# operations envelope encryption needs (Decrypt + a constrained CreateGrant), and
# grants NO principal bare "*" use of the key. This guards the security-requirement
# "no wildcard KMS permissions unless unavoidable and justified": the only kms:*
# is the account-root administration statement, not a use grant.
#
# This block uses `command = apply` (still fully offline and credential-free under
# mock_provider — nothing real is created) rather than `plan`: the policy string
# interpolates the EKS cluster role ARN, a value not known until apply, so the
# NEGATIVE "no bare public principal" assertion cannot be evaluated at plan time.
# The mocked apply resolves the computed ARN so the whole policy JSON is known and
# every statement can be checked.
run "cmk_key_policy_is_least_privilege" {
  command = apply

  assert {
    condition     = strcontains(aws_kms_key.eks_secrets.policy, "kms:Decrypt")
    error_message = "The key policy must grant the EKS cluster role kms:Decrypt so the control plane can decrypt Secret data keys."
  }

  assert {
    condition     = strcontains(aws_kms_key.eks_secrets.policy, "kms:CreateGrant")
    error_message = "The key policy must let the EKS cluster role create the grant EKS uses to access the CMK for secrets encryption."
  }

  assert {
    condition     = strcontains(aws_kms_key.eks_secrets.policy, "kms:GrantIsForAWSResource")
    error_message = "The CreateGrant permission must be constrained with the kms:GrantIsForAWSResource condition, so the role can only grant AWS services (not arbitrary principals)."
  }

  # No statement may grant key USE to a bare public/any principal. jsonencode emits
  # compact JSON, so a public principal would appear literally as {"AWS":"*"}.
  assert {
    condition     = !strcontains(aws_kms_key.eks_secrets.policy, "\"AWS\":\"*\"")
    error_message = "The EKS-secrets key policy must NOT grant any bare \"*\" AWS principal — access is limited to the account root (administration) and the EKS cluster role (use)."
  }
}
