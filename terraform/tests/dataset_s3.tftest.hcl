# Terraform contract tests for the dataset S3 store (Sprint 7, PR 8 — closes M-04).
#
# OFFLINE, credential-free `terraform test` runs: `mock_provider "aws"` replaces
# the real provider so no AWS API is ever called, and every run uses
# `command = plan` (except the CMK-policy run, which uses `apply` against the mock
# so the computed role ARN resolves and the policy JSON is inspectable — the same
# technique the MLflow-artifacts and EKS-secrets key-policy tests use). The suite
# pins the SECURITY and WORKLOAD-IDENTITY contract of the dataset store: the bucket
# is private + CMK-encrypted + versioned, the pipeline gets access via EKS Pod
# Identity (never static keys), and the access is READ-ONLY (no write/delete).
# Executable proof, in the same no-AWS spirit as the rest of the Terraform CI gate
# (ADR-019, ADR-027).
#
# Run locally with:  terraform test   (from terraform/)

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

  # Pin well-formed ARNs so the mocked apply (CMK-policy run) is representative —
  # the computed role ARN is interpolated into the key policy JSON, and EKS's
  # role_arn / key_arn format validation rejects the provider's random default.
  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/mock-role"
    }
  }

  mock_resource "aws_kms_key" {
    defaults = {
      arn = "arn:aws:kms:us-east-1:123456789012:key/00000000-0000-0000-0000-000000000000"
    }
  }

  mock_resource "aws_eks_cluster" {
    defaults = {
      identity = [{ oidc = [{ issuer = "https://oidc.eks.us-east-1.amazonaws.com/id/MOCK" }] }]
    }
  }
}

# The dataset bucket refuses ALL public access — it holds private input data and
# must never be internet-reachable.
run "dataset_bucket_blocks_all_public_access" {
  command = plan

  assert {
    condition = (
      aws_s3_bucket_public_access_block.datasets.block_public_acls &&
      aws_s3_bucket_public_access_block.datasets.block_public_policy &&
      aws_s3_bucket_public_access_block.datasets.ignore_public_acls &&
      aws_s3_bucket_public_access_block.datasets.restrict_public_buckets
    )
    error_message = "The dataset bucket must block all four public-access vectors (ACLs, policy, ignore ACLs, restrict public buckets)."
  }

  assert {
    condition     = aws_s3_bucket_ownership_controls.datasets.rule[0].object_ownership == "BucketOwnerEnforced"
    error_message = "The dataset bucket must enforce BucketOwnerEnforced (ACLs disabled), so access is IAM-only."
  }
}

# The bucket is encrypted at rest with a CUSTOMER-MANAGED KMS key and versioned —
# durability + recoverability + controllable/auditable encryption for the dataset.
run "dataset_bucket_is_kms_encrypted_and_versioned" {
  command = plan

  assert {
    condition = anytrue([
      for r in aws_s3_bucket_server_side_encryption_configuration.datasets.rule :
      r.apply_server_side_encryption_by_default[0].sse_algorithm == "aws:kms"
    ])
    error_message = "The dataset bucket must use SSE-KMS (aws:kms) with the customer-managed key, not the AWS-owned SSE-S3 key."
  }

  assert {
    condition     = aws_s3_bucket_versioning.datasets.versioning_configuration[0].status == "Enabled"
    error_message = "The dataset bucket must have versioning enabled to guard against accidental overwrite/delete of the dataset."
  }
}

# The dataset CMK is rotated and its key policy is least-privilege: only the
# dataset-reader role may use it, and it is granted only Decrypt (READ path) — NOT
# GenerateDataKey/Encrypt, because the pipeline never writes the dataset.
run "dataset_cmk_is_rotated_and_read_only_least_privilege" {
  command = apply

  assert {
    condition     = aws_kms_key.datasets.enable_key_rotation == true
    error_message = "The dataset CMK must have automatic key rotation enabled."
  }

  assert {
    condition     = aws_kms_alias.datasets.name == "alias/mlops-pipeline-dev-datasets"
    error_message = "The dataset CMK must have the environment-scoped alias alias/<name_prefix>-datasets."
  }

  assert {
    condition     = strcontains(aws_kms_key.datasets.policy, "kms:Decrypt")
    error_message = "The dataset CMK policy must grant the dataset-reader role kms:Decrypt (SSE-KMS read path)."
  }

  assert {
    condition     = !strcontains(aws_kms_key.datasets.policy, "kms:GenerateDataKey")
    error_message = "The dataset CMK policy must NOT grant GenerateDataKey — the pipeline only READS the dataset, so it needs no write-path key operation."
  }

  assert {
    condition     = strcontains(aws_kms_key.datasets.policy, "AllowDatasetReaderRoleToDecrypt")
    error_message = "The dataset CMK policy must scope key use to the dataset-reader role (least privilege)."
  }
}

# The pipeline gets dataset S3 access via EKS Pod Identity, bound to exactly the
# mlops/mlops-pipeline service account — not static keys, not the node profile.
run "dataset_access_is_workload_identity" {
  command = plan

  assert {
    condition     = strcontains(aws_iam_role.dataset_reader.assume_role_policy, "pods.eks.amazonaws.com")
    error_message = "The dataset-reader role must trust the EKS Pod Identity service principal pods.eks.amazonaws.com."
  }

  assert {
    condition     = strcontains(aws_iam_role.dataset_reader.assume_role_policy, "sts:TagSession")
    error_message = "The dataset-reader role trust policy must allow sts:TagSession (required by EKS Pod Identity)."
  }

  assert {
    condition     = !strcontains(aws_iam_role.dataset_reader.assume_role_policy, "ec2.amazonaws.com")
    error_message = "The dataset-reader role must NOT trust ec2.amazonaws.com — it is assumed by the pipeline pod via Pod Identity, not by EC2 instances."
  }

  assert {
    condition     = aws_eks_pod_identity_association.dataset_reader.namespace == "mlops" && aws_eks_pod_identity_association.dataset_reader.service_account == "mlops-pipeline"
    error_message = "The dataset-reader Pod Identity association must bind the mlops/mlops-pipeline service account (the pipeline's own SA)."
  }
}

# The S3 access is delivered as a dedicated INLINE policy on the dataset-reader
# role, and it is READ-ONLY: it grants GetObject/ListBucket but NEVER PutObject or
# DeleteObject. The rendered policy JSON is known at plan (no computed values), so
# the read-only content is asserted directly here — this is the least-privilege
# heart of requirement 5.
run "dataset_policy_is_read_only_and_scoped" {
  command = plan

  assert {
    condition     = aws_iam_role_policy.dataset_reader.name == "mlops-pipeline-dev-dataset-read"
    error_message = "The dataset read policy name must be the environment-scoped <name_prefix>-dataset-read."
  }

  assert {
    condition     = strcontains(aws_iam_role_policy.dataset_reader.policy, "s3:GetObject")
    error_message = "The dataset policy must grant s3:GetObject (the read path)."
  }

  assert {
    condition     = strcontains(aws_iam_role_policy.dataset_reader.policy, "s3:ListBucket")
    error_message = "The dataset policy must grant s3:ListBucket on the bucket."
  }

  assert {
    condition     = !strcontains(aws_iam_role_policy.dataset_reader.policy, "s3:PutObject")
    error_message = "The dataset policy must NOT grant s3:PutObject — dataset access is READ-ONLY (least privilege, M-04)."
  }

  assert {
    condition     = !strcontains(aws_iam_role_policy.dataset_reader.policy, "s3:DeleteObject")
    error_message = "The dataset policy must NOT grant s3:DeleteObject — dataset access is READ-ONLY (least privilege, M-04)."
  }

  assert {
    condition     = !strcontains(aws_iam_role_policy.dataset_reader.policy, "\"s3:*\"")
    error_message = "The dataset policy must NOT grant s3:* — actions must be enumerated (least privilege)."
  }
}
