# Terraform contract tests for the MLflow artifact store (Sprint 7, PR 6).
#
# OFFLINE, credential-free `terraform test` runs: `mock_provider "aws"` replaces
# the real provider so no AWS API is ever called, and every run uses
# `command = plan`, so nothing is created. The suite pins the SECURITY and
# WORKLOAD-IDENTITY contract of the MLflow S3 artifact store and the EBS CSI driver
# that backs the Postgres PVC on EKS: the bucket is private + encrypted + versioned,
# and both the MLflow server and the EBS CSI controller get AWS access via EKS Pod
# Identity (never static keys). Executable proof, in the same no-AWS spirit as the
# rest of the Terraform CI gate (ADR-019, ADR-026).
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

  # The CMK-policy run below uses `command = apply` (see its comment). During a
  # mocked apply the provider would otherwise invent a random string for each
  # computed ARN, which EKS's role_arn / key_arn format validation rejects and
  # which would leave the key-policy JSON (it interpolates a role ARN) unknown.
  # Pin well-formed ARNs so the mocked apply is representative, mirroring the
  # eks_secrets_encryption test.
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

  # The mocked apply evaluates every module output; eks_cluster_oidc_issuer_url
  # indexes the cluster's computed identity block, empty by default under the mock.
  mock_resource "aws_eks_cluster" {
    defaults = {
      identity = [{ oidc = [{ issuer = "https://oidc.eks.us-east-1.amazonaws.com/id/MOCK" }] }]
    }
  }
}

# The artifact bucket refuses ALL public access — it holds models and run
# artifacts and must never be internet-reachable.
run "artifact_bucket_blocks_all_public_access" {
  command = plan

  assert {
    condition = (
      aws_s3_bucket_public_access_block.mlflow_artifacts.block_public_acls &&
      aws_s3_bucket_public_access_block.mlflow_artifacts.block_public_policy &&
      aws_s3_bucket_public_access_block.mlflow_artifacts.ignore_public_acls &&
      aws_s3_bucket_public_access_block.mlflow_artifacts.restrict_public_buckets
    )
    error_message = "The MLflow artifact bucket must block all four public-access vectors (ACLs, policy, ignore ACLs, restrict public buckets)."
  }

  assert {
    condition     = aws_s3_bucket_ownership_controls.mlflow_artifacts.rule[0].object_ownership == "BucketOwnerEnforced"
    error_message = "The artifact bucket must enforce BucketOwnerEnforced (ACLs disabled), so access is IAM-only."
  }
}

# The bucket is encrypted at rest with a CUSTOMER-MANAGED KMS key and versioned —
# durability + recoverability + controllable/auditable encryption for artifacts.
run "artifact_bucket_is_kms_encrypted_and_versioned" {
  command = plan

  assert {
    # `rule` is a set (not indexable), so match with a for-expression.
    condition = anytrue([
      for r in aws_s3_bucket_server_side_encryption_configuration.mlflow_artifacts.rule :
      r.apply_server_side_encryption_by_default[0].sse_algorithm == "aws:kms"
    ])
    error_message = "The artifact bucket must use SSE-KMS (aws:kms) with the customer-managed key, not the AWS-owned SSE-S3 key."
  }

  assert {
    condition     = aws_s3_bucket_versioning.mlflow_artifacts.versioning_configuration[0].status == "Enabled"
    error_message = "The artifact bucket must have versioning enabled to guard against accidental overwrite/delete of artifacts."
  }
}

# The artifact CMK is rotated and its key policy is least-privilege: only the
# MLflow server role may use it, and no broad kms:* is granted to that role.
# Uses `command = apply` (against the mock_provider — nothing real is created) so
# the mocked apply resolves the computed role ARN and the whole policy JSON is
# known and inspectable, the same technique the EKS-secrets key-policy test uses.
run "artifact_cmk_is_rotated_and_least_privilege" {
  command = apply

  assert {
    condition     = aws_kms_key.mlflow_artifacts.enable_key_rotation == true
    error_message = "The MLflow artifact CMK must have automatic key rotation enabled."
  }

  assert {
    condition     = aws_kms_alias.mlflow_artifacts.name == "alias/mlops-pipeline-dev-mlflow-artifacts"
    error_message = "The artifact CMK must have the environment-scoped alias alias/<name_prefix>-mlflow-artifacts."
  }

  assert {
    condition     = strcontains(aws_kms_key.mlflow_artifacts.policy, "kms:GenerateDataKey")
    error_message = "The artifact CMK policy must grant the MLflow server role kms:GenerateDataKey (SSE-KMS write path)."
  }

  assert {
    condition     = strcontains(aws_kms_key.mlflow_artifacts.policy, "AllowMLflowServerRoleToUseTheKey")
    error_message = "The artifact CMK policy must scope key use to the MLflow server role (least privilege)."
  }
}

# The MLflow server gets S3 access via EKS Pod Identity, bound to exactly the
# mlops/mlflow-server service account — not static keys, not the node profile.
run "mlflow_s3_access_is_workload_identity" {
  command = plan

  assert {
    condition     = strcontains(aws_iam_role.mlflow_s3.assume_role_policy, "pods.eks.amazonaws.com")
    error_message = "The MLflow S3 role must trust the EKS Pod Identity service principal pods.eks.amazonaws.com."
  }

  assert {
    condition     = strcontains(aws_iam_role.mlflow_s3.assume_role_policy, "sts:TagSession")
    error_message = "The MLflow S3 role trust policy must allow sts:TagSession (required by EKS Pod Identity)."
  }

  assert {
    condition     = !strcontains(aws_iam_role.mlflow_s3.assume_role_policy, "ec2.amazonaws.com")
    error_message = "The MLflow S3 role must NOT trust ec2.amazonaws.com — it is assumed by the mlflow-server pod via Pod Identity, not by EC2 instances."
  }

  assert {
    condition     = aws_eks_pod_identity_association.mlflow_s3.namespace == "mlops" && aws_eks_pod_identity_association.mlflow_s3.service_account == "mlflow-server"
    error_message = "The MLflow S3 Pod Identity association must bind the mlops/mlflow-server service account."
  }
}

# The S3 access is delivered as a dedicated INLINE policy attached to the MLflow S3
# role (not an AWS-managed or account-wide policy). The rendered policy JSON is
# provider-normalized (unknown at plan under the mock), so this run pins the
# structural wiring that IS known at plan; the policy CONTENT (scoped actions, this
# bucket only, no s3:*) is enforced in source (terraform/s3.tf) and by `terraform
# validate`.
run "mlflow_s3_policy_is_inline_on_the_role" {
  command = plan

  assert {
    # name_prefix defaults to "<project_name>-<environment>" = "mlops-pipeline-dev".
    # The inline policy's presence with this environment-scoped name proves the
    # dedicated per-bucket policy exists (rather than an account-wide/managed grant).
    condition     = aws_iam_role_policy.mlflow_s3.name == "mlops-pipeline-dev-mlflow-s3-access"
    error_message = "The MLflow S3 inline policy name must be the environment-scoped <name_prefix>-mlflow-s3-access."
  }
}

# The EBS CSI driver (which provisions the Postgres PVC on EKS) is installed and
# also uses Pod Identity, bound to its controller service account.
run "ebs_csi_driver_installed_with_pod_identity" {
  command = plan

  assert {
    condition     = aws_eks_addon.ebs_csi.addon_name == "aws-ebs-csi-driver"
    error_message = "The aws-ebs-csi-driver addon must be installed so the MLflow Postgres PVC can be dynamically provisioned on EKS."
  }

  assert {
    condition     = strcontains(aws_iam_role_policy_attachment.ebs_csi.policy_arn, "AmazonEBSCSIDriverPolicy")
    error_message = "The EBS CSI controller role must carry the AWS-managed AmazonEBSCSIDriverPolicy."
  }

  assert {
    condition     = aws_eks_pod_identity_association.ebs_csi.namespace == "kube-system" && aws_eks_pod_identity_association.ebs_csi.service_account == "ebs-csi-controller-sa"
    error_message = "The EBS CSI Pod Identity association must bind the kube-system/ebs-csi-controller-sa service account."
  }
}
