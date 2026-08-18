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

  mock_data "aws_availability_zones" {
    defaults = {
      names = ["us-east-1a", "us-east-1b", "us-east-1c"]
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

# The bucket is encrypted at rest and versioned — durability + recoverability for
# experiment artifacts.
run "artifact_bucket_is_encrypted_and_versioned" {
  command = plan

  assert {
    # `rule` is a set (not indexable), so match with a for-expression.
    condition = anytrue([
      for r in aws_s3_bucket_server_side_encryption_configuration.mlflow_artifacts.rule :
      r.apply_server_side_encryption_by_default[0].sse_algorithm == "AES256"
    ])
    error_message = "The artifact bucket must have default server-side encryption (AES256/SSE-S3) enabled."
  }

  assert {
    condition     = aws_s3_bucket_versioning.mlflow_artifacts.versioning_configuration[0].status == "Enabled"
    error_message = "The artifact bucket must have versioning enabled to guard against accidental overwrite/delete of artifacts."
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
