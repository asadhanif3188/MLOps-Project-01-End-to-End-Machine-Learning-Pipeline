# S3 dataset store for the ML pipeline's RAW INPUT DATA (Sprint 7, PR 8 — closes M-04).
#
# Before this PR the runtime dataset was delivered to the pipeline pod through a
# Kubernetes ConfigMap (an out-of-band `mlops-pipeline-dataset` created from the
# git-ignored CSV). That was always documented as a LOCAL-VALIDATION mechanism,
# never production storage: a ConfigMap caps at ~1 MiB, is etcd-backed, and is the
# wrong carrier for a dataset. Finding M-04 called for a professional cloud data
# path. This file provisions the AWS half of that path:
#
#     S3 (this bucket)  ->  EKS Pod Identity  ->  init-container retrieval
#                        ->  /app/data/raw     ->  DVC pipeline
#
# A private, CMK-encrypted, versioned S3 bucket holds the dataset object; a
# dedicated, READ-ONLY IAM role is assumed by the pipeline's OWN service account
# (mlops/mlops-pipeline) via EKS Pod Identity, so the `fetch-dataset` init
# container reads the object with a short-lived, pod-scoped identity and there are
# NO static AWS keys anywhere (requirement: workload identity, no access keys).
# Locally the same retrieval path is exercised against MinIO (the local overlay);
# this bucket is the production expression of the same design.
#
# What this file deliberately does NOT do: it does NOT upload the dataset. The
# object is placed out-of-band by the operator (see the proof runbook and
# k8s/README.md § "Dataset"), exactly as credentials are — so the dataset is never
# committed to Git and never baked into the image (requirements 2, 3). Terraform
# owns the empty, secured bucket; the data itself is content the operator seeds.
#
# Design of record: ADR-027. Mirrors the MLflow artifact store (s3.tf, ADR-026)
# so the two object stores are consistent in security posture and identity model.
# Common Project/Environment/Owner tags come from the provider default_tags
# (providers.tf); only the resource-specific Name tag is set below.

# Globally-unique bucket name. Like the MLflow bucket, the account ID is appended
# to the environment-scoped prefix so the name is unique across the global S3
# namespace without embedding anything secret in the k8s overlay (the overlay
# references it by the value of `terraform output dataset_bucket_name`).
locals {
  dataset_bucket_name = "${local.name_prefix}-datasets-${data.aws_caller_identity.current.account_id}"

  # The canonical object key the pipeline reads. Versioned in the key PATH (not
  # only via S3 object versioning) so the dataset's identity/version is explicit
  # and legible in both the manifest (DATASET_S3_URI) and this file — bumping the
  # dataset means writing a new `.../v2/...` key, a deliberate, reviewable change.
  dataset_object_key = "pima-indians-diabetes/v1/data.csv"
}

resource "aws_s3_bucket" "datasets" {
  bucket = local.dataset_bucket_name

  # Matches the MLflow bucket and the ECR repo: let `terraform destroy` remove the
  # bucket even if it still holds the dataset object, appropriate for the
  # short-lived, single-operator validation environment (ADR-020). A persistent
  # dataset lake would set this false.
  force_destroy = true

  tags = {
    Name = local.dataset_bucket_name
  }
}

# Block ALL public access, unconditionally. The dataset is private input data and
# must never be internet-reachable; the four settings together reject any public
# ACL or bucket policy even if one is later added by mistake.
resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Disable ACLs entirely (BucketOwnerEnforced) — access is IAM-only, removing a
# whole class of accidental-public-object mistakes. Same modern default as the
# artifact bucket.
resource "aws_s3_bucket_ownership_controls" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Versioning ON: the dataset is a first-class, reproducibility-critical input.
# Object versioning guards against accidental overwrite/delete and — combined with
# the versioned key path — gives point-in-time recoverability of the exact bytes a
# past run consumed.
resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Dedicated customer-managed KMS key (CMK) for the dataset bucket. Consistent with
# the artifact bucket (M-02/ADR-026) and the project's remediate-don't-suppress
# stance: encrypt with a key we control (rotation, CloudTrail-auditable use,
# revocable) rather than the AWS-owned SSE-S3 key. A SEPARATE key from the MLflow
# artifact CMK so the two data domains have independent rotation and blast radius.
resource "aws_kms_key" "datasets" {
  description             = "Customer-managed CMK for the ML pipeline dataset S3 bucket (ADR-027)."
  deletion_window_in_days = var.kms_key_deletion_window_days
  enable_key_rotation     = true

  # Key policy — least-privilege, mirroring s3.tf/kms.tf:
  #   1. EnableIAMRootAdministration — the AWS-canonical anti-lockout statement
  #      delegating key administration to same-account IAM (the one justified kms:*).
  #      This is also what lets the operator (an admin principal) UPLOAD the dataset
  #      object with SSE-KMS via the aws CLI.
  #   2. AllowDatasetReaderRoleToDecrypt — grants ONLY the pipeline's dataset-reader
  #      Pod Identity role the READ data-key operation SSE-KMS needs (Decrypt), plus
  #      DescribeKey. NO GenerateDataKey/Encrypt: the pipeline only READS the
  #      dataset, so it never needs the write-path key operations — tighter than the
  #      MLflow server role, which does write artifacts.
  policy = jsonencode({
    Version = "2012-10-17"
    Id      = "${local.name_prefix}-datasets-key-policy"
    Statement = [
      {
        Sid       = "EnableIAMRootAdministration"
        Effect    = "Allow"
        Principal = { AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid       = "AllowDatasetReaderRoleToDecrypt"
        Effect    = "Allow"
        Principal = { AWS = aws_iam_role.dataset_reader.arn }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        Resource = "*"
      },
    ]
  })

  tags = {
    Name = "${local.name_prefix}-datasets"
  }
}

resource "aws_kms_alias" "datasets" {
  name          = "alias/${local.name_prefix}-datasets"
  target_key_id = aws_kms_key.datasets.key_id
}

# Encrypt every object at rest with the CMK above (SSE-KMS). bucket_key_enabled
# uses S3 Bucket Keys to cut KMS request costs.
resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.datasets.arn
    }
    bucket_key_enabled = true
  }
}

# Lifecycle — cost control for the VERSIONED bucket (Sprint 7 § "Cost Controls").
# Versioning keeps every prior object version forever by default, which would grow
# storage cost without bound as the dataset is re-uploaded. This bounds that:
#   * noncurrent_version_expiration — delete superseded (noncurrent) versions after
#     30 days. The CURRENT version is always retained; only stale history is reaped,
#     preserving short-term point-in-time recovery without unbounded accumulation.
#   * abort_incomplete_multipart_upload — reap failed multipart uploads after 7 days
#     so half-uploaded parts never linger as invisible, billable storage.
# Consistent with the project's cost-control stance (ADR-020) and the ECR retention
# policy. depends_on the versioning resource: a lifecycle config referencing
# noncurrent versions must be applied after versioning is enabled.
resource "aws_s3_bucket_lifecycle_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    id     = "expire-noncurrent-and-abort-incomplete-uploads"
    status = "Enabled"

    # Apply to the whole bucket (empty prefix filter).
    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [aws_s3_bucket_versioning.datasets]
}

# --- Pipeline dataset-reader workload identity (EKS Pod Identity) ---------------
# A dedicated IAM role the PIPELINE pod assumes via EKS Pod Identity — the SAME
# mechanism the VPC CNI (ADR-024) and MLflow server (ADR-026) use. It is bound to
# exactly one service account (mlops/mlops-pipeline, the pipeline's existing SA),
# so only the pipeline pod can read the dataset bucket; no other pod, and not the
# node instance profile, can. The `fetch-dataset` init container inside the pod
# uses these pod-scoped credentials to download the dataset; the main pipeline
# container never needs them (it reads the file off the shared volume).
#
# Trust: only the EKS Pod Identity service principal (pods.eks.amazonaws.com), with
# the sts:AssumeRole + sts:TagSession actions Pod Identity requires. No EC2, human,
# or account principal is trusted — the role is unusable by anything except a pod
# the EKS Pod Identity agent has associated.
resource "aws_iam_role" "dataset_reader" {
  name        = "${local.name_prefix}-dataset-reader-role"
  description = "Pipeline dataset-reader role assumed by the mlops/mlops-pipeline service account via EKS Pod Identity; grants READ-ONLY access to the dataset bucket only."

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "pods.eks.amazonaws.com" }
        Action = [
          "sts:AssumeRole",
          "sts:TagSession",
        ]
      }
    ]
  })

  tags = {
    Name = "${local.name_prefix}-dataset-reader-role"
  }
}

# Least-privilege inline policy: exactly the S3 READ actions the retrieval needs,
# scoped to THIS one bucket and its objects — no wildcard bucket, no account-wide
# S3 grant, and crucially NO write/delete (s3:PutObject / s3:DeleteObject are
# absent). ListBucket + GetBucketLocation on the bucket ARN; GetObject on the
# object ARNs beneath it. (An inline policy, not an AWS-managed one, because no
# managed policy scopes read to a single bucket; the breadth here is minimal and
# authored.)
resource "aws_iam_role_policy" "dataset_reader" {
  name = "${local.name_prefix}-dataset-read"
  role = aws_iam_role.dataset_reader.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListDatasetBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = aws_s3_bucket.datasets.arn
      },
      {
        Sid      = "ReadDatasetObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.datasets.arn}/*"
      },
    ]
  })
}

# The association that binds (cluster, namespace mlops, service account
# mlops-pipeline) -> the dataset-reader role. This is what makes boto3 inside the
# fetch-dataset init container resolve pod-scoped credentials automatically, with
# no static keys. The eks-pod-identity-agent addon (installed in eks.tf) serves
# these credentials on-cluster.
resource "aws_eks_pod_identity_association" "dataset_reader" {
  cluster_name    = aws_eks_cluster.this.name
  namespace       = "mlops"
  service_account = "mlops-pipeline"
  role_arn        = aws_iam_role.dataset_reader.arn

  tags = {
    Name = "${local.name_prefix}-dataset-reader-pod-identity"
  }
}
