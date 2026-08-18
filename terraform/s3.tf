# S3 artifact store for the in-cluster MLflow platform (Sprint 7, PR 6).
#
# The MLflow Tracking Server (deployed via Kustomize, k8s/base/mlflow) stores its
# experiment ARTIFACTS in S3 and its METADATA in an in-cluster PostgreSQL. This
# file provisions the AWS half of the artifact store: a private, encrypted S3
# bucket, plus a dedicated IAM role the tracking server assumes via EKS Pod
# Identity — so the server reads/writes the bucket with a short-lived, pod-scoped
# identity and there are NO static AWS keys anywhere on AWS (requirement: workload
# identity where applicable). Locally, the same artifact path is exercised against
# MinIO instead (k8s/overlays/local/minio.yaml); this bucket is the production
# expression of the same design.
#
# Design of record: ADR-026. Common Project/Environment/Owner tags are applied to
# every resource automatically via the provider's default_tags (providers.tf);
# only the resource-specific Name tag is set below.

# Globally-unique bucket name. S3 bucket names share one global namespace, so the
# account ID is appended to the environment-scoped prefix to avoid collisions
# without embedding anything secret in the k8s overlay (the overlay references the
# bucket by the value of `terraform output mlflow_artifact_bucket_name`).
locals {
  mlflow_artifact_bucket_name = "${local.name_prefix}-mlflow-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = local.mlflow_artifact_bucket_name

  # Let `terraform destroy` remove the bucket even if it still holds artifacts —
  # appropriate for the short-lived, single-operator validation environment
  # (ADR-020), matching the ECR repository's force_delete. A persistent/production
  # bucket would set this false to prevent accidental data loss.
  force_destroy = true

  tags = {
    Name = local.mlflow_artifact_bucket_name
  }
}

# Block ALL public access, unconditionally. The artifact store holds models and
# run artifacts and must never be internet-reachable; the four settings together
# reject any public ACL or bucket policy even if one is later added by mistake.
resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Disable ACLs entirely (BucketOwnerEnforced) — the modern S3 best practice. All
# access is granted through the bucket owner and IAM, not object ACLs, which
# removes a whole class of accidental-public-object mistakes.
resource "aws_s3_bucket_ownership_controls" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Versioning ON: MLflow artifacts (models, plots, reports) are experiment records;
# keeping prior object versions guards against accidental overwrite/delete and
# gives the store point-in-time recoverability, reinforcing the persistence goal.
resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Encrypt every object at rest. AES256 (SSE-S3, the AWS-managed key) mirrors the
# ECR repository's encryption choice; a customer-managed KMS CMK is the documented
# hardening follow-up (as with ECR and M-02), deliberately not bundled here to keep
# this PR to the MLflow-platform boundary.
resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# --- MLflow server workload identity (EKS Pod Identity) ------------------------
# A dedicated IAM role the MLflow Tracking Server pod assumes via EKS Pod Identity
# — the SAME mechanism the VPC CNI hardening uses (ADR-024). It is bound to exactly
# one service account (mlops/mlflow-server), so only the tracking server can wield
# the S3 permissions; no other pod, and not the node instance profile, can. This is
# why the pipeline Job needs no AWS credentials: it uploads artifacts THROUGH the
# server, and only the server holds (short-lived, pod-scoped) S3 access.
#
# Trust: only the EKS Pod Identity service principal (pods.eks.amazonaws.com), with
# the sts:AssumeRole + sts:TagSession actions Pod Identity requires. No EC2, human,
# or account principal is trusted.
resource "aws_iam_role" "mlflow_s3" {
  name        = "${local.name_prefix}-mlflow-s3-role"
  description = "MLflow Tracking Server role assumed by the mlops/mlflow-server service account via EKS Pod Identity; grants scoped access to the MLflow artifact bucket only."

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
    Name = "${local.name_prefix}-mlflow-s3-role"
  }
}

# Least-privilege inline policy: exactly the S3 actions MLflow needs, scoped to
# THIS one bucket and its objects — no wildcard bucket, no account-wide S3 grant.
# ListBucket is scoped to the bucket ARN; object read/write/delete to the object
# ARNs beneath it. (An inline policy is used, not an AWS-managed one, because no
# managed policy scopes S3 access to a single bucket; the breadth here is minimal
# and authored, unlike the AWS-owned managed policies used elsewhere.)
resource "aws_iam_role_policy" "mlflow_s3" {
  name = "${local.name_prefix}-mlflow-s3-access"
  role = aws_iam_role.mlflow_s3.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListArtifactBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = aws_s3_bucket.mlflow_artifacts.arn
      },
      {
        Sid      = "ReadWriteArtifactObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.mlflow_artifacts.arn}/*"
      },
    ]
  })
}

# The association that binds (cluster, namespace mlops, service account
# mlflow-server) -> the S3 role. This is what makes boto3 inside the MLflow server
# resolve pod-scoped credentials automatically, with no static keys. The
# eks-pod-identity-agent addon (installed in eks.tf for the VPC CNI work) serves
# these credentials on-cluster.
resource "aws_eks_pod_identity_association" "mlflow_s3" {
  cluster_name    = aws_eks_cluster.this.name
  namespace       = "mlops"
  service_account = "mlflow-server"
  role_arn        = aws_iam_role.mlflow_s3.arn

  tags = {
    Name = "${local.name_prefix}-mlflow-s3-pod-identity"
  }
}
