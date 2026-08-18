# Amazon EBS CSI driver — dynamic block storage for the MLflow PostgreSQL PVC on
# EKS (Sprint 7, PR 6).
#
# Why this is needed: the MLflow metadata backend (k8s/base/mlflow/postgres.yaml)
# is a StatefulSet with a volumeClaimTemplate — a PersistentVolumeClaim from the
# cluster's default StorageClass. On a local cluster that claim is served by the
# built-in local-path provisioner; on EKS there is NO dynamic PV provisioner unless
# the EBS CSI driver is installed. Without it the Postgres PVC would stay Pending
# and the whole tracking platform would never start on AWS. This file installs the
# managed driver and gives its controller a dedicated, least-privilege identity via
# EKS Pod Identity — the SAME mechanism used for the VPC CNI (ADR-024) and the
# MLflow S3 role (s3.tf), keeping AWS access on workload identity, not static keys.
#
# On EKS the driver serves the account's default `gp2` StorageClass (CSI-migrated),
# so the unset storageClassName in the Postgres/StatefulSet claim resolves to an
# EBS-backed volume that survives pod recreation — the AWS expression of the
# persistence guarantee. A dedicated gp3 StorageClass can be added later if a
# specific class is wanted. Design of record: ADR-026.

# Dedicated role for the EBS CSI controller, assumed ONLY by the
# ebs-csi-controller-sa service account (kube-system) via EKS Pod Identity.
resource "aws_iam_role" "ebs_csi" {
  name        = "${local.name_prefix}-ebs-csi-role"
  description = "Amazon EBS CSI driver controller role assumed by the kube-system/ebs-csi-controller-sa service account via EKS Pod Identity; provisions EBS volumes for PVCs (e.g. the MLflow Postgres claim)."

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
    Name = "${local.name_prefix}-ebs-csi-role"
  }
}

# The AWS-managed policy AWS documents for the EBS CSI driver controller (create/
# attach/detach/delete volumes and snapshots). AWS-owned and AWS-maintained; this
# attaches it, it does not author the permissions.
resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

# Bind the controller's service account to the role. The service-account name
# (ebs-csi-controller-sa) is the fixed name the managed addon creates.
resource "aws_eks_pod_identity_association" "ebs_csi" {
  cluster_name    = aws_eks_cluster.this.name
  namespace       = "kube-system"
  service_account = "ebs-csi-controller-sa"
  role_arn        = aws_iam_role.ebs_csi.arn

  tags = {
    Name = "${local.name_prefix}-ebs-csi-pod-identity"
  }

  # The role must carry the CSI policy before the controller assumes it.
  depends_on = [aws_iam_role_policy_attachment.ebs_csi]
}

# The managed EBS CSI driver addon. Like the core addons (eks.tf) addon_version is
# omitted so EKS installs the default version for the pinned cluster version.
# Depends on the node group (the controller schedules on nodes) and on the Pod
# Identity association (so the controller has credentials as soon as it starts).
resource "aws_eks_addon" "ebs_csi" {
  cluster_name = aws_eks_cluster.this.name
  addon_name   = "aws-ebs-csi-driver"

  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  tags = {
    Name = "${local.name_prefix}-addon-aws-ebs-csi-driver"
  }

  depends_on = [
    aws_eks_node_group.this,
    aws_eks_pod_identity_association.ebs_csi,
  ]
}
