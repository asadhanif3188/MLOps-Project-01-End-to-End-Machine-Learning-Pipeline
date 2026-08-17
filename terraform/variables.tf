# Input variables for the AWS foundation.
#
# Every variable has a description (surfaced by `terraform` tooling and docs) and
# a safe, non-secret default so the configuration is valid out of the box. None
# of these carry credentials — secrets never travel through Terraform variables
# in this project (see terraform/README.md § Security).

variable "aws_region" {
  description = "AWS region in which infrastructure is provisioned (e.g. \"us-east-1\"). Drives the provider and all regional resources added in later PRs."
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}(-[a-z]+)+-[0-9]$", var.aws_region))
    error_message = "aws_region must be a valid AWS region identifier, e.g. \"us-east-1\" or \"eu-west-2\"."
  }
}

variable "project_name" {
  description = "Short project identifier used as the naming/tagging prefix for all resources. Lowercase letters, digits and hyphens only."
  type        = string
  default     = "mlops-pipeline"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,30}[a-z0-9]$", var.project_name))
    error_message = "project_name must be 3-32 chars: lowercase alphanumeric and hyphens, starting with a letter and not ending in a hyphen."
  }
}

variable "environment" {
  description = "Deployment environment name. Participates in resource naming and tagging so dev/staging/prod resources are unambiguous."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "owner" {
  description = "Owner/steward recorded in the \"Owner\" tag for cost attribution and accountability. Not a secret."
  type        = string
  default     = "asadhanif3188"
}

variable "repository_url" {
  description = "Source repository URL recorded in the \"Repository\" tag so provisioned resources trace back to the code that created them."
  type        = string
  default     = "https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline"
}

variable "additional_tags" {
  description = "Optional extra tags merged into (and able to extend, not override the reserved keys of) the common tag set."
  type        = map(string)
  default     = {}
}

# --- Container registry / ECR (Sprint 7, PR 1) --------------------------------
# The private Amazon ECR repository that stores the workload image. Managed by
# Terraform (closing Sprint 6 finding H-01) so it shares the lifecycle, tagging,
# and teardown of every other resource. See ADR-021 and terraform/README.md
# § Container registry for the design rationale.

variable "ecr_repository_name" {
  description = "Name of the ECR repository that stores the workload image. Left null to default to project_name (\"mlops-pipeline\"), which keeps it in lock-step with the image reference committed in k8s/overlays/aws. Not environment-scoped: one artifact registry per project."
  type        = string
  default     = null

  validation {
    condition     = var.ecr_repository_name == null ? true : can(regex("^[a-z0-9]([a-z0-9._/-]{0,254}[a-z0-9])?$", var.ecr_repository_name))
    error_message = "ecr_repository_name must be a valid ECR repository name (2-256 chars: lowercase alphanumerics with '.', '_', '-', '/' separators, starting and ending alphanumeric) or null to default to project_name."
  }
}

variable "ecr_max_image_count" {
  description = "Maximum number of images the ECR lifecycle policy retains; older images beyond this count are expired automatically so registry storage cannot grow unbounded across repeated validation pushes."
  type        = number
  default     = 10

  validation {
    condition     = var.ecr_max_image_count >= 1 && var.ecr_max_image_count <= 100
    error_message = "ecr_max_image_count must be between 1 and 100."
  }
}

# --- Network (Sprint 6, PR 2) -------------------------------------------------
# The VPC that hosts the EKS cluster added in a later PR. Everything below is
# variable-driven so the same configuration works for a small dev VPC or a
# larger footprint without editing resource definitions. See ADR-015 and
# terraform/README.md § Network architecture for the design rationale.

variable "vpc_cidr" {
  description = "IPv4 CIDR block for the VPC. A /16 gives ample room for the /24 per-AZ subnets this configuration derives; the prefix is capped at /20 so derived subnets stay usefully sized for EKS ENIs/pods."
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0)) && tonumber(split("/", var.vpc_cidr)[1]) >= 16 && tonumber(split("/", var.vpc_cidr)[1]) <= 20
    error_message = "vpc_cidr must be a valid IPv4 CIDR with a prefix between /16 and /20 (e.g. \"10.0.0.0/16\")."
  }
}

variable "az_count" {
  description = "Number of Availability Zones to spread subnets across. EKS requires subnets in at least two AZs; AZ names are discovered at plan time, never hard-coded. Kept small (2) by default for cost; 3 is supported for broader spread."
  type        = number
  default     = 2

  validation {
    condition     = var.az_count >= 2 && var.az_count <= 3
    error_message = "az_count must be 2 or 3 (EKS needs at least two AZs; three is the practical upper bound for this portfolio scope)."
  }
}

variable "enable_nat_gateway" {
  description = "Whether to create NAT gateway(s) so private-subnet nodes have outbound internet egress (image pulls, dataset fetch, pip, package indexes). Required for the EKS worker nodes, which live in the private subnets. Disable only if nodes are placed in public subnets instead."
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "When NAT is enabled, use one shared NAT gateway for all AZs (true, cost-optimized for this short-lived portfolio environment) instead of one per AZ (false, AZ-fault-tolerant but ~1x NAT cost per additional AZ)."
  type        = bool
  default     = true
}

# --- EKS platform (Sprint 6, PR 4) --------------------------------------------
# The managed Kubernetes cluster and node group that run the MLOps workload.
# Everything is variable-driven so the cluster stays small and cost-conscious by
# default but can be resized without editing resource definitions. See ADR-017
# and terraform/README.md § EKS platform for the design rationale.

variable "kubernetes_version" {
  description = "Kubernetes minor version for the EKS control plane and managed node group, pinned explicitly for reproducibility (e.g. \"1.35\"). Must be a version AWS EKS currently supports in the target region; EKS manages the patch version."
  type        = string
  default     = "1.35"

  validation {
    condition     = can(regex("^1\\.(2[5-9]|3[0-9])$", var.kubernetes_version))
    error_message = "kubernetes_version must be a supported EKS minor version of the form \"1.NN\" (e.g. \"1.35\")."
  }
}

variable "node_instance_types" {
  description = "EC2 instance types for the managed node group. A single small general-purpose type (t3.medium: 2 vCPU / 4 GiB) is the cost-conscious default — enough for the batch Job plus EKS system pods, no GPUs."
  type        = list(string)
  default     = ["t3.medium"]

  validation {
    condition     = length(var.node_instance_types) > 0
    error_message = "node_instance_types must list at least one EC2 instance type."
  }
}

variable "node_capacity_type" {
  description = "Managed node group purchasing option. ON_DEMAND (default) gives predictable capacity for a short validation run; SPOT is cheaper but can be interrupted."
  type        = string
  default     = "ON_DEMAND"

  validation {
    condition     = contains(["ON_DEMAND", "SPOT"], var.node_capacity_type)
    error_message = "node_capacity_type must be either \"ON_DEMAND\" or \"SPOT\"."
  }
}

variable "node_desired_size" {
  description = "Desired number of worker nodes. Two small nodes comfortably host the single batch Job plus EKS system pods (CoreDNS, kube-proxy, VPC CNI) with room to schedule."
  type        = number
  default     = 2

  validation {
    condition     = var.node_desired_size >= 1 && var.node_desired_size <= 4
    error_message = "node_desired_size must be between 1 and 4 for this cost-conscious validation cluster."
  }
}

variable "node_min_size" {
  description = "Minimum node count. Equal to the desired size by default: there is no Cluster Autoscaler in this PR, so the group is effectively fixed-size (no autoscaling)."
  type        = number
  default     = 2

  validation {
    condition     = var.node_min_size >= 1 && var.node_min_size <= 4
    error_message = "node_min_size must be between 1 and 4."
  }
}

variable "node_max_size" {
  description = "Maximum node count. Kept equal to the desired size so the group does not scale on its own (no autoscaler is installed); raise only if a future workload needs headroom for rolling updates."
  type        = number
  default     = 2

  validation {
    condition     = var.node_max_size >= 1 && var.node_max_size <= 4
    error_message = "node_max_size must be between 1 and 4."
  }
}

variable "node_disk_size" {
  description = "EBS root volume size (GiB) per worker node. 20 GiB is enough for the container images and ephemeral pipeline data this validation workload uses."
  type        = number
  default     = 20

  validation {
    condition     = var.node_disk_size >= 20 && var.node_disk_size <= 100
    error_message = "node_disk_size must be between 20 and 100 GiB for this validation cluster."
  }
}

variable "cluster_endpoint_public_access" {
  description = "Whether the EKS API server is reachable from the public internet. SECURE DEFAULT: false — the API is private-only out of the box (closes finding H-02). Enabling it is an explicit opt-in and REQUIRES a scoped cluster_endpoint_public_access_cidrs allow-list (an empty list would let EKS fall back to 0.0.0.0/0, which is rejected). Reaching a private-only endpoint needs in-VPC access (bastion/VPN/SSM/in-VPC runner) — see ADR-022 and terraform/README.md § EKS platform."
  type        = bool
  default     = false
}

variable "cluster_endpoint_private_access" {
  description = "Whether the EKS API server is reachable privately from within the VPC. Enabled by default so in-VPC nodes/tools reach the API over private networking; it is the primary (and, by default, only) access path. Disabling it while the public endpoint is also off is rejected — the API server would be unreachable."
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDR blocks allowed to reach the public EKS API endpoint when public access is opted into. SECURE DEFAULT: [] (empty) — with the private-only default there is no public exposure. If you enable public access, set this to your operator IP/CIDR (e.g. [\"203.0.113.4/32\"]); an unrestricted range (0.0.0.0/0 or any /0) is rejected by validation and must never be used."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for c in var.cluster_endpoint_public_access_cidrs : can(cidrhost(c, 0))])
    error_message = "Every entry in cluster_endpoint_public_access_cidrs must be a valid IPv4/IPv6 CIDR block (e.g. \"203.0.113.4/32\")."
  }

  validation {
    condition = alltrue([
      for c in var.cluster_endpoint_public_access_cidrs :
      can(cidrhost(c, 0)) ? tonumber(split("/", c)[1]) >= 1 : true
    ])
    error_message = "cluster_endpoint_public_access_cidrs must not contain an unrestricted range (a /0 such as 0.0.0.0/0). Scope public API access to specific operator IP(s)/CIDR(s)."
  }
}

variable "cluster_enabled_log_types" {
  description = "EKS control-plane log types shipped to CloudWatch Logs. Defaults to the security-relevant subset (api, audit, authenticator) for auditability; set to [] to eliminate the small CloudWatch Logs cost entirely."
  type        = list(string)
  default     = ["api", "audit", "authenticator"]

  validation {
    condition = alltrue([
      for t in var.cluster_enabled_log_types :
      contains(["api", "audit", "authenticator", "controllerManager", "scheduler"], t)
    ])
    error_message = "cluster_enabled_log_types entries must be from: api, audit, authenticator, controllerManager, scheduler."
  }
}

# --- EKS Secret encryption / KMS (Sprint 7, PR 5) -----------------------------
# The customer-managed KMS key that envelope-encrypts Kubernetes Secrets stored in
# the cluster (closes finding M-02). Encryption itself is unconditional — there is
# deliberately NO "enable" toggle, because a switch to turn a security control off
# is exactly what M-02 is about. See ADR-025 and terraform/README.md § Secrets
# encryption. Only the key's deletion window is tunable.

variable "kms_key_deletion_window_days" {
  description = "Waiting period (in days) before the EKS-secrets KMS key is permanently deleted after `terraform destroy` schedules its deletion. Defaults to the 7-day minimum because this is a short-lived, single-operator validation cluster (ADR-020): a torn-down environment should not leave a pending-deletion key (and its charge) lingering. A persistent/production key would use a longer window as an accidental-deletion safety net. AWS permits 7-30."
  type        = number
  default     = 7

  validation {
    condition     = var.kms_key_deletion_window_days >= 7 && var.kms_key_deletion_window_days <= 30
    error_message = "kms_key_deletion_window_days must be between 7 and 30 (the range AWS KMS allows)."
  }
}

# --- EKS access management (Sprint 7, PR 3) -----------------------------------
# Explicit, access-entry-based cluster access, replacing the old
# "whoever ran apply becomes cluster-admin" bootstrap (closes finding H-03).
# Access is now: AWS identity -> EKS access entry -> scoped EKS access policy ->
# Kubernetes permissions, all declared here and never tied to the creating
# principal. Personal ARNs are NEVER committed: they come from a git-ignored
# terraform.tfvars. See ADR-023 and terraform/README.md § EKS access management.

variable "cluster_authentication_mode" {
  description = "EKS cluster authentication mode. SECURE DEFAULT: \"API\" — access is granted ONLY through EKS access entries (no aws-auth ConfigMap path), which is the explicit model this project standardises on. \"API_AND_CONFIG_MAP\" is allowed for migrating a cluster that still needs the legacy ConfigMap. \"CONFIG_MAP\" (aws-auth only, access entries ignored) is rejected: it bypasses the access-entry model H-03 mandates."
  type        = string
  default     = "API"

  validation {
    condition     = contains(["API", "API_AND_CONFIG_MAP"], var.cluster_authentication_mode)
    error_message = "cluster_authentication_mode must be \"API\" (recommended; access entries only) or \"API_AND_CONFIG_MAP\". \"CONFIG_MAP\" (aws-auth only) is not permitted because it bypasses EKS access entries."
  }
}

variable "cluster_bootstrap_creator_admin_permissions" {
  description = "Whether EKS automatically grants the IAM principal that CREATED the cluster implicit cluster-admin. SECURE DEFAULT: false (closes finding H-03) — access must instead be declared explicitly via cluster_access_entries. This is a TRIPWIRE variable: the validation below REJECTS true, so the old insecure bootstrap setting cannot be reintroduced (even deliberately) without failing plan/CI. Grant admins explicitly and scoped rather than implicitly to whoever happened to run apply."
  type        = bool
  default     = false

  validation {
    condition     = var.cluster_bootstrap_creator_admin_permissions == false
    error_message = "SECURITY (H-03): cluster_bootstrap_creator_admin_permissions must be false. Automatic cluster-creator cluster-admin is the insecure bootstrap this project removed — grant access explicitly via cluster_access_entries with a scoped EKS access policy instead."
  }
}

variable "cluster_access_entries" {
  description = <<-EOT
    Explicit EKS access entries: the ONLY way cluster access is granted (H-03). A
    map keyed by a short, stable operator-chosen label (used only for resource
    addressing and the Name tag), each value declaring one IAM principal and the
    scoped EKS managed access policy it receives.

    NEVER commit personal ARNs: leave this empty in the repo and set it in a
    git-ignored terraform.tfvars (see terraform.tfvars.example). Fields per entry:
      - principal_arn : ARN of an IAM ROLE (preferred) or user to grant access to.
      - policy        : short name of an AWS-managed EKS access policy. Narrowest
                        practical first — AmazonEKSViewPolicy (read-only),
                        AmazonEKSEditPolicy, AmazonEKSAdminPolicy (default; scoped
                        admin, NOT system:masters). AmazonEKSClusterAdminPolicy
                        (full cluster-admin) is allowed but discouraged — do not
                        use it merely for convenience.
      - access_scope  : "cluster" (default) or "namespace".
      - namespaces    : required non-empty list when access_scope = "namespace";
                        ignored for "cluster" scope.

    The managed node group's own access entry is created automatically by EKS and
    must NOT be listed here.
  EOT

  type = map(object({
    principal_arn = string
    policy        = optional(string, "AmazonEKSAdminPolicy")
    access_scope  = optional(string, "cluster")
    namespaces    = optional(list(string), [])
  }))
  default = {}

  validation {
    condition = alltrue([
      for k, e in var.cluster_access_entries :
      can(regex("^arn:aws[a-z-]*:iam::[0-9]{12}:(role|user)/.+$", e.principal_arn))
    ])
    error_message = "Every cluster_access_entries principal_arn must be an IAM role or user ARN (e.g. \"arn:aws:iam::<account-id>:role/<name>\"). Assumed-role session ARNs and non-IAM ARNs are not valid access-entry principals."
  }

  validation {
    condition = alltrue([
      for k, e in var.cluster_access_entries :
      contains(["AmazonEKSViewPolicy", "AmazonEKSEditPolicy", "AmazonEKSAdminPolicy", "AmazonEKSAdminViewPolicy", "AmazonEKSClusterAdminPolicy"], e.policy)
    ])
    error_message = "Every cluster_access_entries policy must be one of the AWS-managed EKS access policies: AmazonEKSViewPolicy, AmazonEKSEditPolicy, AmazonEKSAdminPolicy, AmazonEKSAdminViewPolicy, AmazonEKSClusterAdminPolicy."
  }

  validation {
    condition = alltrue([
      for k, e in var.cluster_access_entries :
      contains(["cluster", "namespace"], e.access_scope)
    ])
    error_message = "Every cluster_access_entries access_scope must be either \"cluster\" or \"namespace\"."
  }

  validation {
    condition = alltrue([
      for k, e in var.cluster_access_entries :
      e.access_scope == "cluster" || length(e.namespaces) > 0
    ])
    error_message = "A namespace-scoped access entry (access_scope = \"namespace\") must list at least one namespace in `namespaces`."
  }
}
