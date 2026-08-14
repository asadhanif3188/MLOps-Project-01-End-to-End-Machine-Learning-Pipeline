# ADR-015: AWS Network Architecture (VPC, Subnets, AZs, NAT)

- **Status:** Accepted (design)
- **Date:** 2026-08-14
- **Deciders:** Asad Hanif
- **Related:** [`terraform/network.tf`](../../terraform/network.tf),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-014 (Terraform Architecture & Foundation)](ADR-014-terraform-architecture.md),
  [Sprint 6 plan](../../Sprint-06-Terraform-Cloud-Platform-Foundation.md),
  [ADR-009 (Kubernetes Workload Model)](ADR-009-kubernetes-workload-model.md)

> **Scope note.** This ADR ratifies the *AWS network foundation* delivered in
> Sprint 6, PR 2: the VPC, its public/private subnets across Availability Zones,
> routing, and the NAT/Internet Gateway topology — the minimum network an EKS
> cluster needs. It provisions **no EKS, IAM, or application resources**; those
> arrive in later PRs (IAM → EKS → workload). The AWS/EKS platform *choice*
> (why AWS, why EKS, node sizing) and the infrastructure/workload separation get
> their own records with the EKS PR. This record covers only the network shape
> and why it is built this way.

## Context

[ADR-014](ADR-014-terraform-architecture.md) established the resource-free
Terraform foundation (versions, provider, naming, tagging, state). PR 2 lays the
first real infrastructure: the network that the EKS cluster in a later PR will
sit inside. A VPC cannot be meaningfully retrofitted under a running cluster, so
the subnet/AZ/routing shape has to be right before EKS is provisioned.

The requirements come from two directions:

- **What EKS needs.** A managed EKS control plane requires subnets in **at least
  two Availability Zones**. Worker nodes need a place to run with reliable
  **outbound** internet access (to pull container images, the dataset, Python
  packages, and to reach AWS APIs). Subnets intended for load balancers must
  carry specific discovery tags for the Kubernetes AWS integration to find them.
- **What the workload needs.** The workload is the existing Sprint 5 batch
  **`Job`** ([ADR-009](ADR-009-kubernetes-workload-model.md)) — it has *no*
  inbound surface: no `Service`, no `Ingress`, no `LoadBalancer`, no listening
  port. So the network needs **egress, not ingress**. There is no requirement to
  expose anything to the internet.

Constraints from the sprint ([Sprint 6 plan](../../Sprint-06-Terraform-Cloud-Platform-Foundation.md)):
the environment is **small, short-lived, single-operator, and destroyed after
evidence capture**, and cost control is treated as an engineering requirement.
"Avoid unnecessary network complexity" is an explicit acceptance criterion. NAT
gateways bill per hour plus per-GB, so they are the decision most worth being
deliberate about.

## Decision

Provision a **single VPC** (`10.0.0.0/16` by default) with a **public and a
private subnet in each of two Availability Zones**, an **Internet Gateway**, a
**single shared NAT gateway**, and per-tier route tables. Everything is
variable-driven.

**VPC.** One `/16` VPC with DNS support and DNS hostnames enabled (EKS needs both
for private endpoint resolution and in-cluster name resolution). The CIDR is a
variable; the prefix is validated to `/16`–`/20` so the derived subnets stay
large enough for EKS ENIs and pod IPs.

**Availability Zones — discovered, not hard-coded.** The AZ names are read at
plan time from `aws_availability_zones` (filtered to standard, non-opt-in zones)
and the first `az_count` are used. `az_count` defaults to **2** — the EKS
minimum — and is capped at 3. Nothing in the code names `us-east-1a`; changing
region or AZ count needs no edit to a resource.

**Subnet strategy — public/private per AZ.** Each AZ gets:

- a **public** subnet (`map_public_ip_on_launch = true`) that hosts the NAT
  gateway and any future public load balancers, tagged
  `kubernetes.io/role/elb = 1`; and
- a **private** subnet (no public IPs) where the **EKS worker nodes run**, tagged
  `kubernetes.io/role/internal-elb = 1`.

Per-AZ subnet CIDRs are derived from the VPC CIDR with `cidrsubnet(..., 8, ...)`
(a `/16` yields `/24`s); public subnets take the low indices and private subnets
are offset so the two ranges never overlap. Placing nodes in **private** subnets
keeps them off the public internet — a security default that costs nothing extra
and matches the least-privilege posture of the Sprint 5 workload.

**Routing.** A single public route table (default route → Internet Gateway),
associated with all public subnets. One **private route table per AZ** (default
route → NAT gateway), so that if per-AZ NAT is later enabled each AZ routes
through its own gateway without restructuring.

**NAT — single, shared, by default.** Nodes in private subnets reach the internet
outbound through NAT. A **single** NAT gateway is shared across both AZs by
default (`single_nat_gateway = true`), because a NAT gateway is the dominant
hourly cost here and one is sufficient for a short-lived single-operator
environment. `single_nat_gateway = false` gives one NAT per AZ (AZ-fault-tolerant
egress) when the trade-off is worth it, and `enable_nat_gateway = false` removes
NAT entirely. This is the one place a cost-vs-resilience choice is made, so it is
an explicit, documented variable rather than a buried constant.

**No inbound exposure.** Because the workload is a batch Job with no service
surface, the network creates **no** public load balancer, no ingress path to the
nodes, and no bastion. The `elb`/`internal-elb` subnet tags are added so that
load balancers *can* be discovered later if a serving workload ever justifies
one — they cost nothing and are the conventional EKS preparation — but none is
provisioned now.

**Tagging.** Every resource inherits the common `Project`/`Environment`/
`ManagedBy`/`Owner`/`Repository` tags automatically through the provider's
`default_tags` ([ADR-014](ADR-014-terraform-architecture.md)); the network code
adds only resource-specific tags (`Name`, `Tier`, and the EKS role tags).

**Still no module.** The network stays in the root module as `network.tf`. A
module abstracts a boundary that is instantiated more than once; this network is
instantiated exactly once, for one environment. Extracting a `modules/network`
now would add indirection without reuse — the same "no premature modules"
reasoning [ADR-014](ADR-014-terraform-architecture.md) applied, re-evaluated for
this PR and reaching the same answer.

## Alternatives Considered

1. **Public-only subnets, no NAT (nodes get public IPs).**
   - *Rejected* — it is the cheapest option (NAT is the main cost), but it puts
     worker nodes directly on the internet. For a portfolio that markets a
     security-conscious posture, private nodes + NAT is the defensible default.
     Disabling NAT and using public nodes remains reachable via variables for a
     deliberately ultra-low-cost run, but it is not the default.
2. **One NAT gateway per AZ (the production default).**
   - *Deferred behind a variable* — per-AZ NAT removes a single point of failure
     for egress, but doubles NAT cost for a two-AZ VPC that lives for hours. Not
     worth it for this environment; available via `single_nat_gateway = false`.
3. **Three AZs by default.**
   - *Rejected as default* — EKS needs two; a third AZ adds subnets (and, with
     per-AZ NAT, cost) for resilience this short-lived environment does not need.
     Supported via `az_count = 3`.
4. **Hard-code AZ names (e.g. `us-east-1a`, `us-east-1b`).**
   - *Rejected* — brittle and region-locked. Discovering AZs at plan time keeps
     the configuration portable, per the sprint's "define AZs through
     configuration rather than hard-coding" requirement.
5. **Use the community `terraform-aws-modules/vpc` module.**
   - *Rejected* — it is excellent, but wrapping a well-understood ~18-resource
     network in a large external module hides exactly the design (subnet split,
     routing, NAT trade-off) this PR exists to demonstrate, and pulls in options
     far beyond scope. Hand-writing a small, readable VPC is the stronger proof
     here.
6. **A `modules/network` in this repo now.**
   - *Rejected* — single instantiation; see "Still no module" above.

## Consequences

**Positive**

- The network satisfies EKS prerequisites (≥2 AZs, correctly tagged subnets,
  private node placement with egress) so the EKS PR can consume the outputs
  (`vpc_id`, `private_subnet_ids`, …) directly.
- Nodes are private by default; the only internet-facing components are the IGW
  and the NAT gateway.
- The single cost-sensitive decision (NAT count) is explicit, variable-driven,
  and documented, not hidden.
- Naming and tagging are consistent by construction via `default_tags`.

**Trade-offs and follow-ups**

- **A single NAT gateway is a single point of egress failure** and a running
  hourly cost. That is an accepted trade-off for a short-lived environment and is
  reversible with one variable. Cost drivers are documented in
  [`terraform/README.md`](../../terraform/README.md); teardown is PR 8.
- **Network only.** No security groups, IAM, or EKS exist yet — EKS creates its
  own cluster/node security groups in a later PR, so none are defined here to
  avoid unused resources.
- **`plan` requires credentials.** The subnet/AZ data source and the plan read
  the live account; `validate` alone does not. Applying this PR **does** create
  billable resources (chiefly the NAT gateway), unlike the resource-free PR 1.

## What This Decision Does *Not* Imply

- It does **not** provision or configure EKS, node groups, IAM roles, or security
  groups — only the VPC/subnet/routing substrate they will use.
- It does **not** create any inbound path or public service. The workload is a
  batch Job; the load-balancer subnet tags are preparation, not provisioning.
- It does **not** claim a production network: single NAT, two AZs, no flow logs,
  no network firewall, and no multi-region are deliberate scope choices for a
  portfolio environment, not oversights.
- It does **not** move workload configuration into Terraform. Terraform owns the
  VPC; the Kubernetes workload stays in [`k8s/`](../../k8s/).
