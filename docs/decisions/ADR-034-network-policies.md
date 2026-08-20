# ADR-034: Least-privilege Kubernetes NetworkPolicies (Sprint 8, PR 7)

- **Status:** Accepted (design; policies + static contract + runtime harness + CNI enablement + docs added, no components deployed / no live allowed-or-denied path exercised on a cluster in this PR)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:**
  [ADR-024 (VPC CNI identity via EKS Pod Identity)](ADR-024-vpc-cni-pod-identity.md),
  [ADR-026 (In-cluster MLflow platform — the data-flow this policy mirrors)](ADR-026-in-cluster-mlflow-platform.md),
  [ADR-027 (S3 dataset runtime retrieval — Pod Identity S3 access)](ADR-027-s3-dataset-runtime-retrieval.md),
  [ADR-028 (Observability architecture — the scrape graph)](ADR-028-observability-architecture.md),
  [ADR-029 (Monitoring foundation — the `monitoring` namespace + PSA exception)](ADR-029-monitoring-foundation.md),
  [ADR-030 (Pipeline operational metrics — the Pushgateway push path)](ADR-030-pipeline-operational-metrics.md),
  [ADR-031 (MLflow & PostgreSQL monitoring — blackbox `/health` + postgres-exporter)](ADR-031-mlflow-postgres-monitoring.md),
  [ADR-010 (Kubernetes workload security hardening)](ADR-010-kubernetes-security-hardening.md),
  [ADR-012 (Kubernetes manifest validation)](ADR-012-kubernetes-manifest-validation.md),
  [ADR-015 (AWS network architecture — VPC/NAT)](ADR-015-aws-network-architecture.md),
  [`docs/network-policies.md`](../network-policies.md),
  [`k8s/tests/netpol/`](../../k8s/tests/netpol/)

> **Scope.** This ADR ratifies the **network** layer of least privilege: a
> default-deny baseline plus explicit, minimal allow-rules for every real
> communication path, expressed as Kubernetes `NetworkPolicy`. It **adds policies +
> a static contract (k8s/validate.py) + a runtime test harness + the EKS CNI
> enablement + docs**; **nothing is deployed** and **no allowed/denied path has been
> exercised on a live cluster** in this PR. It deliberately adds **no service mesh**
> (no Istio/Cilium/Linkerd), per the brief. It does not re-decide identity (Pod
> Identity, ADR-024/027) or the scrape graph (ADR-028–031); it fences them.

## Context

The platform runs two namespaces — `mlops` (the pipeline Job, the MLflow tracking
server, PostgreSQL, postgres-exporter, and locally MinIO) and `monitoring`
(Prometheus, Grafana, kube-state-metrics, node-exporter, Pushgateway,
blackbox-exporter). Until this PR there was **no network segmentation**: any pod
could open a connection to any other pod or Service on any port. Identity (Pod
Identity), RBAC (read-only), and Pod Security (`restricted`) were all in place, but
the *network* was flat — a compromised or misbehaving pod had an unrestricted
lateral surface (e.g. straight to the PostgreSQL metadata DB, or to the
unauthenticated Prometheus `/-/reload` and Pushgateway push endpoints).

The brief is **evidence-driven**: map the *actual* communication paths first, write
policies to that map, apply default-deny where appropriate, and — critically — **do
not break DNS, Pod Identity/AWS access, or Prometheus scraping**, and **do not
create brittle IP-based policies for AWS services with dynamic addresses**; where
least-privilege AWS egress cannot be expressed safely, *document the limitation*.

### The communication matrix (mapped from the manifests, not assumed)

The full evidence table is in [`docs/network-policies.md`](../network-policies.md).
In brief (source → destination : port):

- **Pipeline** → CoreDNS:53; → MLflow:5000; → Pushgateway(monitoring):9091; → S3
  (local MinIO:9000 / AWS S3:443). No ingress (nothing connects to it); no API
  server (token automount off).
- **MLflow** → CoreDNS:53; → PostgreSQL:5432; → S3 (MinIO:9000 / S3:443). Ingress
  from the pipeline:5000 and blackbox:5000.
- **PostgreSQL** — ingress from MLflow:5432 and postgres-exporter:5432. **No
  egress** (it never initiates; `pg_isready` is loopback).
- **postgres-exporter** → CoreDNS:53; → PostgreSQL:5432. Ingress from
  Prometheus:9187.
- **Prometheus** → CoreDNS:53; → the **Kubernetes API server**:443 (SD + the
  kubelet/cAdvisor proxy); → KSM:8080, node-exporter:9100, Pushgateway:9091,
  blackbox:9115 (monitoring), postgres-exporter:9187 (mlops). Ingress from
  Grafana:9090.
- **Grafana** → CoreDNS:53; → Prometheus:9090.
- **blackbox** → CoreDNS:53; → MLflow:5000 (`/health`). Ingress from
  Prometheus:9115.
- **KSM** → CoreDNS:53; → API server:443. Ingress from Prometheus:8080.
- **node-exporter** — no egress; ingress from Prometheus:9100.
- **Pushgateway** — no egress; ingress from Prometheus:9091 (scrape) and the
  pipeline:9091 (push).

### CNI support (inspected)

- **EKS:** the Amazon VPC CNI (`vpc-cni` addon, ADR-024). It *admits* NetworkPolicy
  objects but **only enforces** them when its network-policy agent is enabled, which
  is **off by default**. Requires VPC CNI ≥ v1.14.
- **Local:** the default Docker Desktop / kind (kindnet) CNI does **not** enforce
  NetworkPolicy at all — objects are admitted but inert.

Both facts are load-bearing: a policy nobody enforces is theatre. This PR turns
enforcement **on** for EKS and is explicit that the default local CNI does not
enforce (use kind+Calico/Cilium locally for enforcement).

## Decision

**1. Default-deny + explicit allow, per namespace, selected by workload identity.**
Every policy selects workloads by the `app.kubernetes.io/name` label they already
carry (not by IP). NetworkPolicies are additive, so the model is: deny everything,
then add back exactly one path at a time.

- **`mlops`:** a `default-deny-all` (all pods, ingress **and** egress), then:
  DNS egress to CoreDNS:53 for all pods; the pipeline's egress (MLflow, Pushgateway,
  + the overlay S3 leg); MLflow's ingress (pipeline + blackbox) and egress
  (Postgres + overlay S3); Postgres's ingress (server + exporter only) and **no
  egress**; the exporter's ingress (Prometheus) and egress (Postgres). Files:
  [`k8s/base/networkpolicy.yaml`](../../k8s/base/networkpolicy.yaml),
  [`k8s/base/mlflow/networkpolicy.yaml`](../../k8s/base/mlflow/networkpolicy.yaml).
- **`monitoring`:** a `default-deny-ingress` plus one explicit ingress path per
  component matching the scrape graph. **Egress is deliberately left unrestricted**
  in this namespace (see § *Monitoring egress*). File:
  [`k8s/monitoring/base/networkpolicy.yaml`](../../k8s/monitoring/base/networkpolicy.yaml).

**2. DNS, Pod Identity, and monitoring are preserved by construction.** DNS is an
explicit egress allow to CoreDNS on 53 (UDP+TCP). Pod Identity's credential
delivery is node-local (the agent is a `hostNetwork` DaemonSet), so it does not
traverse the pod network and is unaffected by pod policies; the AWS *data* path
(S3) is preserved by the S3 egress rules. Every scrape/push/query path in the
matrix has an explicit allow, and the static contract asserts each one so a
default-deny cannot silently break it.

**3. Enable enforcement on EKS.** The `vpc-cni` addon now sets
`configuration_values = {"enableNetworkPolicy":"true"}`
([`terraform/eks.tf`](../../terraform/eks.tf)). This is what makes "enforce" real on
the cloud platform.

**4. Two-layer verification.**
- *Static* ([`k8s/validate.py`](../../k8s/validate.py) Section 8 / M12, in CI):
  proves the policy *set* — default-deny present, DNS preserved, each required path
  (and only it) allowed, no brittle CIDR in-cluster, and the AWS S3 rule has the
  expected internet-only shape.
- *Runtime* ([`k8s/tests/netpol/run.sh`](../../k8s/tests/netpol/run.sh)): launches
  hardened probe pods carrying the workloads' labels, asserts every allowed path
  connects and every denied path is blocked, and runs an **enforcement canary** so a
  non-enforcing CNI yields INCONCLUSIVE — never a false PASS.

### The AWS-egress limitation (documented, not pretended solved)

On EKS, "S3" is real Amazon S3 over HTTPS at a **large, dynamic** set of public IP
ranges. A standard Kubernetes NetworkPolicy can match egress only by pod/namespace
selector or a literal `ipBlock` CIDR. It **cannot** reference an AWS-managed **prefix
list** (`com.amazonaws.<region>.s3`), a **VPC S3 gateway/interface endpoint**, a
**security group**, or a **DNS name**. The VPC CNI's policy engine implements the
*standard* API only — none of the AWS-native S3 constructs are expressible in it,
and we add no mesh with FQDN policy. **Therefore precise least-privilege S3 egress
cannot be expressed at the NetworkPolicy layer here.** Pinning S3's current CIDRs
would be brittle (they rotate) — the exact anti-pattern the brief forbids.

What the AWS overlay *does* express is the tightest **honest** bound: the two S3
clients (pipeline + MLflow) may egress on **TCP/443 to the public internet only** —
`0.0.0.0/0` with every RFC1918 range **and link-local `169.254.0.0/16`** in
`except`, so the allowance cannot reach any in-VPC / in-cluster address (pod net,
Service CIDR, node/control-plane ENIs, the API server) or any link-local endpoint
(the EC2 IMDS `169.254.169.254`, the Pod Identity agent `169.254.170.23`). It
restricts the *port* and *direction*, **not which S3 host/bucket**. (The `except`
list assumes an RFC1918 VPC CIDR — the default `10.0.0.0/16`, `terraform/variables.tf`;
a CGNAT `100.64.0.0/10` VPC would need adding, a coupling noted here since the two
files must stay consistent.)

The missing precision lives where it *can* be expressed (defence in depth):

1. **IAM via Pod Identity** (ADR-024/027, `terraform/s3.tf`, `terraform/datasets.tf`)
   — the pipeline's and MLflow's roles already grant only the specific bucket +
   actions. A packet leaving on 443 still cannot read/write any S3 object outside
   those grants.
2. **Recommended (roadmap, terraform scope):** a **VPC S3 gateway endpoint** +
   endpoint policy (keeps S3 traffic on the AWS backbone, scoped by prefix list +
   bucket), and/or a **security-group egress rule to the S3 managed prefix list** —
   the AWS-native mechanisms that *do* express S3 least privilege, at the route/SG
   layer NetworkPolicy cannot reach. Not provisioned here; recorded, not assumed.

Locally the same logical path targets the in-cluster MinIO **pod**, so it is
expressed **precisely** (a pod selector, no CIDR) — the local overlay is fully
least-privilege with no wildcard anywhere.

### Monitoring egress — a deliberate, scoped non-restriction

The `monitoring` namespace restricts **ingress** (high value, cleanly expressible)
but leaves **egress unrestricted**. Prometheus and kube-state-metrics must reach the
**Kubernetes API server** — an endpoint *outside* the pod network (the control-plane
ENIs on EKS, a host IP locally) whose address is environment-specific — for service
discovery and the kubelet/cAdvisor proxy scrape, and Prometheus discovers/scrapes
targets cluster-wide. Pinning that egress to CIDRs would be brittle and risks
silently breaking scraping — the failure mode the brief explicitly warns against.
The strong ingress controls (nothing may reach Prometheus/Grafana/Pushgateway/
exporters except the intended client) deliver the least-privilege value without that
brittleness. This is a documented trade-off, not an oversight.

### Health probes under default-deny

Kubelet liveness/readiness probes arrive from the **node**, not from a pod. The
enforcing CNIs this project targets (AWS VPC CNI network policy, Calico, Cilium)
permit node-sourced probe traffic, so a default-deny ingress does not break probes.
The runtime allowed-path test verifies pods stay Ready under policy; if a CNI ever
did block probes, that test — not a silent outage — would catch it.

This is the **highest-blast-radius unverified assumption** in the PR (every workload
except PostgreSQL, which probes over loopback, health-checks over the pod network),
so it is the **first** thing to exercise via `k8s/tests/netpol/run.sh` on the actual
target CNI before the policy set is trusted there. Were it wrong, MLflow would never
become Ready, its Service would have zero Endpoints, and the pipeline's
`wait-for-mlflow` init container would fail the whole Job — a platform-wide outage,
not a partial one. There is no portable way to select "the node" in a NetworkPolicy,
so the mitigation is the CNI's documented node-probe exemption plus this runtime
guard, not an extra ingress rule.

## Alternatives considered

- **Service mesh (Istio/Linkerd/Cilium mesh) for mTLS + L7 policy.** Rejected —
  **explicitly out of scope** per the brief, and disproportionate for a
  single-operator batch platform. NetworkPolicy gives L3/L4 segmentation with zero
  new runtime components.
- **Pin S3 CIDRs in the NetworkPolicy.** Rejected — brittle (AWS rotates the ranges)
  and the anti-pattern the brief forbids. Precision is delegated to IAM + the
  recommended VPC endpoint instead.
- **Also default-deny egress in `monitoring`.** Rejected for Prometheus/KSM — their
  egress to the env-specific API server cannot be pinned without brittleness and
  risks breaking scraping. Ingress restriction captures the value safely.
- **Allow-all-egress for the S3 clients on AWS.** Rejected — the RFC1918-excepted
  `0.0.0.0/0:443` bound is strictly tighter (internet-only, one port) at no cost.
- **Skip enabling the VPC CNI flag (ship inert policies).** Rejected — that is
  policy theatre; the PR's title says *enforce*.

## Consequences

**Positive.** The lateral network surface is now deny-by-default in both namespaces;
the most sensitive workload (PostgreSQL) is reachable by exactly two peers and has
zero egress; the pipeline provably cannot reach the DB directly. DNS, Pod Identity,
the S3 data path, and the full scrape graph are preserved and *asserted* in CI. The
controls are verifiable at two levels (static contract + runtime harness) and add no
new runtime components.

**Negative / limitations (stated, not hidden).**
- Precise least-privilege **S3 egress is not expressible** at the NetworkPolicy layer
  on the VPC CNI (§ above); it is bounded to internet-only:443 and delegated to IAM +
  a recommended VPC endpoint.
- **`monitoring` egress is unrestricted** by design (§ above).
- Policies are **inert on the default local CNI**; enforcement needs EKS (flag on) or
  kind+Calico/Cilium.
- **No live evidence in this PR:** the allowed/denied paths have not been exercised on
  a running enforcing cluster. The runtime harness is the executable proof; capture
  is a runtime-evidence activity (see `docs/network-policies.md` § Runtime evidence).
- Enabling the VPC CNI network-policy agent adds a small per-node eBPF component
  (part of the addon) — negligible for this cluster size.

**Runtime evidence.** To be captured on an enforcing cluster: apply an overlay, run
`k8s/tests/netpol/run.sh`, and record the allowed/denied/​canary results, as the
Sprint 6 runtime test did for the network foundation. The step-by-step capture
checklist (with the probe-assumption gate and a results table to fill in) is tracked
in [`docs/proof/sprint-08-network-policy-runtime-evidence.md`](../proof/sprint-08-network-policy-runtime-evidence.md).
