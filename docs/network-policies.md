# Network policies — least-privilege in-cluster paths

**Sprint 8, PR 7.** Design of record: [ADR-034](decisions/ADR-034-network-policies.md).

This document is the evidence base for the Kubernetes `NetworkPolicy` set: the
**communication matrix** the policies are built from, the **policies** themselves,
the **AWS-egress limitation**, and how to **verify** the paths (static + runtime).

The model is **deny-by-default + explicit allow**. Policies select workloads by the
`app.kubernetes.io/name` label they already carry, so a rule targets a workload by
identity, not by IP. NetworkPolicies are additive: the union of all `allow` rules is
what is permitted; everything else is denied.

> **Enforcement is CNI-dependent.** A NetworkPolicy is inert unless the CNI enforces
> it. On **EKS** this requires the Amazon VPC CNI with `enableNetworkPolicy=true`
> (set on the `vpc-cni` addon in [`terraform/eks.tf`](../terraform/eks.tf); VPC CNI ≥
> v1.14). The default **Docker Desktop / kind (kindnet)** CNI does **not** enforce
> NetworkPolicy — objects are admitted but inert. Use kind + Calico/Cilium for local
> enforcement.

---

## 1. Communication matrix

Mapped from the manifests (not assumed). Namespaces: `mlops` (workload), `monitoring`
(observability), `kube-system` (CoreDNS `k8s-app=kube-dns`, the API server, the
kubelet). "S3" is the in-cluster MinIO locally and real Amazon S3 on EKS.

### Egress (who initiates, and to where)

| Source (mlops) | Destination | Port/Proto | Purpose | Expressible precisely? |
|---|---|---|---|---|
| pipeline (Job + init containers) | CoreDNS (kube-system) | 53 UDP/TCP | name resolution | ✅ |
| pipeline | MLflow (mlops) | 5000 TCP | tracking + readiness poll | ✅ |
| pipeline | Pushgateway (monitoring) | 9091 TCP | operational metrics push | ✅ |
| pipeline (fetch-dataset) | S3 | local 9000 / **AWS 443** TCP | dataset fetch | ✅ local / ⚠️ AWS |
| MLflow | CoreDNS | 53 UDP/TCP | name resolution | ✅ |
| MLflow | PostgreSQL (mlops) | 5432 TCP | metadata backend | ✅ |
| MLflow | S3 | local 9000 / **AWS 443** TCP | artifact store | ✅ local / ⚠️ AWS |
| PostgreSQL | — | — | **none** (never initiates) | ✅ (no egress policy) |
| postgres-exporter | CoreDNS | 53 UDP/TCP | name resolution | ✅ |
| postgres-exporter | PostgreSQL | 5432 TCP | read stats views | ✅ |
| minio-setup (local) | CoreDNS; MinIO | 53; 9000 | bucket bootstrap | ✅ |

| Source (monitoring) | Destination | Port/Proto | Purpose | Expressible precisely? |
|---|---|---|---|---|
| Prometheus | CoreDNS | 53 UDP/TCP | name resolution | ✅ |
| Prometheus | **Kubernetes API server** | 443 TCP | SD + kubelet/cAdvisor proxy | ⚠️ env-specific IP |
| Prometheus | KSM / node-exporter / Pushgateway / blackbox (monitoring) | 8080 / 9100 / 9091 / 9115 | scrape | ✅ |
| Prometheus | postgres-exporter (mlops) | 9187 TCP | Layer 4 scrape | ✅ |
| kube-state-metrics | API server | 443 TCP | list/watch objects | ⚠️ env-specific IP |
| Grafana | Prometheus | 9090 TCP | datasource queries | ✅ |
| blackbox-exporter | MLflow (mlops) | 5000 TCP | `/health` probe | ✅ |
| node-exporter / Pushgateway | — | — | **none** | ✅ |

Because Prometheus/KSM must reach the **env-specific API server** (outside the pod
network), the `monitoring` namespace restricts **ingress** and leaves **egress**
unrestricted — a documented trade-off ([ADR-034](decisions/ADR-034-network-policies.md)
§ Monitoring egress).

### Ingress (who may connect in)

| Destination | Allowed sources | Port |
|---|---|---|
| MLflow (mlops) | pipeline; blackbox-exporter (monitoring) | 5000 |
| PostgreSQL (mlops) | MLflow; postgres-exporter | 5432 |
| postgres-exporter (mlops) | Prometheus (monitoring) | 9187 |
| MinIO (mlops, local) | pipeline; MLflow; minio-setup | 9000 |
| pipeline (mlops) | **none** | — |
| Prometheus (monitoring) | Grafana | 9090 |
| kube-state-metrics | Prometheus | 8080 |
| node-exporter | Prometheus | 9100 |
| blackbox-exporter | Prometheus | 9115 |
| Pushgateway | Prometheus (scrape); pipeline (push, cross-ns) | 9091 |
| Grafana | **none** in-cluster (operator via `port-forward` — node path) | — |

DNS, Pod Identity, and the S3 data path are all preserved (DNS is an explicit allow;
Pod Identity credential delivery is node-local and does not traverse the pod
network; S3 egress is allowed per environment).

---

## 2. Policies created

| Namespace | Policy | Effect |
|---|---|---|
| mlops | `default-deny-all` | deny all ingress + egress for every pod |
| mlops | `allow-dns-egress` | all pods → CoreDNS:53 |
| mlops | `allow-pipeline-egress` | pipeline → MLflow:5000, Pushgateway:9091 |
| mlops | `allow-mlflow-ingress` | MLflow ← pipeline, blackbox :5000 |
| mlops | `allow-mlflow-egress` | MLflow → PostgreSQL:5432 |
| mlops | `allow-postgres-ingress` | PostgreSQL ← MLflow, exporter :5432 |
| mlops | `allow-postgres-exporter-ingress` | exporter ← Prometheus :9187 |
| mlops | `allow-postgres-exporter-egress` | exporter → PostgreSQL:5432 |
| mlops (local) | `allow-pipeline-s3-egress` | pipeline → MinIO:9000 |
| mlops (local) | `allow-mlflow-s3-egress` | MLflow → MinIO:9000 |
| mlops (local) | `allow-minio-setup-egress` | minio-setup → MinIO:9000 |
| mlops (local) | `allow-minio-ingress` | MinIO ← pipeline, MLflow, minio-setup :9000 |
| mlops (aws) | `allow-pipeline-s3-egress` | pipeline → 443, public-internet-only |
| mlops (aws) | `allow-mlflow-s3-egress` | MLflow → 443, public-internet-only |
| monitoring | `default-deny-ingress` | deny all ingress for every pod |
| monitoring | `allow-prometheus-ingress` | Prometheus ← Grafana :9090 |
| monitoring | `allow-kube-state-metrics-ingress` | KSM ← Prometheus :8080 |
| monitoring | `allow-node-exporter-ingress` | node-exporter ← Prometheus :9100 |
| monitoring | `allow-blackbox-exporter-ingress` | blackbox ← Prometheus :9115 |
| monitoring | `allow-pushgateway-ingress` | Pushgateway ← Prometheus, pipeline :9091 |

Files: [`k8s/base/networkpolicy.yaml`](../k8s/base/networkpolicy.yaml),
[`k8s/base/mlflow/networkpolicy.yaml`](../k8s/base/mlflow/networkpolicy.yaml),
[`k8s/monitoring/base/networkpolicy.yaml`](../k8s/monitoring/base/networkpolicy.yaml),
[`k8s/overlays/local/networkpolicy.yaml`](../k8s/overlays/local/networkpolicy.yaml),
[`k8s/overlays/aws/networkpolicy.yaml`](../k8s/overlays/aws/networkpolicy.yaml).

---

## 3. The AWS-egress limitation

On EKS, S3 is real Amazon S3 over HTTPS at a **large, dynamic** set of public IPs. A
standard NetworkPolicy can match egress only by pod/namespace selector or a literal
`ipBlock` CIDR — it **cannot** reference an AWS **prefix list**, a **VPC S3
endpoint**, a **security group**, or a **DNS name**, and the VPC CNI implements only
the standard API (no FQDN policy; no mesh added). **So precise least-privilege S3
egress is not expressible at this layer.** Pinning S3's rotating CIDRs would be
brittle — the anti-pattern the brief forbids.

The AWS overlay expresses the tightest **honest** bound: the pipeline and MLflow may
egress on **TCP/443 to the public internet only** (`0.0.0.0/0` minus RFC1918), which
cannot reach any in-VPC / in-cluster address. It restricts *port* and *direction*,
not *which bucket*. The "which bucket / which actions" precision lives where it can
be expressed:

1. **IAM via Pod Identity** (ADR-024/027) — roles grant only the specific bucket +
   actions today.
2. **Recommended (roadmap):** a **VPC S3 gateway endpoint** + endpoint policy, and/or
   an SG egress rule to the **S3 managed prefix list** — AWS-native least privilege at
   the route/SG layer.

Locally, S3 is the MinIO **pod**, so the same path is expressed **precisely** (a pod
selector, no CIDR).

---

## 4. Verification

### Static (CI, no cluster) — [`k8s/validate.py`](../k8s/validate.py)

Section 8 (mlops) and M12 (monitoring) assert the policy *set*: default-deny present,
DNS preserved, each required path (and only it) allowed, the pipeline cannot reach the
DB, PostgreSQL has no egress, no brittle CIDR in-cluster, and the AWS S3 rule has the
internet-only shape. Runs in the `k8s-validate` job for both overlays.

```bash
python k8s/validate.py k8s/overlays/local
python k8s/validate.py k8s/overlays/aws
```

### Runtime (needs an enforcing cluster) — [`k8s/tests/netpol/`](../k8s/tests/netpol/)

Launches hardened probe pods labelled as the real workloads and asserts allowed paths
connect and denied paths are blocked, with an **enforcement canary** so a
non-enforcing CNI yields INCONCLUSIVE (never a false PASS). See the suite's
[README](../k8s/tests/netpol/README.md).

```bash
kubectl apply -k k8s/overlays/local      # or k8s/overlays/aws on EKS
k8s/tests/netpol/run.sh
```

### Runtime evidence — status

The runtime allowed/denied paths have **not** yet been exercised on a live enforcing
cluster in this PR (the static contract passes in CI; the runtime suite is the
executable harness). Capture is a runtime-evidence activity: on EKS (VPC CNI flag on)
or a local kind+Calico cluster, run the suite and record the allowed/denied/canary
results here, as the Sprint 6 runtime test did for the network foundation.
