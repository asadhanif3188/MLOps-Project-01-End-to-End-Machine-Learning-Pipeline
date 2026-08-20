# NetworkPolicy runtime tests (Sprint 8, PR 7; ADR-034)

Runtime verification of the least-privilege network paths — the counterpart to the
**static** contract in [`k8s/validate.py`](../../validate.py) (Section 8 / M12).

- **Static (CI, no cluster):** proves the policy *set* encodes least privilege —
  default-deny present, DNS preserved, each required path (and only that path)
  allowed. Runs in the `k8s-validate` CI job.
- **Runtime (this suite, needs a cluster):** proves a CNI *actually enforces* the
  policies — every needed path is ALLOWED and an intentionally prohibited path is
  DENIED.

## What it checks

Allowed (must all succeed — required paths, and proof the policies didn't break
anything):

| From (probe label) | To | Port | Why |
|---|---|---|---|
| `mlops-pipeline` | `mlflow` | 5000 | run logging / readiness |
| `mlops-pipeline` | `pushgateway` (monitoring) | 9091 | metrics push |
| `mlflow-server` | `mlflow-postgres` | 5432 | backend store |
| `postgres-exporter` | `mlflow-postgres` | 5432 | Layer 4 scrape |
| `prometheus` (monitoring) | `postgres-exporter` (mlops) | 9187 | scrape preserved |
| `blackbox-exporter` (monitoring) | `mlflow` | 5000 | health probe preserved |

Denied (must all be blocked on an enforcing CNI):

| From | To | Port | Why it must fail |
|---|---|---|---|
| `mlops-pipeline` | `mlflow-postgres` | 5432 | pipeline must reach the DB only *through* MLflow |
| unlabelled pod | `mlflow` | 5000 | default-deny, no allow applies |
| unlabelled pod | `mlflow-postgres` | 5432 | default-deny (also the enforcement canary) |

## How it works

The policies select workloads by their `app.kubernetes.io/name` label, so the suite
launches tiny hardened `curl` probe pods carrying the **same labels** the real
workloads use — a probe exercises the identical policy selectors. Each probe makes
one short-timeout connection; connect-vs-timeout is the evidence.

## Enforcement is required — and detected

A NetworkPolicy is inert unless the CNI enforces it. The script runs an
**enforcement canary** (a known-denied connection) first:

- Canary **blocked** → CNI is enforcing → denied-path checks are trustworthy.
- Canary **connects** → CNI is **not** enforcing (default Docker Desktop CNI, or
  EKS without the VPC CNI `enableNetworkPolicy` flag). The denied-path checks are
  reported **INCONCLUSIVE** — never a false PASS — while allowed-path checks still
  run.

Where NetworkPolicy is enforced for this project:

- **EKS:** the Amazon VPC CNI with `enableNetworkPolicy=true`, set on the `vpc-cni`
  addon in [`terraform/eks.tf`](../../../terraform/eks.tf) (ADR-034).
- **Local:** the default Docker Desktop / kind (kindnet) CNI does **not** enforce
  NetworkPolicy. Use a `kind` cluster with Calico or Cilium for local denied-path
  evidence.

## Usage

```bash
# Deploy the workload first (its NetworkPolicies are part of the overlay):
kubectl apply -k k8s/overlays/local     # or k8s/overlays/aws on EKS

# Then run the suite against the current kube-context:
k8s/tests/netpol/run.sh
# Override namespaces / image / timeout via env vars:
NAMESPACE=mlops MONITORING_NAMESPACE=monitoring k8s/tests/netpol/run.sh
```

Exit `0` = all allowed paths worked **and** (denied paths blocked **or**
non-enforcing CNI clearly reported). Exit `1` = a required path was blocked, or a
denied path was open on an enforcing cluster (a real least-privilege regression).

> **Status in this PR:** the suite is the executable proof harness. Live
> allowed/denied evidence must be captured on an enforcing cluster (EKS with the
> VPC CNI flag, or kind+Calico). See ADR-034 § "Runtime evidence" and
> [docs/network-policies.md](../../../docs/network-policies.md) for the current
> capture status.
