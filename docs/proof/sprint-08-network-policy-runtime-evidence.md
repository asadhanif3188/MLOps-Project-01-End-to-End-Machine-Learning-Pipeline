# Sprint 8 PR 7 — NetworkPolicy Runtime Evidence

> **STATUS: EXECUTED 2026-08-21** on real EKS (`enableNetworkPolicy=true`). RESULT
> **PASS** — canary **BLOCKED**, **6/6 allowed**, **3/3 denied**. See the consolidated
> record: [sprint-08-live-eks-evidence.md §5](sprint-08-live-eks-evidence.md#5-pr-7--networkpolicy-runtime).
> Two findings surfaced here: (1) enforced NetworkPolicy **blocked EKS Pod Identity**
> (`169.254.170.23:80`) — fixed with `allow-pod-identity-egress`; (2) the harness judged
> success on curl's exit code, which is **unreliable on EKS** (curl exit 23 on a
> successful request) — fixed to judge on `%{time_connect}`
> ([findings §3](sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed)).

---

**Original pre-execution checklist (retained for reference):**

> The least-privilege NetworkPolicy set
> ([ADR-034](../decisions/ADR-034-network-policies.md)) is merged with a **static**
> contract passing in CI (`k8s/validate.py` §8/M12), but the **live** allowed/denied
> paths have not been exercised on an enforcing cluster. This file is the
> **checklist to complete that capture** — do it the next time an enforcing cluster
> exists. It is safe to defer: the policies are **inert** until a CNI enforces them,
> so nothing at runtime changes until then (the default local Docker Desktop CNI does
> not enforce NetworkPolicy).

> **⏳ Capture this TOGETHER with the other pending Sprint 8 live-EKS captures:**
> [sprint-08-sbom-provenance-evidence.md § 4b](sprint-08-sbom-provenance-evidence.md#4b-operator-checklist-run-on-the-next-enforcing-cluster-session)
> (PR 9 — push → immutable digest → `verify-deployed-digest.sh` PASS),
> [sprint-08-dataset-failure-tests-evidence.md](sprint-08-dataset-failure-tests-evidence.md)
> (PR 10 — dataset unavailable + checksum-mismatch failure paths), and
> [sprint-08-mlflow-failure-tests-evidence.md](sprint-08-mlflow-failure-tests-evidence.md)
> (PR 11 — MLflow outage detection & recovery). Standing up EKS is
> the billable/expensive part (`provision → prove → destroy`, ADR-020); the checks are
> minutes each. So on the **next enforcing-cluster session, do all FOUR in one run** to
> amortise the cluster cost — deploy the workload once, then run this harness, the PR 9
> digest verification, the PR 10 dataset-failure harness, *and* the PR 11 MLflow-outage
> harness against the same live cluster before teardown.

**Design of record:** [ADR-034](../decisions/ADR-034-network-policies.md) ·
**Matrix + policies:** [docs/network-policies.md](../network-policies.md) ·
**Harness:** [`k8s/tests/netpol/run.sh`](../../k8s/tests/netpol/run.sh)

---

## ⚠️ Verify this FIRST (highest blast radius)

**Kubelet health probes under default-deny ingress.** Every workload except
PostgreSQL (which probes over loopback) health-checks over the pod network. The
policy set assumes the enforcing CNI permits **node-sourced** probe traffic. If it
does not, MLflow never becomes Ready → its Service has zero Endpoints → the
pipeline's `wait-for-mlflow` init container fails the whole Job — a **platform-wide
outage**. The runtime harness's allowed-path checks (pods reachable **and Ready**
under policy) are the guard. **Confirm pods stay Ready before trusting enforcement.**

---

## Prerequisites

- [ ] An **enforcing** cluster:
  - **EKS** — `terraform apply` with the VPC CNI `enableNetworkPolicy=true` flag
    (already in [`terraform/eks.tf`](../../terraform/eks.tf)); **or**
  - **Local** — a `kind` cluster with **Calico** (or Cilium) installed (the default
    Docker Desktop / kindnet CNI does **not** enforce NetworkPolicy).
- [ ] The workload deployed: `kubectl apply -k k8s/overlays/<aws|local>` (its
      NetworkPolicies are part of the overlay) and all pods **Ready**.
- [ ] The monitoring stack deployed if verifying its ingress paths:
      `kubectl apply -k k8s/monitoring/overlays/<aws|local>`.

## Steps

- [ ] **1. Pods stay Ready under policy** (the probe check above):
      `kubectl -n mlops get pods` and `kubectl -n monitoring get pods` — all Ready.
- [ ] **2. Run the harness:** `k8s/tests/netpol/run.sh`
- [ ] **3. Confirm the enforcement canary is BLOCKED** (line begins
      `[ok] canary blocked`). If it says `[WARN] canary CONNECTED`, the CNI is **not**
      enforcing — the denied-path results are INCONCLUSIVE; fix enforcement first.
- [ ] **4. Confirm all ALLOWED paths PASS** (pipeline→MLflow, pipeline→Pushgateway,
      MLflow→Postgres, exporter→Postgres, Prometheus→exporter, blackbox→MLflow).
- [ ] **5. Confirm all DENIED paths PASS** (pipeline→Postgres blocked,
      unlabelled→MLflow blocked, unlabelled→Postgres blocked).
- [ ] **6. Capture** the `RESULT:` line and the allowed/denied/canary summary.

## Record results here (fill in on execution)

| Dimension | Result |
|---|---|
| **Date / cluster / CNI** | _pending_ (e.g. EKS VPC CNI `enableNetworkPolicy`, or kind+Calico) |
| **Pods Ready under policy** | _pending_ (the probe-assumption gate) |
| **Enforcement canary** | _pending_ (must be BLOCKED) |
| **Allowed paths** | _pending_ (6/6 expected) |
| **Denied paths** | _pending_ (3/3 expected) |
| **`run.sh` RESULT** | _pending_ (PASS expected on an enforcing cluster) |
| **Teardown** | _pending_ |

> Redact account IDs / operator IPs / any secret material, per the Sprint 7 evidence
> convention. Paste the harness output (canary + allowed + denied lines) below the
> table when executed, and flip the STATUS banner at the top to **EXECUTED &lt;date&gt;**.
