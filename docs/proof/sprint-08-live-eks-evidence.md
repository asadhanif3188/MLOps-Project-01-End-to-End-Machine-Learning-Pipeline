# Sprint 8 — Batched Live-EKS Runtime Evidence

> **STATUS: EXECUTED 2026-08-21.** This is the authoritative record of the single
> batched `provision → prove → destroy` EKS session that closed every pending Sprint 8
> runtime-evidence item at once (per each doc's "do all of these in one cluster
> session to amortise the cost" instruction and [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md)).
> The five previously-`PENDING` proof docs (PR 7, 9, 10, 11, 12) are closed by the
> results below and each links here.

**Account/region redaction:** all AWS account IDs, bucket names, and operator IPs are
redacted (`<ACCOUNT>`) per the Sprint 7 evidence convention.

---

## 1. Environment

| Dimension | Value |
|---|---|
| Cluster | `mlops-pipeline-dev-eks`, EKS **v1.35** (`v1.35.6-eks`) |
| Nodes | **2 × t3.large** (AL2023 x86_64), private subnets, 2 AZs |
| Provisioned | 2026-08-21 ~12:00Z — `Apply complete! Resources: 65 added` |
| Torn down | 2026-08-21 ~14:30Z — see [§8 Teardown](#8-teardown) (lifetime ~2.5 h) |
| Network policy | VPC CNI `enableNetworkPolicy=true` — `aws-node` **2/2** (nodeagent enforcing) |
| Workload identity | EKS Pod Identity (dataset-reader, mlflow-s3, vpc-cni) — no static keys |
| Pipeline image | `…/mlops-pipeline@sha256:3a27c6de…58f0498c` (git `72fad8a`) |
| MLflow image | `…/mlflow-server@sha256:e49ead8f…329fc286` |

> **Node sizing note.** Deliberately **t3.large** (not the documented `t3.medium`)
> because this campaign runs the full monitoring stack (Prometheus/Grafana/exporters)
> *alongside* MLflow + PostgreSQL + the pipeline — heavier than the Sprint 7 runs.
> Memory headroom prevents an unrelated eviction from confounding the OOM/crash-loop
> evidence. Node count stayed 2 (same 2-AZ shape).

---

## 2. Results summary

| # | Item | Result |
|---|---|---|
| — | **Baseline run** | Job `Complete`, pod `Succeeded`, **exit 0**, QoS Burstable; **2 MLflow runs** logged (metadata in PostgreSQL, `artifact_location: mlflow-artifacts:/1` in S3 via Pod Identity); model *"Best Random Forest Classifier"* registered |
| — | **Observability baseline** | **8/8** scrape jobs UP; `pg_up=1`; **8** alert rules loaded (closes the PR 2–6 monitoring-stack runtime proof) |
| **9** | Digest provenance | `git 72fad8a → tag 1.6.0 → sha256:3a27c6de…` (ECR cross-checked); **`verify-deployed-digest.sh` PASS — all 3 containers** run the pinned digest; CycloneDX SBOM 322 components |
| **7** | NetworkPolicy | canary **BLOCKED** (CNI enforcing), **6/6 allowed** PASS, **3/3 denied** PASS — RESULT **PASS** |
| **10** | Dataset failure | A (missing object → **404 HeadObject**) + B (**checksum mismatch**) both fail **before training**; `PipelineJobFailed` **FIRING** (13:26:13Z) |
| **12** | Resource failure | Real **`OOMKilled`** (exit **137**) — the EKS-specific reason; `KubePodCrashLooping` **FIRING**; `PipelineJobOOMKilled` **FIRING** (after Finding 4 fix) |
| **11** | MLflow outage | outage → `probe_success` 1→0 → `MLflowDown` **FIRING @ 5m** → gate blocks (**no wasted compute**) → recover → **persistence proven** (runs 2→5, `pg_up=1` throughout) |
| **13** | Retry hardening | transient **~90 s** mid-run MLflow blip **absorbed**; training **not discarded**; Job **exit 0** (train stage extended to 117 s while retrying) |

**All four operational alerts fired live:** `PipelineJobFailed`, `PipelineJobOOMKilled`,
`KubePodCrashLooping`, `MLflowDown`.

---

## 3. Findings — 4 real defects the live run surfaced (all fixed)

Static validation (CI `kubeconform`/Kustomize/`validate.py`/`promtool`) passed for all
of these; each only manifested **under live enforcement/runtime on EKS**. This is the
core value of the batched live session.

### Finding 1 — Enforced NetworkPolicy blocked EKS Pod Identity (CRITICAL)
- **Symptom:** `fetch-dataset` failed with
  `CredentialRetrievalError: Connect timeout on endpoint URL "http://169.254.170.23/v1/credentials"`.
- **Cause:** the S3-egress policies `except` the whole `169.254.0.0/16` link-local range
  (to keep the `:443` rule tight) and **nothing re-allowed** the Pod Identity agent at
  `169.254.170.23:80`, so default-deny blocked credential retrieval. Sprint 7 proved Pod
  Identity live but had **no enforced NetworkPolicies** — this campaign is the first time
  the two ran together. The static "Pod Identity preserved" claim (validate.py §8) never
  held under a real enforcing CNI.
- **Fix:** `allow-pod-identity-egress` — a least-privilege egress rule (one `/32`, TCP/80)
  for the two Pod-Identity workloads ([`k8s/overlays/aws/networkpolicy.yaml`](../../k8s/overlays/aws/networkpolicy.yaml)).
- **Proof:** after the fix, `fetch-dataset` → **exit 0**, baseline run **Complete**, MLflow
  logged artifacts to S3.

### Finding 2 — postgres-exporter arg rejected by its pinned image
- **Symptom:** `postgres-exporter` **CrashLoopBackOff**, `error: unexpected false, try --help`.
- **Cause:** `--auto-discover-databases=false` — the `v0.15.0` flag parser rejects the
  `=false` form of that boolean.
- **Fix:** `--no-auto-discover-databases` ([`k8s/base/mlflow/postgres-exporter.yaml`](../../k8s/base/mlflow/postgres-exporter.yaml)).
- **Proof:** exporter **Running 1/1**, `pg_up=1`, `postgres-exporter` scrape target UP.

### Finding 3 — netpol harness trusted curl's (unreliable) exit code
- **Symptom:** harness reported **6/6 allowed paths FAIL** while the paths demonstrably
  worked (manual probe → HTTP 200).
- **Cause:** in a hardened, drop-ALL container via `kubectl exec` on EKS/containerd, curl
  completes the request (HTTP 200) but **exits 23** on the response-write phase
  ("client returned ERROR on write"). The harness judged success on the exit code only.
- **Fix:** judge on **TCP-connect success** (`%{time_connect} > 0`), robust to the
  write-phase exit code, valid for both `http://` and `telnet://` probes
  ([`k8s/tests/netpol/run.sh`](../../k8s/tests/netpol/run.sh)).
- **Proof:** re-run → **9 passed, 0 failed** — RESULT PASS.

### Finding 4 — `PipelineJobOOMKilled` alert could never fire (memory-safety gap)
- **Symptom:** pods genuinely `OOMKilled` (kubectl + `state.terminated.reason`), yet the
  alert stayed **absent**; the alert expr evaluated **empty**.
- **Cause:** the alert keyed on `kube_pod_container_status_last_terminated_reason`, which
  KSM derives from a container's **`lastState`** (its *previous*, post-restart termination).
  The pipeline Job runs `restartPolicy: Never` → each container terminates **once, no
  restart** → `lastState` empty → KSM emits **no series** → the alert is **unfireable for
  this workload**. Confirmed live: `..._terminated_reason{reason="OOMKilled"}=1` (current
  state) while `..._last_terminated_reason` had **0 series**.
- **Fix:** key on `kube_pod_container_status_terminated_reason` (current state)
  ([`k8s/monitoring/base/prometheus/alerts.yml`](../../k8s/monitoring/base/prometheus/alerts.yml)).
- **Proof:** after the fix + live reload, `PipelineJobOOMKilled` → **FIRING** (14:19:49Z).

> **Deploy-runbook gap (minor):** the Sprint 8 monitoring stack needs one out-of-band
> Secret the [cloud-operations §3.8 runbook](../cloud-operations.md) does not list —
> `mlflow-postgres-exporter-credentials` (+ the `mlflow_exporter` `pg_monitor` role).
> Documented in [monitoring-operations.md](../monitoring-operations.md); the runbook
> should cross-reference it.

---

## 4. PR 9 — SBOM & immutable provenance (live digest)

```
git commit : 72fad8aa5f8f21e884b9f77f76d6a0d1c4dcab4b
image tag  : <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline:1.6.0
digest     : sha256:3a27c6de5a4b389df11c6fb99113213cc4bb0a5ae2ee4cf268d39bae58f0498c   (ECR cross-checked)
SBOM       : mlops-pipeline-1.6.0.cdx.json (~322 components)

scripts/verify-deployed-digest.sh --expect sha256:3a27c6de…
  OK  mlops-pipeline-<pod>/fetch-dataset:    sha256:3a27c6de…
  OK  mlops-pipeline-<pod>/wait-for-mlflow:  sha256:3a27c6de…
  OK  mlops-pipeline-<pod>/pipeline:         sha256:3a27c6de…
  PASS: all 3 container(s) run the expected digest
```
Deployed **pinned by digest** (`IMAGE_DIGEST=… render-cloud-manifests.sh --apply`).
cosign signing remains opt-in (not installed); unchanged.

## 5. PR 7 — NetworkPolicy runtime

```
Enforcement canary (unlabelled -> PostgreSQL:5432, must be blocked)... [ok] canary blocked
ALLOWED (6/6 PASS): pipeline->MLflow:5000, pipeline->Pushgateway:9091, mlflow->Postgres:5432,
                    postgres-exporter->Postgres:5432, Prometheus->postgres-exporter:9187, blackbox->MLflow:5000
DENIED  (3/3 PASS): pipeline->Postgres (no bypass), unlabelled->MLflow, unlabelled->Postgres
Summary: 9 passed, 0 failed, 0 inconclusive.  RESULT: PASS
```
All workload pods **Ready under policy** (the probe-assumption gate). See Finding 3 for
the harness fix that made the allowed-path results trustworthy.

## 6. PR 10 / 12 — failure paths & alerts

**Dataset (PR 10):** Scenario A → `404 HeadObject Not Found`, `fetch-dataset` exit 1,
pipeline never started. Scenario B → `Dataset integrity check failed … expected <bad>,
got ee5b0c92…` (the real digest), exit 1, pipeline never started. Real Job override →
Failed (BackoffLimitExceeded, 3 pods) → **`PipelineJobFailed` FIRING** (critical, 13:26:13Z).

**Resource (PR 12 + PR 13-E6):** 128Mi limit → pipeline container **`OOMKilled`, exit 137**
(the real EKS reason — Docker Desktop only ever reported `Error`, closing PR 12 finding #3
/ ADR-037 E6). Job Failed (BackoffLimitExceeded). Crash-loop (restartPolicy: Never →
`RESTARTS=0`, new pod per retry, backoffLimit=2 → 3 pods → Failed). A representative
`restartPolicy: Always` pod held **>15 min** → **`KubePodCrashLooping` FIRING**.
`PipelineJobOOMKilled` **FIRING** after the Finding-4 fix (14:19:49Z).

## 7. PR 11 — MLflow outage detection & recovery (8 items)

| # | Item | Result |
|---|---|---|
| 1 | Outage method | `kubectl scale deploy/mlflow --replicas=0` (13:54:22Z); PostgreSQL/S3 untouched |
| 2 | Detection | `probe_success` **1→0** (13:55:50Z); Service **Endpoints drained empty** |
| 3 | Alert | `MLflowDown` Pending → **FIRING at 14:00:06Z** (5m `for:`, warning); **Resolved** on restore |
| 4 | Pipeline effect | `fetch-dataset` **exit 0**; `wait-for-mlflow` **blocks/times out** (`urlopen error timed out`, poll 53/60…); **pipeline container never started** → **no wasted compute** |
| 5 | Diagnosis | `MLflowDown` firing while the pipeline is blocked at the gate (runbook signature); `PipelineJobFailed` firing captured independently (§6) |
| 6 | Recovery | scale → 1 (14:03:22Z); `probe_success` **0→1** (14:03:47Z); `MLflowDown` **RESOLVED** (14:04:04Z) |
| 7 | Persistence | `pg_up=1` **throughout**; runs **2 → 5 → 10** across the campaign (previous runs survived; Postgres + S3 intact) |
| 8 | Candidates | #1 (bounded retry) — implemented PR 13, **now proven live** (§ below) |

## PR 13 — bounded retry rides out a transient blip

Blip 14:25:30Z→14:26:25Z (MLflow to 0 and back). Train **compute finished 14:25:38**
(model saved, acc 0.7398) — but MLflow was down, so the tracking call **retried**
(connection-refused at 14:26:48/53, 14:27:02) and **succeeded at 14:27:33** once MLflow
returned. **Train stage duration = 117 s** (vs ~3 s compute — the retry waiting). Job
**exit 0**. The completed training was **not discarded** by the transient blip — exactly
ADR-037's intent. (A *persistent* outage still fails fast — proven at the start-gate in
item 4 and by the unit tests.)

---

## Screenshots

Under [`docs/screenshots/`](../screenshots/):
`grafana-platform-health-baseline.png`, `grafana-pipeline-ops-baseline.png`,
`grafana-mlflow-health-baseline-green.png`, `mlflow-runs-baseline.png`,
`grafana-mlflow-health-outage-red.png` (RED availability gauge, PostgreSQL panels green
during the outage), `prometheus-mlflowdown-firing.png`, and platform-health captures with
failing pods. Raw PromQL / REST / `kubectl` captures are the primary evidence; screenshots
are the visual complement.

## 8. Teardown

`terraform destroy` executed at end of session (start 14:30:50Z; see
[Cloud Operations §5](../cloud-operations.md#5-safe-teardown)). The **billable** resources
(EKS control plane, both `t3.large` nodes, NAT gateway, EIP) were removed early in the
destroy. Two well-known EKS-teardown races required manual finishing (the runbook's "if
any check still shows a resource, re-run / delete stragglers" step):

1. A lingering **VPC CNI ENI** (`aws-K8S-…`, `available`) held a private subnet →
   `DependencyViolation`. Deleted the detached ENI; the subnet then deleted.
2. The **EKS-auto-created cluster security group** (`eks-cluster-sg-…`, **not**
   Terraform-managed) held the VPC → `DependencyViolation`. Deleted it, then the VPC.
   (Operator DNS flakiness — the session's network instability — also caused transient
   `dial tcp … no such host` retries against the EC2 API during this step.)

The manually-deleted VPC was then reconciled out of state (`terraform state rm aws_vpc.this`).

**Verified clean (3 angles):** Terraform state **0 managed resources**; `aws eks
list-clusters` **none**; `aws ec2 describe-nat-gateways` (available) **none**; VPC
**not found**; `aws ecr describe-repositories mlops-pipeline` **not found**; no unattached
EIP. The three customer-managed **KMS keys** move to a 7-day `PendingDeletion` window
(`kms_key_deletion_window_days=7`, [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md))
— expected, not a leak; they auto-delete. **Nothing is billing.**

## 9. Honesty boundary

- Everything above was executed on a **real, Terraform-provisioned EKS cluster** on
  2026-08-21 and captured live; it is **not** a static/local claim.
- The cluster was a **short-lived, single-operator validation environment**, destroyed and
  verified clean the same session — **not** production.
- The four fixes were **applied to the committed manifests/harness** and re-validated
  (`kustomize build` OK; `k8s/validate.py` 201/201). CI (`promtool`, `kubeconform`) runs
  the full static gate on push.
- PR 11 item 5's *combined* `MLflowDown`+`PipelineJobFailed` on the **same** Job would need
  ~15 min (3× the 300 s start-gate); both alerts are proven firing (§6/§7) and the
  correlation signature (blocked pipeline during `MLflowDown`) is captured — the single-Job
  15-min soak was skipped for cost, not faked.
