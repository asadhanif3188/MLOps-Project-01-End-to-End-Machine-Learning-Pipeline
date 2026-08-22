# Sprint 8 — PR 16 Release-Candidate Validation on Live EKS

> **STATUS: EXECUTED 2026-08-22.** This is the authoritative record of the final
> integrated `provision → prove → destroy` EKS session that validated the **complete
> merged PRs 1–15 system** as a release candidate. Unlike the earlier exploratory
> campaign ([sprint-08-live-eks-evidence.md](sprint-08-live-eks-evidence.md)), which
> *discovered* defects, this session **re-validates the release candidate** and, for
> every controlled failure, drives recovery **using the PR 14 runbooks as the operator
> procedure** — proving the documented operations actually work.
>
> Primary proof objective — the full operational loop, on real EKS:
> healthy → controlled failure → detection → alert → **runbook-driven diagnosis** →
> documented remediation → recovery → alert resolution → final healthy verification.

**Account/region redaction:** AWS account IDs, bucket names, and operator IPs are
redacted (`<ACCOUNT>`, `<DATASET_BUCKET>`, `<ARTIFACT_BUCKET>`, `<IP>`) per the Sprint 7
evidence convention. Image `sha256:` digests are non-sensitive and shown in full.

---

## 1. Release candidate

| Dimension | Value |
|---|---|
| Commit | `f39cc87` (`f39cc87de960c38a72db8d966ac2cbbd464a0cc8`) — PRs 1–15 merged |
| Pipeline image | `<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline:1.6.0` @ `sha256:2f355dc2247d6895a832cea8999be0dc26fed6e03dc2e9919b7a497395a71766` |
| MLflow image | `<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/mlflow-server:0.1.0` @ `sha256:369d0f1fc3444a3a1f91c23218ad59e8ddf5e4dd2458a8e4f56e919bd3d701c7` |

> **Release-build note.** Both images were built + pushed from a working tree whose
> only tracked-file delta from `f39cc87` was `.claude/settings.json` (a local editor
> config, excluded from the Docker build context), so `--allow-dirty` was used and the
> released digests map faithfully to the RC application code. The digest **matched** on
> every running container (§13–14).

## 2. Environment

| Dimension | Value |
|---|---|
| Cluster | `mlops-pipeline-dev-eks`, EKS **v1.35** (node `v1.35.6-eks-b3f9404`) |
| Nodes | **2 × t3.large** (AL2023 x86_64, containerd 2.2.5), private subnets, 2 AZs — both `Ready` |
| Provisioned | 2026-08-22 — `Apply complete! Resources: 65 added, 0 changed, 0 destroyed` |
| API access | private **on**; public **on** scoped to operator `<IP>/32` only (never `0.0.0.0/0`) |
| Auth mode | `API` (access entries); **no `aws-auth` ConfigMap**; creator-admin off |
| Workload identity | EKS Pod Identity (aws-node, ebs-csi, dataset-reader, mlflow-s3) — no static keys |
| CNI role separation | node role carries **no** `AmazonEKS_CNI_Policy`; dedicated CNI role does (H-01) |
| NetworkPolicy | VPC CNI `enableNetworkPolicy=true` — enforcing (canary blocked, §10) |
| Deploy method | digest-pinned (`IMAGE_DIGEST=…` `render-cloud-manifests.sh --apply`) |
| Torn down | 2026-08-22 — see [§18 Cleanup](#18-cleanup) |

Times below are **UTC** (`Z`), taken from Prometheus/Kubernetes telemetry (authoritative
and internally consistent); the workstation's local clock label is not used.

---

## 3. Static / pre-flight results (credential-free, before AWS spend)

Gate order per the release-checklist; AWS was **not** provisioned until these passed.

| Gate | Result |
|---|---|
| PRs 1–15 present in history | ✅ (prometheus-foundation → observability-contracts) |
| `ruff check` / `ruff format --check` | ✅ pass / 39 files formatted |
| `mypy` (strict, `src`) | ✅ no issues, 14 files |
| `pytest` (incl. PR 15 contracts) | ✅ **233 passed, 1 skipped** (mlflow not installed locally) |
| `terraform fmt` / `init` / `validate` | ✅ pass |
| `terraform test` (offline, mocked) | ✅ **41/42**; the 1 non-pass is environmental, not a RC defect† |
| `kustomize build` + `kubeconform` (6 roots) | ✅ all valid (22/30/25/35/35/35) |
| `k8s/validate.py` (local + aws overlays) | ✅ **201/201** each |
| `promtool` / `trivy` / `tflint` / docker-build+SBOM | covered green by CI on this exact `main` (tools installed on demand for the live phases) |

† `terraform test`'s `eks_api_is_private_by_default` asserts the **default** is private;
it auto-loads the git-ignored operator `terraform.tfvars`, which sets
`cluster_endpoint_public_access = true` (the scoped-`/32` opt-in required for workstation
`kubectl`). The **repo default remains `false`** (`variables.tf`), CI carries no tfvars,
and `main` is green — so the secure-by-default contract is intact; the local override is
expected operator state, not a regression.

---

## 4. Healthy baseline

| Check | Result |
|---|---|
| Nodes | 2 × `Ready` |
| Workloads | mlflow, mlflow-postgres, postgres-exporter **Running**; monitoring 7/7 **Running**; **0 Pending, 0 CrashLoop** |
| Pipeline | Job `Complete`, pod `Completed` **exit 0**, all **5/5** stages `success=1`, 5 stage durations recorded |
| MLflow | run persisted (`FINISHED`), metadata in PostgreSQL, artifacts in S3 (`MLmodel`, `model.skops` ~2 MB) |
| Running digests | pipeline (3/3 containers) & MLflow pod match the pushed digests (§13–14) |

## 5. Monitoring evidence

- **8 scrape jobs / 11 target instances all UP:** `prometheus`, `kube-state-metrics`,
  `node-exporter` (×2), `kubernetes-cadvisor` (×2), `kubelet` (×2), `pushgateway`,
  `postgres-exporter`, `blackbox-mlflow-health` (`count(up==1) = 11`, matching §15).
- `pg_up = 1`; `probe_success{blackbox-mlflow-health} = 1`.
- **8 alert rules loaded:** `PipelineJobFailed`, `PipelineJobOOMKilled`, `MLflowDown`,
  `MLflowMemoryHigh`, `PostgresDown`, `PostgresMemoryHigh`, `PostgresPVCAlmostFull`,
  `KubePodCrashLooping`. **0 unexpected active alerts** at baseline.
- Pipeline operational metrics (`mlops_pipeline_stage_success` / `_duration_seconds`)
  present in Prometheus via the Pushgateway.

> **Deploy-order note (non-blocking).** The first pipeline Job (created by the initial
> `mlops` overlay apply) ran **before** the monitoring stack existed, so its metric push
> had no Pushgateway target. Re-running the pipeline after the observability stack was up
> produced the full metric set. Deploying monitoring **before** the first pipeline run
> avoids the transient gap; recorded as an operational observation.

## 6. Dashboard evidence (headless / API)

Grafana `database: ok`, v11.2.0. All **3 dashboards** served —
`mlops-eks-platform-health` (EKS / Platform Health), `mlops-pipeline-operations`
(MLOps Pipeline Operations), `mlops-mlflow-platform-health` (MLflow Platform Health).
The Grafana→Prometheus datasource proxy returned live data (`count(up==1) = 11`),
proving the dashboards' data path. *(Headless session: dashboard data verified via the
Grafana HTTP API, not screenshots — see [§19 Limitations](#19-remaining-limitations).)*

---

## 7. Dataset failure / recovery — **PASS**

Runbook: [`dataset-retrieval-failure.md`](../runbooks/dataset-retrieval-failure.md) ·
alert `PipelineJobFailed` (2m). Mechanism: Scenario A — object **unavailable**.

| Step | Evidence |
|---|---|
| Inject `06:56:49Z` | `aws s3 rm <dataset>` then re-drove the real Job |
| Symptom | 3 pods `Init:Error`; `fetch-dataset` exit 1: **`An error occurred (404) … HeadObject … Not Found`**; stage `fetch_dataset success=False`; Job `Failed` / `BackoffLimitExceeded` |
| Runbook detection | `mlops_pipeline_stage_success{stage="fetch_dataset"} = 0` (the documented discriminator) ✓ |
| Runbook diagnosis | printed `DATASET_S3_URI`; `aws s3 ls` **empty** → **"Object missing"** cause — correct ✓ |
| Alert | `PipelineJobFailed` **pending 07:00:02Z → FIRING 07:00:18Z** |
| Runbook remediation | re-upload `aws s3 cp … --sse aws:kms --sse-kms-key-id …`, then delete+re-apply Job |
| Recovery | 5/5 stages `=1`; Job `Complete` `succeeded=1`; `fetch-dataset`: *"Dataset retrieved … (23872 bytes); verifying integrity"* |
| Alert resolution | `PipelineJobFailed` firing → **0 by 07:04:22Z** |

Runbook guided diagnosis and recovery with **no undocumented knowledge**; no changes required.

## 8. MLflow outage / recovery — **PASS**

Runbook: [`mlflow-unavailable.md`](../runbooks/mlflow-unavailable.md) · alert `MLflowDown`
(5m). Mechanism: scale the **stateless** Deployment to 0 (PostgreSQL StatefulSet, PVC, S3
untouched).

| Step | Evidence |
|---|---|
| Baseline | `probe_success=1`, `pg_up=1`, **runs_before=6** |
| Inject `07:06:09Z` | `kubectl scale deploy/mlflow --replicas=0` |
| Runbook detection | `probe_success=0`; deploy `0/0`; endpoints empty; **`pg_up=1` → runbook rules OUT PostgreSQL**, points at MLflow itself — correct ✓ |
| Alert | `MLflowDown` **pending 07:07:41Z → FIRING 07:11:33Z** |
| Runbook remediation `07:12:00Z` | `scale --replicas=1` + `rollout status` |
| Recovery | `probe_success=1` @ `07:12:57Z`; endpoints back (`10.0.8.82:5000`) |
| Alert resolution | `MLflowDown` firing → **0 @ 07:13:00Z** |
| Persistence | **runs_after=6 (= before)**; prior run still `FINISHED` + `artifact_uri` intact → PostgreSQL metadata + S3 artifacts survived |

**PR 13 bounded-retry (reliability hardening).** With MLflow held down (07:19:34→07:20:58Z),
the pipeline's `wait-for-mlflow` init container logged bounded retries
**`[1/60]…[16/60]`** (`Connection refused` / `timed out`), then **`MLflow ready`** on
restore → pipeline **exit 0**. Confirms the retry is *bounded* (max 60) and *recovers*.

> **Methodological note.** `render-cloud-manifests.sh --apply` re-applies the MLflow
> Deployment (`replicas: 1`), so to hold MLflow down you must `scale` **after** the
> apply, not before — otherwise the apply silently restores it. Recorded so the runbook's
> outage procedure is reproducible.

## 9. OOM / resource failure / recovery — **PASS**

Runbook: [`oomkilled.md`](../runbooks/oomkilled.md) · alert `PipelineJobOOMKilled` (2m).
Mechanism: the documented low-memory override applied to the **real** Job (128Mi limit /
64Mi request) so the real alert fires and the runbook's `job mlops-pipeline` selectors
apply.

| Step | Evidence |
|---|---|
| Inject `07:28:41Z` | real Job patched to 128Mi mem limit; pipeline OOMs in `train` |
| Symptom | 3 pods `Failed`, **`reason=OOMKilled` exit=137** (the EKS-specific reason) |
| Runbook detection | `kube_pod_container_status_terminated_reason{reason="OOMKilled"} = 1` for all 3 pods |
| Alert | `PipelineJobOOMKilled` `activeAt 07:29:40Z` → **FIRING** (~07:31:40Z) |
| Runbook remediation | delete + `render-cloud-manifests.sh --apply` → restores normal **512Mi/256Mi** (no temp config remains) |
| Recovery | pod `Completed` exit 0; 5/5 stages `=1` |
| Alert resolution | `PipelineJobOOMKilled` firing → **0 @ 07:44:46Z** |

> **Instrument-integrity note (non-blocking, no platform defect).** Mid-phase, the
> Prometheus **port-forward silently dropped**, making the alert/metric queries briefly
> return blank — which *looked* like "the alert never fired." Diagnosis proved the
> opposite: **kube-state-metrics was emitting** `kube_pod_container_status_terminated_reason{reason="OOMKilled"} 1`
> for all 3 pods (1388 `kube_pod_` series raw), and a **fresh port-forward** showed the
> metric present and `PipelineJobOOMKilled` **firing**. Lesson: health-check / refresh
> the port-forward before trusting a "no data" PromQL result. The monitoring pipeline
> itself was correct throughout.

## 10. NetworkPolicy allow / deny — **PASS**

Harness: [`k8s/tests/netpol/run.sh`](../../k8s/tests/netpol/run.sh) (enforcing VPC CNI,
judged on TCP-connect). Result: **9 passed, 0 failed, 0 inconclusive.**

- **Enforcement canary:** unlabelled → PostgreSQL:5432 **BLOCKED** (CNI enforcing).
- **Allowed 6/6:** pipeline→MLflow:5000, pipeline→Pushgateway:9091, MLflow→PostgreSQL:5432,
  postgres-exporter→PostgreSQL:5432, Prometheus→postgres-exporter:9187, blackbox→MLflow:5000.
- **Denied 3/3:** pipeline→PostgreSQL:5432 (no MLflow bypass), unlabelled→MLflow:5000,
  unlabelled→PostgreSQL:5432.
- **Pod Identity + DNS + S3 egress** independently proven by the successful S3 dataset
  fetch every healthy run (requires `allow-pod-identity-egress`, DNS, and S3 egress
  policies to all function together).

## 11. Alert firing / resolution proof

| Alert | For | Fired (UTC) | Resolved (UTC) |
|---|---|---|---|
| `PipelineJobFailed` | 2m | 07:00:18Z | 07:04:22Z |
| `MLflowDown` | 5m | 07:11:33Z | 07:13:00Z |
| `PipelineJobOOMKilled` | 2m | ~07:31:40Z (`activeAt 07:29:40Z`) | 07:44:46Z |

Every mandatory alert transitioned **inactive → pending → firing → resolved** on the real
platform. Final state: **0 firing alerts** (§15).

## 12. Runbook validation matrix (Phase 9)

The target claim — *"the operational runbooks were exercised against controlled failures
on the real EKS platform and successfully guided diagnosis and recovery"* — is supported:

| Runbook | Scenario | Symptom match | Detection cmd | Diagnostic cmd | Diagnosis reached | Remediation | Recovery-verify | Changes | Status |
|---|---|---|---|---|---|---|---|---|---|
| `dataset-retrieval-failure.md` | missing S3 object → 404 | yes | yes | yes | yes ("object missing") | yes (re-upload) | yes | none | **PASS** |
| `mlflow-unavailable.md` | Deployment scaled to 0 | yes | yes | yes | yes (pg_up=1 → MLflow) | yes (scale=1) | yes | none | **PASS** |
| `oomkilled.md` | 128Mi limit → OOMKilled | yes | yes | yes | yes (limit too low) | yes (restore 512Mi) | yes | none | **PASS** |

All three exercised runbooks guided diagnosis and recovery **without undocumented
knowledge**; none required correction.

## 13. Pipeline image provenance

`git f39cc87` → tag `1.6.0` → **ECR digest `sha256:2f355dc2…71766`** (cross-checked at
push via `aws ecr describe-images`) → **running pod `imageID`** = same digest on **all 3
containers** (`fetch-dataset`, `wait-for-mlflow`, `pipeline`) per
`verify-deployed-digest.sh` → **PASS**. CycloneDX SBOM: **321 components**. Vulnerability
gate (`trivy --ignore-unfixed --severity HIGH,CRITICAL --ignorefile .trivyignore.yaml`):
**exit 0** (no fixable HIGH/CRITICAL).

## 14. MLflow image digest verification

`git f39cc87` → tag `0.1.0` → **ECR digest `sha256:369d0f1f…701c7`** → **running MLflow
pod `imageID`** = same digest → verified. **Closes the minor gap** from the earlier
campaign, where the MLflow running digest was not independently re-captured. SBOM: **322
components**; vuln gate **exit 0**.

## 15. Final healthy pipeline + platform state

| Check | Result |
|---|---|
| Final pipeline | Job `Complete`, pod `Completed` **exit 0** |
| Targets | **11 UP** (`up==1`) |
| Alerts | **0 firing** — every failure alert resolved, none lingering |
| Pipeline | 5/5 stages `=1` |
| MLflow | `probe_success=1`; **16 runs** persisted (final run stored) |
| PostgreSQL | `pg_up=1`; StatefulSet Running, 0 restarts — state intact across all phases |
| Workloads | all Running, **0 restarts**; monitoring 0 restarts |
| Network | required communication functioning (final run's S3 fetch + MLflow log succeeded) |

The failure campaign left the platform **fully healthy** — no residual degradation.

## 16. Reliability-hardening validation (PR 13)

Bounded `wait-for-mlflow` retry observed live tolerating a sustained MLflow outage
(`[1/60]…[16/60]`) and recovering on restore (pipeline exit 0) — see §8.

## 17. Cost / resource observations

- **Nodes:** 2 × t3.large (2 vCPU / 8 GiB each), 2 AZs; EKS v1.35.6.
- **Observability components: 7** — Prometheus, Grafana, kube-state-metrics,
  blackbox-exporter, Pushgateway (monitoring ns); node-exporter DaemonSet (×2 nodes);
  postgres-exporter (mlops ns).
- **Observability requests (approx):** CPU ~240m, memory ~590Mi (Prometheus 100m/256Mi
  the largest single).
- **App requests:** mlflow-server 250m/1Gi, mlflow-postgres 100m/256Mi,
  mlops-pipeline 250m/256Mi, postgres-exporter 10m/32Mi.
- **Storage / retention:** Prometheus TSDB 7d / 1GB; PostgreSQL PVC 1Gi (EBS gp3).
- **Incremental cost drivers (short-lived):** EKS control plane, 2× t3.large EC2,
  1× NAT gateway, EBS volumes, ECR storage, 3× customer-managed KMS keys.
- **Session:** single provision→validate→destroy, ~1.5 h of active validation.

*No precise long-term TCO is implied from a short-lived validation session.*

## 18. Cleanup

Documented teardown ([`docs/cloud-operations.md` §5](../cloud-operations.md)):
`kubectl delete -k k8s/monitoring/overlays/aws` → delete `mlops`/`monitoring` namespaces
→ `terraform destroy`.

| Check | Result |
|---|---|
| `terraform destroy` | **`Destroy complete! Resources: 65 destroyed`** (symmetric with the 65 added — no orphans) |
| Terraform state | *"The state file is empty. No resources are represented."* |
| EKS clusters | none (`aws eks list-clusters` empty) |
| NAT gateways (available) | none |
| ECR `mlops-pipeline` | `RepositoryNotFoundException` (gone; `force_delete` removed both repos) |
| Unattached Elastic IPs | none |

> **Teardown was clean this session** — unlike the earlier exploratory campaign
> ([live-EKS §8](sprint-08-live-eks-evidence.md)), the VPC destroyed **without** a
> lingering VPC-CNI ENI or `eks-cluster-sg-*` `DependencyViolation`, so **no manual ENI/SG
> deletion or `terraform state rm` was needed**. The three customer-managed **KMS keys**
> enter a **7-day `PendingDeletion`** window (`kms_key_deletion_window_days = 7`, the
> minimum) and then auto-delete — **expected, not a leak and not ongoing compute billing**.

## 19. Remaining limitations

- **Dashboards verified via Grafana/Prometheus HTTP API, not screenshots** — this was a
  headless CLI session. Panel data paths are proven (§6); pixel rendering is not captured.
- **`terraform test` local 41/42** is an operator-tfvars artifact, not a RC defect (§3†).
- **Single-operator, short-lived validation** — no claim of 24/7 SRE, SLA/SLO compliance,
  multi-region/HA/DR, GitOps, remote Terraform state, service mesh, distributed tracing,
  or centralized logging beyond what is implemented. Terraform state is local; the KMS
  keys enter a **7-day `PendingDeletion`** window on destroy (by design, not a leak).
- **Alerting is Prometheus-native** (no Alertmanager) — "firing" is proven via
  `ALERTS{alertstate="firing"}` and `/api/v1/alerts`, not paging.

---

## Verdict

**PASS.** On live Amazon EKS, the merged PRs 1–15 release candidate demonstrated the full
loop for every mandatory scenario — healthy → controlled failure → detection → alert →
**runbook-driven diagnosis** → documented remediation → recovery → alert resolution →
final healthy verification — with all three exercised PR 14 runbooks guiding diagnosis and
recovery without correction, image provenance verified end-to-end for both images, and the
platform returned to a fully healthy final state. **No release blockers found.**
