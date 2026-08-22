# Final Platform Architecture

**Title.** Final Platform Architecture — commit to observed run on AWS EKS.

**Purpose.** One screen that shows how a Git commit becomes a verified pipeline run
on real Amazon EKS, and how that run is observed — with the CI / operator /
Terraform / runtime **ownership boundaries** drawn explicitly.

Design of record: [architecture.md](../../architecture.md); the cloud platform is
[ADR-014](../../decisions/ADR-014-terraform-architecture.md) …
[ADR-027](../../decisions/ADR-027-s3-dataset-runtime-retrieval.md).

> **Status.** ✅ Reflects the platform validated on live EKS in Sprint 8
> (v1.7.0). It is a **short-lived validation environment**, not a production
> deployment — see *Limitations* below.

## Diagram

```mermaid
flowchart TB
    subgraph gh["GitHub — CI (validate only, credential-free)"]
        ci["GitHub Actions<br/><i>lint · test · DVC integrity · image build · SBOM</i><br/><b>never pushes · never deploys</b>"]
    end

    subgraph op["Operator — from own AWS account"]
        build["docker build<br/><i>multi-stage · non-root runtime</i>"]
        tf["terraform apply<br/><i>~65 managed resources</i>"]
        kap["kubectl apply -k overlays/aws"]
    end

    subgraph aws["AWS — Terraform-managed infrastructure"]
        ecr["ECR<br/><i>immutable tags · scan-on-push</i>"]
        kms["KMS CMKs ×3<br/><i>EKS secrets · dataset · artifacts</i>"]
        s3d[("S3 — dataset bucket<br/><i>SSE-KMS · versioned · private</i>")]
        s3a[("S3 — MLflow artifacts<br/><i>SSE-KMS · versioned · private</i>")]

        subgraph eks["EKS cluster · K8s 1.35 · 2× t3.large · 2 AZs"]
            subgraph mlops["namespace: mlops"]
                job["Job: mlops-pipeline<br/><i>batch/v1 · preprocess→split→train→evaluate</i>"]
                mlflow["MLflow server<br/><i>Deployment · --serve-artifacts</i>"]
                pg[("PostgreSQL<br/><i>StatefulSet · 1Gi PVC</i>")]
            end
            subgraph mon["namespace: monitoring"]
                prom["Prometheus<br/><i>8 alert rules</i>"]
                graf["Grafana<br/><i>4-layer dashboards</i>"]
            end
        end
    end

    ci -. "gates every PR" .-> build
    build --> ecr
    tf --> aws
    kap --> job

    ecr -- "image pull (node role)" --> job
    s3d -- "dataset · Pod Identity · read-only" --> job
    job -- "params / metrics / artifacts" --> mlflow
    mlflow -- "run metadata" --> pg
    mlflow -- "artifact bytes (SSE-KMS)" --> s3a
    prom -- "scrape operational metrics" --> mlops
    graf -- PromQL --> prom

    classDef boundary fill:#eef,stroke:#557,stroke-width:1px;
    classDef store fill:#eefaf0,stroke:#2e7d5b;
    class gh,op,aws,eks,mlops,mon boundary;
    class s3d,s3a,pg store;
```

**What it proves / helps explain.**

- The **ownership split** a reviewer needs first: CI only *validates* (it never
  holds AWS credentials, never pushes an image, never deploys); the **operator**
  builds, pushes, and applies from their own account; **Terraform** owns the
  infrastructure; the **cluster** owns the runtime.
- The exact runtime data path: pipeline **image** from ECR, **dataset** from S3 by
  Pod Identity (no static keys), tracking to **MLflow → PostgreSQL** (metadata) and
  **S3** (artifacts, SSE-KMS), with **Prometheus/Grafana** observing operational
  signals.

**Limitations (deliberately not shown, because not built).** Single region, single
node group, local Terraform state, ephemeral `provision → prove → destroy` lifecycle
— **no** GitOps/ArgoCD, **no** service mesh, **no** remote state, **no** model
serving, **no** HA/DR, **no** distributed tracing. These are recorded in
[case-study.md § 15](../../case-study.md) and [roadmap.md](../../roadmap.md), not
implied here.

## ASCII fallback

```text
GitHub CI (validate only) ┈gates┈▶ Operator (own AWS account)
                                     │ build──▶ ECR ──image──┐
                                     │ terraform apply       │
                                     │ kubectl apply         ▼
   ┌─────────────────────── AWS (Terraform-managed) ───────────────────────┐
   │  ECR   KMS×3   S3(dataset, SSE-KMS)   S3(artifacts, SSE-KMS)           │
   │                    │Pod Identity            ▲                          │
   │   ┌──── EKS (1.35 · 2× t3.large · 2 AZs) ───┼──────────────┐          │
   │   │ ns mlops:  Job(preprocess→split→train→evaluate)         │          │
   │   │            └─▶ MLflow ──metadata──▶ PostgreSQL          │          │
   │   │                     └─────────artifacts────────────────▶          │
   │   │ ns monitoring:  Prometheus ◀─scrape─ mlops   Grafana─PromQL─▶Prom │
   │   └────────────────────────────────────────────────────────┘          │
   └───────────────────────────────────────────────────────────────────────┘
```
