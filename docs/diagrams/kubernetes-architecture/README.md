# Kubernetes Architecture Diagram

Source for the Kubernetes batch-workload architecture. Rendered and discussed in
[docs/kubernetes-architecture.md](../../kubernetes-architecture.md); the design of
record is [ADR-009](../../decisions/ADR-009-kubernetes-workload-model.md).

> **Status.** Reflects Sprint 5 through PR 2: the `mlops` namespace and the
> **runnable** `batch/v1` Job (real image + `dvc repro` + finite-run lifecycle).
> Objects marked *(deferred)* are design contracts implemented in later PRs
> (config/secrets, security, resources).

## Workload flow

```mermaid
flowchart TD
    dev["Developer<br/><i>kubectl apply -k k8s/overlays/local</i>"]

    subgraph cluster["Kubernetes cluster (local: kind / minikube)"]
        subgraph ns["Namespace: mlops"]
            job["Job: mlops-pipeline<br/><i>batch/v1 · restartPolicy: Never · backoffLimit: 2 · activeDeadlineSeconds: 1800</i>"]
            pod["Pod (one attempt)"]
            container["ML container<br/><i>ml-pipeline:local · CMD: dvc repro</i>"]

            cfg["ConfigMap / Secret / ServiceAccount<br/><i>(deferred — PR 3)</i>"]

            job --> pod --> container
            cfg -. "injected (PR 3)" .-> container

            subgraph pipeline["DVC pipeline (dvc repro)"]
                direction TB
                pre["preprocess"] --> split["split"] --> train["train"] --> evaluate["evaluate"]
            end
            container --> pipeline

            artifacts["Artifacts + metrics<br/><i>models/model.pkl · metrics/metrics.json</i>"]
            evaluate --> artifacts
        end

        completion(["Job completion<br/><i>pod exits 0 → Job Complete</i>"])
        artifacts --> completion
    end

    dev --> job

    classDef boundary fill:#eef,stroke:#557,stroke-width:1px;
    classDef deferred fill:#f7f7f7,stroke:#aaa,stroke-dasharray:4 3,color:#666;
    class cluster,ns boundary;
    class cfg deferred;
```

## ASCII fallback

```text
Developer ── kubectl apply ─▶ Kubernetes cluster
                                    │
                             Namespace: mlops
                                    │
                              Job: mlops-pipeline
                                    │
                                   Pod
                                    │
                              ML container (dvc repro)
                                    │
              preprocess ─▶ split ─▶ train ─▶ evaluate
                                                  │
                                     artifacts / metrics
                                                  │
                                          Job completion (exit 0)
```
