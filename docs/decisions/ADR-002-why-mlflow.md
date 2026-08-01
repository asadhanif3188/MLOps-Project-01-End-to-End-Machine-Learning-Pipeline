# ADR-002: Why MLflow (via DagsHub) for Experiment Tracking

- **Status:** Accepted
- **Date:** 2026-08-01
- **Deciders:** Asad Hanif
- **Related:** [ADR-003 (DVC)](ADR-003-why-dvc.md), [architecture.md](../architecture.md)

## Context

The pipeline tunes a Random Forest with `GridSearchCV`, producing many candidate
models across hyperparameter combinations. To make the workflow credible and
reproducible, we need to:

- record parameters, metrics (e.g., `accuracy`), and artifacts (confusion matrix,
  classification report, the model itself) per run,
- compare runs to select the best model,
- optionally maintain a model registry, and
- do the above via a **remote** tracker so results are shareable and not tied to
  one machine.

## Decision

The project uses **MLflow** for experiment tracking, with the tracking server
**hosted on DagsHub**.

- `train.py` and `evaluate.py` set the tracking URI from the
  `MLFLOW_TRACKING_URI` environment variable and log parameters, metrics, and
  artifacts within an MLflow run.
- Credentials (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
  `MLFLOW_TRACKING_PASSWORD`) are supplied via `.env` (see
  [`.env.example`](../../.env.example)) and loaded with `python-dotenv`.
- When the artifact store is remote, the best model is also registered
  (`registered_model_name="Best Random Forest Classifier"`).

DagsHub hosts the tracker because it provides a **managed MLflow endpoint plus a
DVC-compatible remote in one place**, keeping the tracking and data-versioning
stack cohesive (see [ADR-003](ADR-003-why-dvc.md)).

## Alternatives Considered

1. **Local MLflow tracking (`file:` store / local server).**
   - *Pros:* zero external setup.
   - *Cons:* not shareable, no managed UI, harder to demonstrate in a portfolio.
2. **Weights & Biases.**
   - *Pros:* polished UX, strong collaboration features.
   - *Cons:* another SaaS account; MLflow is open-source and pairs naturally with
     DagsHub/DVC here.
3. **Manual logging (CSV/JSON + custom scripts).**
   - *Decision:* rejected — reinvents a solved problem with poor comparison/UX.
4. **Neptune / Comet.**
   - *Decision:* deferred — viable, but no advantage over MLflow for this
     project's needs.

## Consequences

**Positive**

- Standardized, queryable record of every run (params, metrics, artifacts).
- Shareable, hosted UI suitable for a portfolio demonstration.
- Optional model registry for promoting the best model.
- Cohesive stack: MLflow and DVC both hosted on DagsHub.

**Trade-offs and follow-ups**

- Runtime dependency on network access and valid DagsHub credentials; runs fail
  fast if `MLFLOW_TRACKING_URI` is unset (accessed via `os.environ[...]`). A
  local-MLflow fallback for offline development is planned in Roadmap v2.
- Vendor coupling to DagsHub for the hosted endpoint.
- Secrets management relies on `.env` hygiene; real credentials are never
  committed.
