# Local Development with Docker Compose

This guide covers the **Docker Compose development workflow** — the fastest way
to get a fully-provisioned development environment without installing Python, the
toolchain, or DVC on your host. If you only want to *run* the production image,
see [containerization.md](containerization.md) and
[ADR-005](decisions/ADR-005-containerization-strategy.md). This document is about
the **inner development loop**.

> **Scope.** Compose here is a local developer convenience, **not** a production
> deployment tool. Production orchestration is deferred to Kubernetes
> (see [roadmap.md](roadmap.md), v4).

---

## Prerequisites

- **Docker Desktop** (or Docker Engine) with the Compose v2 plugin.
  Verify with `docker compose version` — this project is tested with Compose v2.
- That's it. No local Python, virtualenv, or dependency install is required.

---

## The two services

Both services are built from the single multi-stage
[`Dockerfile`](../Dockerfile); Compose just chooses the target.

| Service    | Built from target | Runs as | Default? | Purpose |
|------------|-------------------|---------|----------|---------|
| `dev`      | `development`     | root    | **Yes**  | Long-lived dev container. Your working tree is bind-mounted, so edits are live. You `exec` in and run `make check`, `dvc repro`, etc. |
| `pipeline` | `runtime`         | non-root (`appuser`) | No (profile `pipeline`) | One-shot run of the **production** image executing `dvc repro`. Use it to exercise the real image locally. |

The `pipeline` service is gated behind a Compose **profile**, so a bare
`docker compose up` starts only `dev`.

---

## First-time setup

```bash
# 1. Provide credentials (optional for lint/type/test; required for dvc repro / MLflow)
cp .env.example .env
#    then edit .env and fill in:
#      MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD

# 2. Build and start the dev environment
docker compose up -d

# 3. Open a shell inside it
docker compose exec dev bash
```

You are now in `/app` inside the container with the full toolchain on `PATH`.
Because `./src`, `./tests`, `dvc.yaml`, `params.yaml`, and `.dvc/` are
bind-mounted, anything you edit on the host is immediately visible here — no
rebuild needed for a code change.

Inside the container, the usual entry points work:

```bash
make check      # lint + format-check + typecheck + tests (what CI runs)
make test       # pytest
make lint-fix   # Ruff autofix
dvc repro       # reproduce the pipeline
```

---

## Startup

Start the development environment (builds the image on first run):

```bash
docker compose up -d          # detached; container stays up in the background
docker compose exec dev bash  # attach a shell
```

Notes:
- `docker compose up` (without `-d`) also works and streams the container to your
  terminal, but since `dev` runs `sleep infinity` there is nothing to see —
  `-d` + `exec` is the intended flow.
- The environment is "up" as soon as the container is running; you do your work
  through `exec`, not through the container's own foreground process.

Run the **production** pipeline image once instead of the dev shell:

```bash
docker compose --profile pipeline run --rm pipeline
# runs `dvc repro` in the non-root runtime image, then exits and is removed
```

---

## Shutdown

```bash
docker compose stop           # stop containers, keep them for a fast restart
docker compose down           # stop AND remove containers + the project network
docker compose down -v        # also remove the named pip-cache volume (full reset)
```

Your source, data, models, and logs live on the **host** (bind mounts), so
`down` never deletes your work — it only removes containers and the network. Only
`-v` touches the `pip-cache` volume, and that is just a cache.

---

## Logs

```bash
docker compose logs               # all services
docker compose logs dev           # just the dev container
docker compose logs -f dev        # follow (stream) live
docker compose logs --tail=100 dev
```

The `dev` container itself is mostly quiet (it runs `sleep infinity`); the useful
output is whatever you run via `exec`, which prints straight to your shell. The
`pipeline` service, by contrast, streams the full `dvc repro` / MLflow output —
`docker compose --profile pipeline logs -f pipeline` to follow a batch run.

`LOG_LEVEL` (default `INFO`) is passed into both services; set it in `.env` or
inline, e.g. `LOG_LEVEL=DEBUG docker compose up -d`.

---

## Rebuild

Rebuild when the **image** must change — a dependency edit
(`requirements.txt` / `requirements-dev.txt`) or a `Dockerfile` change. You do
**not** need to rebuild for source edits (they are bind-mounted).

```bash
docker compose build                 # rebuild images using cache
docker compose build --no-cache dev  # force a clean rebuild of one service
docker compose up -d --build         # rebuild then (re)start in one step
```

After changing dependencies:

```bash
docker compose up -d --build dev
docker compose exec dev bash
```

Stamp OCI image metadata into a build via the same build args the Dockerfile
accepts:

```bash
VCS_REF=$(git rev-parse --short HEAD) BUILD_VERSION=1.2.0 docker compose build
```

---

## Troubleshooting

**`docker compose up` fails: "failed to connect to the docker API ... dockerDesktopLinuxEngine".**
Docker Desktop isn't running. Start it, wait for the whale icon to settle, and
retry. Confirm with `docker info`.

**"Permission denied" writing to `data/`, `models/`, or `logs/`.**
The `dev` container runs as **root** by design (so bind-mounted files stay
writable across host/container), so this is rare in `dev`. If you hit it, it is
usually from files created by the non-root `pipeline` service. Fix ownership on
the host, or run the offending command in the `dev` service instead.

**An empty `models/` (or `data/`, `logs/`) directory appeared on the host.**
Expected. Those paths are bind-mounted, so Docker creates the host directory if
it is missing. `models/` is a pipeline output directory and is safe to leave
empty; it fills in when you run `dvc repro`.

**`dvc repro` / MLflow fails with an auth or tracking-URI error.**
Credentials are missing. Ensure `.env` exists (`cp .env.example .env`) and has
valid `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_USERNAME` /
`MLFLOW_TRACKING_PASSWORD`. `.env` is optional for lint/type/test but **required**
for anything that talks to MLflow/DagsHub.

**A tool (e.g. `ruff`) is "not found" after `docker compose exec dev bash`.**
The default `bash` shell is non-login and keeps the image's `PATH`
(`/opt/venv/bin`), so tools resolve. If you launch a **login** shell
(`bash -l`), it re-sources `/etc/profile` and drops that `PATH` — avoid `-l`, or
call the tool via `python -m ruff` / `make lint`.

**Source edits aren't reflected in the container.**
Confirm you edited a bind-mounted path (`src/`, `tests/`, `dvc.yaml`,
`params.yaml`, `.dvc/`). Files outside those mounts (or new top-level files) need
either a matching volume entry in `docker-compose.yml` or an image rebuild.

**Port already in use / stale state.**
`docker compose down` to clear containers and the network, then `up -d` again.
For a completely clean slate including caches: `docker compose down -v`.

**Rebuild didn't pick up new dependencies.**
The dependency layer is cached. Force it:
`docker compose build --no-cache dev`, then `up -d`.

---

## See also

- [containerization.md](containerization.md) — image design, build/run, hardening
- [ADR-005](decisions/ADR-005-containerization-strategy.md) — the containerization decision record
- [roadmap.md](roadmap.md) — where Compose fits (dev-only) and what comes next (CI/CD, Kubernetes)
