#!/usr/bin/env bash
# Runtime NetworkPolicy verification (Sprint 8, PR 7; ADR-034).
#
# Proves the LIVE least-privilege network paths on a running cluster: that every
# path the workloads need is ALLOWED and that an intentionally prohibited path is
# DENIED. This is the runtime counterpart to the static contract in
# k8s/validate.py (Section 8 / M12) — the static pass proves the policy SET encodes
# least privilege; this proves a CNI actually enforces it.
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS. The policies select workloads by their `app.kubernetes.io/name`
# label, so this suite launches tiny, hardened `curl` probe pods carrying the SAME
# labels the real workloads use. A probe therefore exercises the IDENTICAL policy
# selectors the real pod would — a pod labelled `mlops-pipeline` is bound by the
# pipeline's egress policy, an unlabelled pod is bound only by the default-deny,
# and so on. Each probe attempts one TCP/HTTP connection with a short timeout; the
# result (connect vs. time-out) is the evidence.
#
# ENFORCEMENT REQUIRED. A NetworkPolicy is inert unless the CNI enforces it. This
# script first runs a CANARY: a known-denied connection. If the canary SUCCEEDS,
# the cluster is NOT enforcing NetworkPolicy (e.g. the default Docker Desktop CNI,
# or an EKS cluster without the VPC CNI `enableNetworkPolicy=true` flag). In that
# case the DENIED-path assertions cannot be trusted, so the script reports the
# environment as NON-ENFORCING and marks the denied checks INCONCLUSIVE rather than
# emitting a false PASS. Allowed-path checks still run (they prove the wiring works
# and, crucially, that the policies did not break a needed path).
#
# PREREQUISITES: a reachable cluster (kubectl configured) with the mlops workloads
# deployed and Ready, an enforcing CNI for the denied-path evidence to be
# meaningful (EKS with VPC CNI network policy, or a local kind cluster with Calico/
# Cilium), and the NetworkPolicies applied (they are part of the kustomize overlay).
#
# USAGE:
#   k8s/tests/netpol/run.sh                 # against the current kube-context
#   NAMESPACE=mlops k8s/tests/netpol/run.sh
# Exit 0 = all allowed paths worked AND (denied paths blocked OR non-enforcing CNI
# clearly reported). Exit 1 = a required path was blocked, or a denied path was
# open on an ENFORCING cluster (a real least-privilege regression).
set -uo pipefail

NS="${NAMESPACE:-mlops}"
MON_NS="${MONITORING_NAMESPACE:-monitoring}"
# curlimages/curl runs as a non-root user and has curl (HTTP + telnet://) built in,
# so a probe satisfies the `restricted` Pod Security Standard the mlops namespace
# enforces. Pin a digest-free but explicit tag for reproducibility.
PROBE_IMAGE="${PROBE_IMAGE:-curlimages/curl:8.10.1}"
TIMEOUT="${PROBE_TIMEOUT:-6}"

pass=0
fail=0
inconclusive=0

log() { printf '%s\n' "$*"; }
ok() {
  pass=$((pass + 1))
  log "  [PASS] $1"
}
bad() {
  fail=$((fail + 1))
  log "  [FAIL] $1${2:+ -> $2}"
}
skip() {
  inconclusive=$((inconclusive + 1))
  log "  [INCONCLUSIVE] $1${2:+ -> $2}"
}

# Launch a hardened probe pod with a given name + label set, then keep it alive.
# Args: pod-name namespace label(app.kubernetes.io/name value, or "none")
start_probe() {
  local name="$1" ns="$2" appname="$3"
  local label_args=()
  if [ "$appname" != "none" ]; then
    label_args=(--labels "app.kubernetes.io/name=$appname")
  fi
  # Expand the array with the `${arr[@]+"${arr[@]}"}` guard so a ZERO-element array
  # (the unlabelled probe) does not trip `set -u` on bash < 4.4 (e.g. macOS's
  # default /bin/bash 3.2), which this suite's local-dev audience may run on.
  kubectl -n "$ns" run "$name" --image="$PROBE_IMAGE" \
    ${label_args[@]+"${label_args[@]}"} \
    --restart=Never \
    --overrides='{
      "spec": {
        "containers": [{
          "name": "'"$name"'",
          "image": "'"$PROBE_IMAGE"'",
          "command": ["sleep", "600"],
          "securityContext": {
            "allowPrivilegeEscalation": false,
            "runAsNonRoot": true,
            "runAsUser": 65534,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"}
          }
        }]
      }
    }' >/dev/null 2>&1 || true
  kubectl -n "$ns" wait --for=condition=Ready "pod/$name" --timeout=60s >/dev/null 2>&1
}

# Run one connection attempt from a probe pod. Returns 0 on connect, non-zero on
# blocked/timeout. `spec` is a curl URL (http://host:port/path or telnet://host:port).
#
# Success is judged on whether the TCP connection was ESTABLISHED (curl's
# %{time_connect} > 0), NOT on curl's process exit code. In some hardened
# environments (curl in a restricted, drop-ALL container invoked via `kubectl exec`
# on EKS/containerd) curl completes the request but exits non-zero on the response
# write phase (e.g. exit 23 "client returned ERROR on write"), even though the
# connection plainly succeeded (the HTTP status is 200). Relying on the exit code
# there reports every ALLOWED path as blocked — a false FAIL. time_connect is set
# once the TCP handshake completes, so it is 0 for a policy-blocked path (SYN
# dropped/refused → no connect) and > 0 for an allowed one, for both http:// and
# telnet:// probes.
probe() {
  local pod="$1" ns="$2" url="$3" tc
  tc="$(kubectl -n "$ns" exec "$pod" -- \
    curl -s -o /dev/null --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" \
    -w '%{time_connect}' "$url" 2>/dev/null)"
  awk -v t="$tc" 'BEGIN { exit !((t + 0) > 0) }'
}

# Assert an ALLOWED path: the connection MUST succeed.
assert_allow() {
  local desc="$1" pod="$2" ns="$3" url="$4"
  if probe "$pod" "$ns" "$url"; then
    ok "ALLOW  $desc"
  else
    bad "ALLOW  $desc" "connection failed but the policy should permit it"
  fi
}

# Assert a DENIED path: the connection MUST fail — but only trustworthy on an
# enforcing CNI (see the canary below).
assert_deny() {
  local desc="$1" pod="$2" ns="$3" url="$4"
  if [ "${ENFORCING:-unknown}" != "yes" ]; then
    skip "DENY   $desc" "CNI is not enforcing NetworkPolicy (see canary)"
    return
  fi
  if probe "$pod" "$ns" "$url"; then
    bad "DENY   $desc" "connection SUCCEEDED but the policy should block it"
  else
    ok "DENY   $desc"
  fi
}

cleanup() {
  kubectl -n "$NS" delete pod np-probe-pipeline np-probe-mlflow np-probe-exporter \
    np-probe-none --ignore-not-found --wait=false >/dev/null 2>&1 || true
  kubectl -n "$MON_NS" delete pod np-probe-prometheus np-probe-blackbox \
    --ignore-not-found --wait=false >/dev/null 2>&1 || true
}
trap cleanup EXIT

command -v kubectl >/dev/null 2>&1 || {
  log "ERROR: kubectl not found on PATH."
  exit 2
}
kubectl cluster-info >/dev/null 2>&1 || {
  log "ERROR: no reachable cluster (kubectl cluster-info failed)."
  exit 2
}

log "NetworkPolicy runtime verification (namespace: $NS / $MON_NS)"
log ""
log "Launching hardened probe pods (labelled as the real workloads)..."
start_probe np-probe-pipeline "$NS" "mlops-pipeline"
start_probe np-probe-mlflow "$NS" "mlflow-server"
start_probe np-probe-exporter "$NS" "postgres-exporter"
start_probe np-probe-none "$NS" "none"
start_probe np-probe-prometheus "$MON_NS" "prometheus"
start_probe np-probe-blackbox "$MON_NS" "blackbox-exporter"

# ─────────────────────────────────────────────────────────────────────────────
# Enforcement canary. The unlabelled probe has NO policy allowing it to reach
# PostgreSQL, so on an ENFORCING cluster this MUST be blocked. If it connects, the
# CNI is not enforcing NetworkPolicy and denied-path evidence is not trustworthy.
log ""
log "Enforcement canary (unlabelled pod -> PostgreSQL:5432, must be blocked)..."
if probe np-probe-none "$NS" "telnet://mlflow-postgres:5432"; then
  ENFORCING="no"
  log "  [WARN] canary CONNECTED — this CNI does NOT enforce NetworkPolicy."
  log "         Denied-path checks will be reported INCONCLUSIVE (no false PASS)."
  log "         For real denied-path evidence use EKS with the VPC CNI"
  log "         enableNetworkPolicy flag, or a local kind cluster with Calico/Cilium."
else
  ENFORCING="yes"
  log "  [ok] canary blocked — the CNI is enforcing NetworkPolicy."
fi

# ─────────────────────────────────────────────────────────────────────────────
log ""
log "ALLOWED paths (must all succeed — the workloads' required paths):"
assert_allow "pipeline -> MLflow:5000" \
  np-probe-pipeline "$NS" "http://mlflow.$NS.svc.cluster.local:5000/health"
assert_allow "pipeline -> Pushgateway:9091 (cross-ns push)" \
  np-probe-pipeline "$NS" "http://pushgateway.$MON_NS.svc.cluster.local:9091/-/healthy"
assert_allow "mlflow -> PostgreSQL:5432" \
  np-probe-mlflow "$NS" "telnet://mlflow-postgres.$NS.svc.cluster.local:5432"
assert_allow "postgres-exporter -> PostgreSQL:5432" \
  np-probe-exporter "$NS" "telnet://mlflow-postgres.$NS.svc.cluster.local:5432"
assert_allow "Prometheus -> postgres-exporter:9187 (scrape preserved)" \
  np-probe-prometheus "$MON_NS" "http://postgres-exporter.$NS.svc.cluster.local:9187/"
assert_allow "blackbox -> MLflow:5000 (health probe preserved)" \
  np-probe-blackbox "$MON_NS" "http://mlflow.$NS.svc.cluster.local:5000/health"

# ─────────────────────────────────────────────────────────────────────────────
log ""
log "DENIED paths (must all be blocked on an enforcing CNI):"
# The pipeline may reach MLflow but NEVER the DB directly.
assert_deny "pipeline -> PostgreSQL:5432 (must not bypass MLflow)" \
  np-probe-pipeline "$NS" "telnet://mlflow-postgres.$NS.svc.cluster.local:5432"
# An unlabelled pod matches only the default-deny — no allow applies.
assert_deny "unlabelled pod -> MLflow:5000 (default-deny)" \
  np-probe-none "$NS" "http://mlflow.$NS.svc.cluster.local:5000/health"
# An unlabelled pod may not reach the DB either (this is also the canary).
assert_deny "unlabelled pod -> PostgreSQL:5432 (default-deny)" \
  np-probe-none "$NS" "telnet://mlflow-postgres.$NS.svc.cluster.local:5432"

# ─────────────────────────────────────────────────────────────────────────────
log ""
log "Summary: ${pass} passed, ${fail} failed, ${inconclusive} inconclusive."
if [ "$fail" -ne 0 ]; then
  log "RESULT: FAIL — a required path was blocked or a denied path was open."
  exit 1
fi
if [ "${ENFORCING:-no}" != "yes" ]; then
  log "RESULT: PARTIAL — allowed paths verified; denied paths INCONCLUSIVE"
  log "        (non-enforcing CNI). Re-run on an enforcing cluster for full proof."
  exit 0
fi
log "RESULT: PASS — all allowed paths work and all denied paths are blocked."
exit 0
