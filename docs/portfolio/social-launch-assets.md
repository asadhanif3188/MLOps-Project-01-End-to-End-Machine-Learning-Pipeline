<!--
Social launch assets for the MLOps Platform on AWS EKS project.

Public distribution copy — LinkedIn launch post, a follow-up technical series,
a shorter Facebook/GKMBA version, a posting sequence, hashtag guidance, and
claim guardrails. Every claim is grounded in the repository's evidence
(docs/case-study.md, docs/proof/README.md); the claim guardrails at the bottom
are the boundary — do not exceed them when adapting this copy.

Replace [REPO LINK] with the canonical repository URL before posting:
https://github.com/asadhanif3188/mlops-platform-on-eks
-->

# Social Launch Assets — MLOps Platform on AWS EKS

Public distribution copy for the finished repository. Each post presents
engineering judgment and traces back to real project evidence, not a completion
announcement. Adapt the tone to each platform, but keep the claims inside the
guardrails at the end of this file.

---

## 1. Primary LinkedIn launch post

> **I started with a course-style ML pipeline. The interesting engineering problem was never getting the model to train — it was building and operating the platform around it.**
>
> The model is a small RandomForest classifier. Deliberately small, because the
> model was never the point. The point was everything a laptop notebook quietly
> skips: how the environment gets provisioned and destroyed, how a workload gets
> AWS credentials without static keys, where data and experiment state actually
> live, how you know the system is healthy, and what happens when it breaks.
>
> So I re-engineered the whole thing, one layer at a time:
>
> local / course-style → engineering quality gates → containers + CI → Kubernetes
> → Terraform / EKS → cloud-backed data + in-cluster MLflow → security →
> observability → controlled failure injection → runbook-driven recovery.
>
> A few things I can point to as evidence, not adjectives:
>
> • **Terraform provisioned 65 resources** into a real EKS cluster (v1.35.6, two
>   nodes across two AZs), then destroyed all 65 and verified nothing was left
>   billing. Symmetric teardown is a feature, not an afterthought.
> • **No static AWS keys anywhere** — the pipeline gets scoped, short-lived
>   credentials through EKS Pod Identity.
> • **The pipeline ran to completion on live EKS** — exit 0, five of five stages
>   green — with the dataset pulled from S3 and checksum-verified, and the MLflow
>   run persisted to PostgreSQL + S3.
> • **Four defects that every static check passed** surfaced only under live
>   enforcement, and got root-caused and fixed.
>
> The one I keep coming back to: an OOM alert I had written and unit-tested —
> that could never actually fire. It was keyed on a metric that a finite
> Kubernetes Job (`restartPolicy: Never`) never emits. It looked correct in every
> test. It was only disproven when I injected a real out-of-memory kill and
> watched the alert stay silent. I re-keyed it on the signal the Job does emit,
> then re-verified against another real OOM. Failure testing found what static
> validation structurally could not.
>
> The full write-up — architecture, the runtime evidence, the failure/recovery
> log, and the decisions behind each layer — is in the repository.
>
> [REPO LINK]
>
> If you work on MLOps or platform engineering, I'd genuinely value a second pair
> of eyes on the trade-offs.

*Grounding: 65 resources / EKS v1.35.6 / 2 nodes / 2 AZs; Pod Identity; exit 0, 5/5 stages; S3 + checksum; MLflow → PostgreSQL/S3; 4 static-invisible defects; the OOM-alert failure; symmetric verified teardown — all from docs/case-study.md and docs/proof/README.md.*

---

## 2. Follow-up technical series

Ten posts, each self-contained, each following the same spine:
**Hook → Engineering context → Decision/failure → Trade-off → Result → CTA (where it fits).**
Publish in the order given in the posting sequence below, not necessarily this
numbering.

### 2.1 — Why a Kubernetes Job, not a Deployment

**Hook:** The default answer for "run this on Kubernetes" is a Deployment. For a
training pipeline, that default is wrong.

**Context:** The pipeline is finite batch work — it loads data, trains, logs, and
exits. A Deployment models a long-running service that should always be up.

**Decision:** I ran it as a `batch/v1` Job with `restartPolicy: Never`. Modeling
finite work as a service would have misrepresented what the system actually does
— and would have quietly restarted failed runs I wanted to see fail.

**Trade-off:** Jobs don't give you the always-on semantics people reach for
reflexively, and — as I learned the hard way — their metrics profile differs from
long-running pods, which broke one of my alerts.

**Result:** The workload model matches reality, and the mismatch it exposed
became one of the most instructive bugs in the project.

**CTA:** What's your rule of thumb for Job vs. Deployment on batch workloads?

### 2.2 — Why ConfigMap gave way to S3 dataset delivery

**Hook:** The dataset started life closer to the cluster than it should have.

**Context:** Baking data into an image or shipping it through a ConfigMap is easy
and fast — until you need it versioned, integrity-checked, and independent of the
container build.

**Decision:** The dataset is fetched from S3 at runtime and checksum-verified
before training starts. Data stays cloud-backed and versionable; the image stays
about code.

**Trade-off:** A runtime fetch adds a network dependency and a failure mode at
startup — which I deliberately kept, because it's a real one worth testing.

**Result:** When I injected an S3 404, the pipeline failed cleanly at the fetch
stage with a clear per-stage signal — exactly the behavior I wanted.

**CTA:** Where do you draw the line between "config" and "data" on Kubernetes?

### 2.3 — Why MLflow moved in-cluster

**Hook:** The easiest MLflow is someone else's hosted MLflow. I ran my own.

**Context:** Experiment tracking needs durable state — run metadata and
artifacts that survive a pod restart. A managed SaaS solves that, at the cost of
an external dependency and less control over where data lives.

**Decision:** I ran MLflow in-cluster, backing metadata to PostgreSQL and
artifacts to SSE-KMS-encrypted S3. Self-owned state instead of an outside
service.

**Trade-off:** I now own MLflow's availability — which is why it became one of
the three failures I injected on purpose.

**Result:** When MLflow went down (`probe_success=0` while `pg_up=1` proved the
database was fine), the runbook path was to scale replicas back to 1; the
persisted runs were never at risk because state lived in PostgreSQL and S3, not
in the pod.

**CTA:** Self-host or SaaS for experiment tracking — where do you land?

### 2.4 — Workload identity vs. static AWS credentials

**Hook:** The fastest way to give a pod AWS access is an access key. It's also the
one I refused to use.

**Context:** Static credentials in a public repository — or even mounted into a
cluster — are a standing liability. They leak, they linger, they over-grant.

**Decision:** The workload authenticates through EKS Pod Identity, which issues
scoped, short-lived credentials. No long-lived keys anywhere in the cluster or
the repo.

**Trade-off:** More upfront IAM and trust-policy wiring than pasting a key into a
secret. That wiring is the point.

**Result:** The dataset fetch and artifact writes on live EKS ran entirely on
short-lived identity — verified, with zero static keys in the system.

**CTA:** If you're still shipping static keys to workloads, what's blocking the
move?

### 2.5 — When NetworkPolicy broke Pod Identity

**Hook:** Two security controls, each correct on its own, that were jointly
wrong.

**Context:** I run a deny-by-default NetworkPolicy — east-west traffic is blocked
unless explicitly allowed. I also run EKS Pod Identity for credentials. Both are
things you're told to do.

**Decision/failure:** Deny-by-default silently blocked the very traffic Pod
Identity needs to issue credentials. Nothing in CI caught it — each control was
individually valid. It only failed when the two met on a live cluster.

**Trade-off:** Deny-by-default means you *will* block something you needed; the
security posture is worth the debugging cost, but only if you actually run it
under load.

**Result:** Root-caused to the interaction, added the explicit allow rule Pod
Identity requires, and re-verified on EKS.

**CTA:** How do you test for failures that only emerge when two correct
components interact?

### 2.6 — The OOM alert that could never fire

**Hook:** I wrote an alert, unit-tested it, and shipped it. It was incapable of
ever firing.

**Context:** I had an out-of-memory alert for the pipeline. It passed its tests.
It looked right in review.

**Failure:** It was keyed on a metric that a finite Job with `restartPolicy:
Never` structurally never emits. No amount of unit testing would reveal that —
the test asserted the rule's logic, not that the underlying signal exists for
this workload.

**Trade-off:** Static and unit-level validation is necessary and cheap, but it
verifies the shape of your logic, not the physics of your runtime.

**Result:** A real injected OOM (exit 137, `terminated_reason="OOMKilled"`)
proved the alert silent. I re-keyed it on the signal the Job actually emits,
restored the memory limit, and re-verified against a second real OOM.

**CTA:** What's the most confidently-wrong test you've ever shipped?

### 2.7 — Why failure testing came before reliability hardening

**Hook:** I broke the system on purpose before I made it more reliable. That
order was deliberate.

**Context:** The tempting path is to add retries, timeouts, and guards up front
because they're "best practice." That's how you accumulate defensive code no
observed failure ever justified.

**Decision:** I injected failures first — missing dataset, MLflow outage, OOM —
observed how the system actually broke, and only then implemented the fixes those
specific failures justified. Other candidate hardening was declined, with reasons
recorded.

**Trade-off:** It feels backwards, and it means shipping something you know isn't
yet hardened. But it keeps reliability work honest and evidence-driven.

**Result:** Every reliability change traces to a failure I actually saw —
bounded retry, checksum integrity, resource coupling — not to a checklist.

**CTA:** Do you harden first, or break first? I've become a break-first convert.

### 2.8 — What Prometheus should actually monitor in batch ML

**Hook:** Monitoring advice is written for services that stay up. Batch ML pods
exit on purpose.

**Context:** A finite Job's healthy end state is *gone*. Alerting on "pod not
running" is noise; the signals that matter are per-stage success, completion, and
the resource conditions that kill a run.

**Decision:** Prometheus scrapes four signal layers — the Kubernetes platform,
the ephemeral pipeline Job, MLflow, and PostgreSQL — feeding three Grafana
dashboards and eight unit-tested alert rules, each mapped to a concrete operator
action.

**Trade-off:** Designing signals for something that's supposed to disappear takes
more thought than dropping in a generic uptime check.

**Result:** On the live run, Prometheus reported 11 targets up and the dashboards
served live data; the final healthy state was 0 alerts firing with 16 runs
persisted. (And one alert design was wrong — see the OOM post — which is exactly
why failure testing sits next to observability.)

**CTA:** How do you monitor workloads whose success state is "terminated"?

### 2.9 — SBOM and digest controls, and where they stop

**Hook:** I added supply-chain controls — and I'm equally clear about what they
don't cover.

**Context:** "Secure supply chain" gets used loosely. I wanted concrete controls
and an honest boundary.

**Decision:** Images are non-root and restricted, scanned with Trivy in CI, ship
with a CycloneDX SBOM, and are deployed by immutable digest rather than a mutable
tag.

**Trade-off:** That gives provenance and a known ingredient list — but it is
*not* a signed, attested chain. Cosign signing and SLSA attestation are
deliberately deferred, not silently missing.

**Result:** Real, verifiable controls (scan + SBOM + digest pinning), with the
boundary stated where the evidence stops.

**CTA:** SBOM and digest pinning without signing — meaningful, or half a
measure? I think it's meaningful; convince me otherwise.

### 2.10 — Why I deferred GitOps and Terraform remote state

**Hook:** Some things I left out on purpose, and I'd rather name them than let you
assume I forgot.

**Context:** GitOps (e.g. Argo/Flux) and Terraform remote state with locking are
standard for team-operated infrastructure. They also solve problems a
single-operator portfolio proof doesn't have.

**Decision:** One reproducible, destroyable Terraform source of truth, operated
by one person. Remote state/locking and GitOps were deferred — declared at the
boundary, not omitted quietly.

**Trade-off:** No multi-operator state safety, no continuous reconciliation. For
a single-operator proof that provisions and tears down in one session, that's an
acceptable, stated limit — not a gap I'm hoping you won't notice.

**Result:** The scope stays honest: the claims stop precisely where the evidence
stops, and every deferral has a recorded reason.

**CTA:** When is deferring a "best practice" the more senior decision?

---

## 3. Facebook / GKMBA post (shorter, less technical)

> A while ago I finished a course project: a small machine-learning pipeline that
> ran on my laptop. It worked. It also taught me how much a working laptop demo
> quietly leaves out.
>
> So I spent the following stretch turning it into something closer to how real
> systems are run — not by making the model fancier (it's deliberately simple),
> but by building the whole platform around it: real cloud infrastructure I could
> create and destroy on command, security so there are no passwords lying around,
> monitoring so I'd know the moment something went wrong, and — the part I'm
> proudest of — deliberately breaking it to see if my safety nets actually held.
>
> They didn't always. One alarm I'd built and tested turned out to be incapable
> of ever going off. I only found that out by causing the exact failure it was
> supposed to catch and watching it stay silent. Fixing that taught me more than
> any tutorial did: testing your logic and testing reality are two different
> things.
>
> I built it on real cloud infrastructure, proved it worked, broke it, recovered
> it using my own written procedures, and then shut it all down cleanly so it
> wasn't costing anything.
>
> The biggest lesson wasn't a tool. It was discipline — being honest about what
> I could prove versus what I'd merely assumed, and being willing to break my own
> work to find out which was which.
>
> Full details (technical) are on my GitHub if anyone's curious: [REPO LINK]

*Grounding: same evidence base; the OOM-alert story and the break-then-recover-then-teardown arc mirror the case study, phrased for a general audience without new claims.*

---

## 4. Suggested posting sequence

A logical order — start with the whole, then go one layer deep at a time,
following the story the project actually tells (build → break → recover →
harden → secure → observe → supply chain → deferrals).

1. **Launch** — the primary LinkedIn post (§1). The full arc.
2. **Architecture decision** — Job vs. Deployment (§2.1). The first real
   engineering fork.
3. **Failure story** — the OOM alert that could never fire (§2.6). The hook
   people remember.
4. **Reliability story** — why failure testing came before hardening (§2.7).
   Turns the failure into a method.
5. **Security story** — workload identity vs. static keys (§2.4), followed by
   NetworkPolicy breaking Pod Identity (§2.5).
6. **Observability story** — what Prometheus should monitor in batch ML (§2.8).
7. **Supply-chain story** — SBOM/digest controls and their limits (§2.9).
8. **Deliberate-deferral story** — why GitOps and remote state were deferred
   (§2.10).

Fit the remaining data/tracking posts — ConfigMap → S3 (§2.2) and in-cluster
MLflow (§2.3) — between the architecture and security beats where they read most
naturally. Space posts a few days apart; don't dump the series in one week.

---

## 5. Hashtag guidance

Restraint reads as more senior than a wall of tags. Use **3–5 relevant tags per
post**, not 10–15. Pick from the set below to match the post's topic.

- **Core (most posts):** `#MLOps` `#PlatformEngineering` `#Kubernetes` `#AWS`
- **Infrastructure posts:** `#Terraform` `#EKS` `#InfrastructureAsCode`
- **Security posts:** `#CloudSecurity` `#DevSecOps`
- **Observability posts:** `#Observability` `#Prometheus` `#SRE`
- **Reliability/failure posts:** `#SRE` `#ReliabilityEngineering`

Rules of thumb: never exceed five; put the two most relevant first; skip generic
filler (`#tech`, `#coding`, `#innovation`); and don't repeat the identical tag
block on every post — tune it to the topic.

---

## 6. Claim guardrails

Every public post must trace back to real project evidence. These phrases keep
the copy honest; they are the boundary, not suggestions.

**Use — supported by evidence:**

- **cloud-native** — the platform is designed for and validated on AWS EKS.
- **production-oriented** — engineered toward production practices (IaC, identity,
  observability), while scoped as a portfolio proof.
- **real EKS validation** — provisioned and exercised on live Amazon EKS, not a
  local emulator.
- **failure-tested** — three failures injected and recovered on the live cluster.
- **observable** — four signal layers, three dashboards, eight actionable alert
  rules.
- **security-hardened** — Pod Identity, KMS encryption, least-privilege IAM,
  restricted runtime, deny-by-default NetworkPolicy.

**Avoid — not supported by the evidence:**

- **enterprise-grade** — no enterprise SRE, SLA/SLO, or 24/7 operations were
  proven.
- **production proven** — validated as a portfolio-scoped proof, not run in
  production.
- **hyperscale** — the cluster was two nodes, single-operator, short-lived.
- **battle-tested** — controlled failure injection is not sustained production
  exposure.
- **fully production ready** — HA/DR, multi-region, GitOps, and remote state are
  deliberately deferred.

**The honest framing to fall back on:** *a portfolio-scoped platform-engineering
proof, backed by real runtime evidence — the claims stop precisely where the
evidence stops.*
