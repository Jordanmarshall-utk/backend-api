# POLARIS on the National Research Platform

This directory contains a conservative first Kubernetes base for the NRP Nautilus cluster. It is a deployment starting point, not a production authorization to expose laboratory instruments.

## Architecture boundary

Run these responsibilities separately:

- **POLARIS API/MCP gateway:** stateless request handling, read-only agent discovery, local evidence reads, protocol-draft validation.
- **Workers:** future Kubernetes Jobs for long LiteratureAgent/model workloads. The current local subprocess job manager is single-replica only.
- **Persistent data:** a PVC for the current transition; object storage plus PostgreSQL are required before horizontal scaling.
- **Laboratory connectors:** on the building/lab network, never inside NRP. They may accept only approved immutable artifacts and return telemetry.

The base therefore pins the API to one replica with `Recreate`. Scaling it before moving file-backed sessions, SQLite, jobs, outputs, and audit state to shared services would create inconsistent state.

## Prerequisites

1. Obtain an NRP namespace and working `kubectl`/`kubelogin` access.
2. Build the backend image and push it to a registry accessible from Nautilus.
3. Replace `replace-athena-namespace`, `replace-athena-project`, and `replace-version` in `deploy/nrp`.
4. Create the required secret without committing it:

   ```powershell
   kubectl -n <namespace> create secret generic polaris-secrets `
     --from-literal=supabase-jwt-secret='<value>' `
     --from-literal=huggingface-api-key='<value>'
   ```

5. Seed `/data/literature/outputs` and other required corpus artifacts through an approved data-transfer process. Do not put protected or regulated data on NRP.

## Validate before applying

```powershell
kubectl kustomize deploy/nrp
kubectl apply --dry-run=server -k deploy/nrp
kubectl apply -k deploy/nrp
kubectl rollout status deployment/polaris-api -n <namespace>
```

Verify `/api/v1/health`, authenticated REST calls, and the read-only MCP capability response. Do not enable cloud MCP while application authentication is disabled.

## Cloudification milestones

1. Container and single-replica API smoke tests.
2. Authenticated, read-only Streamable HTTP MCP gateway.
3. PostgreSQL for application/checkpoint state and object storage for papers, evidence packets, models, and exports.
4. Queue-backed workers implemented as Jobs or a durable worker service; remove local `subprocess.Popen` job ownership.
5. Per-user/project tenancy, quotas, trace IDs, idempotency, and append-only audit events.
6. Separate simulated instrument MCP connectors.
7. Laboratory-side connectors with approval/interlock controls; no direct NRP-to-device route.

## Operational notes

- NRP uses Kubernetes namespaces, container images, PVCs, resource requests/limits, and HAProxy ingress with TLS.
- The cluster can evict containers that overuse ephemeral scratch space, so large literature/model outputs belong on persistent or object storage.
- Long batch computation should move to Kubernetes Jobs rather than keeping an idle API pod busy.
- Secrets stay in Kubernetes Secrets or an external secret manager, never in images, manifests, logs, or MCP results.
