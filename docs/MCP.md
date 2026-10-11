# POLARIS MCP gateway

POLARIS exposes an opt-in, read-only Model Context Protocol gateway from the existing FastAPI process. The Streamable HTTP endpoint is `/mcp`. The gateway is disabled by default.

## Current allowlist

- `polaris_capabilities` reports the MCP boundary and available tools.
- `literature_search_local` searches the local literature knowledge graph with claim and DOI provenance.
- `literature_evidence_local` adds retained evidence excerpts and causal-status labels.
- `experiment_validate_draft` applies the existing planning release gate and can run deterministic static preflight on an optional OT-2 protocol source/worklist pair. It never runs the official simulator or hardware.

Every tool is declared read-only, non-destructive, idempotent, and closed-world. This surface does not expose job starts, arbitrary agent dispatch, files or device-specific paths, Google Drive, Jupyter, uploads, subprocesses, credentials, protocol generation, OT-2 simulation or execution, or any other hardware action. `experiment_validate_draft` always reports `automation_ready=false`; a prose/static gate is not a simulator result or a release authorization.

## Local desktop use

Set these environment values before starting the backend:

```text
MCP_ENABLED=true
MCP_ACCESS_MODE=local
AUTH_DISABLED=true
```

Bind the API to loopback and connect the MCP client to `http://127.0.0.1:8080/mcp` (replace the port if the desktop backend uses a different one). Local mode rejects non-loopback clients at the ASGI boundary.

## Cloud and NRP deployments

Leave `MCP_ENABLED=false` until authentication and the deployment hostname are configured. For an authenticated deployment, set:

```text
MCP_ENABLED=true
MCP_ACCESS_MODE=authenticated
AUTH_DISABLED=false
SUPABASE_JWT_SECRET=<deployment secret>
MCP_ALLOWED_HOSTS=polaris.example.org
MCP_ALLOWED_ORIGINS=https://polaris.example.org
```

Authenticated mode requires a valid `Authorization: Bearer ...` token. It returns `503` when API authentication or its signing secret is disabled, so an unauthenticated cloud MCP endpoint fails closed. `MCP_ALLOWED_HOSTS` and `MCP_ALLOWED_ORIGINS` are comma-separated and drive the SDK's DNS-rebinding checks; leaving the cloud host allowlist empty denies transport requests.

Place TLS termination, request-size limits, rate limits, and request/audit correlation at the ingress layer. Do not use `local` mode behind a reverse proxy, because a proxy can make remote traffic appear to originate on loopback.

## Lifecycle and scaling

The FastMCP session manager runs inside the FastAPI lifespan. The transport uses stateless Streamable HTTP with JSON responses so replicas do not depend on process-local MCP sessions. The tools themselves remain local and read-only; a shared artifact/object store and durable audit sink are still needed before horizontally scaling the broader POLARIS workflow.

## Instrumentation roadmap boundary

This gateway is the safe discovery and static-validation foundation, not an instrument-control server. Its deterministic OT-2 check covers Python structure and supplied worklist constraints only. OT-2 and other Athena equipment should be separate MCP servers with versioned capability schemas and an explicit progression:

1. protocol schema validation;
2. deterministic digital-twin simulation and collision/volume checks;
3. human approval and signed release artifact;
4. authenticated queue submission with idempotency keys;
5. hardware execution isolated behind the equipment owner's control plane.

No hardware execute tool should be added to this POLARIS gateway. A future orchestrator can discover separate equipment servers, but simulation and execution must remain distinct capabilities with fail-closed policy gates.
