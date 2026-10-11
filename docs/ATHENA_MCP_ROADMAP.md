# Athena MCP interoperability roadmap

## Goal

Let POLARIS, LiteratureAgent, research-team agents, digital twins, and laboratory connectors share explicit capabilities without turning one agent into an unrestricted remote-control plane.

## Common contract

Every MCP service should publish:

- service identity and semantic version;
- owner/team and support contact;
- capability/tool name, JSON schema, and result schema;
- risk class: `read_only`, `compute`, `external_write`, `simulation`, or `hardware_execution`;
- required scopes and data classification;
- idempotency behavior, timeout, and cancellation semantics;
- artifact/checksum inputs and provenance outputs;
- health, protocol version, and compatibility metadata.

Tool names should be domain-scoped, for example `literature.search_local`, `ot2.simulate_protocol`, and `xrd.get_run_status`. Avoid generic `run`, arbitrary tool forwarding, and filesystem-path parameters.

## Delivery phases

### Phase 1 — read-only platform gateway

- Mount Streamable HTTP MCP on the existing authenticated API lifecycle.
- Expose capability discovery, local literature search/evidence, and deterministic protocol-draft validation only.
- Wire the desktop MCP page to real status and capability calls.

### Phase 2 — digital twins

- Package vendor-supported simulators in isolated workers.
- Use immutable protocol/worklist artifacts and return structured validation plus run logs.
- Compare expected deck state, transfers, timing, and consumable usage.

### Phase 3 — team-agent federation

- Register allowlisted MCP endpoints through environment/deployment configuration, not workstation paths.
- Add OAuth/service identities, per-tool scopes, schema/version checks, timeouts, and circuit breakers.
- Route tool calls explicitly; do not let LLM output select arbitrary endpoints or tool names.

### Phase 4 — lab execution connectors

- Deploy connectors beside equipment on controlled laboratory networks.
- Require simulation, human approval, checksums, idempotency, interlocks, and append-only audits.
- Keep NRP as the planning/compute plane; keep hardware authority local.

### Phase 5 — NRP scale-out

- Move sessions/checkpoints to PostgreSQL and artifacts to object storage.
- Move long work to queue-backed Kubernetes Jobs.
- Add namespace isolation, quotas, observability, backups, and disaster-recovery tests.

## First definition of done

- MCP initialize, tool discovery, and every allowlisted call pass integration tests.
- No mutating, Jupyter, Drive, local-path, generic dispatch, or hardware-execution tool is discoverable.
- Generated OT-2 drafts compile and unsafe worklists fail deterministically.
- Cloud startup fails closed when auth is disabled.
- The NRP manifest passes schema/dry-run validation and the container runs as a non-root user.
