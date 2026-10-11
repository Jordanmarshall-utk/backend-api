"""Allowlisted, read-only MCP gateway mounted by the POLARIS API."""

from __future__ import annotations

import ipaddress
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.auth import verify_supabase_jwt
from app.core.config import Settings
from app.services.literature_kg_retrieval import retrieve_literature_kg_claims


TOOL_NAMES = (
    "polaris_capabilities",
    "literature_search_local",
    "literature_evidence_local",
    "experiment_validate_draft",
)

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def _normalized_query(query: str) -> str:
    normalized = " ".join(str(query or "").split())
    if len(normalized) < 3:
        raise ValueError("query must contain at least 3 characters")
    if len(normalized) > 1000:
        raise ValueError("query must contain no more than 1000 characters")
    return normalized


def _safe_limit(limit: int) -> int:
    return max(1, min(int(limit), 10))


def _literature_results(query: str, limit: int, *, include_evidence: bool) -> dict[str, Any]:
    normalized = _normalized_query(query)
    bounded_limit = _safe_limit(limit)
    try:
        claims = retrieve_literature_kg_claims(normalized, bounded_limit)
    except Exception:
        # The local graph loader includes absolute paths in its exceptions. Do
        # not disclose those implementation details through the MCP boundary.
        return {
            "status": "unavailable",
            "source": "local_literature_knowledge_graph",
            "query": normalized,
            "count": 0,
            "results": [],
            "message": "The local literature knowledge graph is unavailable.",
        }

    results: list[dict[str, Any]] = []
    for claim in claims:
        item: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "claim_text": claim.claim_text,
            "predicate": claim.predicate,
            "causal_status": claim.causal_status,
            "support_status": claim.support_status,
            "doi": claim.doi,
            "paper_title": claim.paper_title,
            "subjects": list(claim.subjects),
            "objects": list(claim.objects),
            "relevance_score": claim.score,
        }
        if include_evidence:
            item["evidence_excerpt"] = claim.evidence_excerpt
        results.append(item)
    return {
        "status": "ok",
        "source": "local_literature_knowledge_graph",
        "query": normalized,
        "count": len(results),
        "results": results,
        "causal_warning": "Retrieved reports and associations are not causal proof.",
    }


def _transport_security(settings: Settings | None) -> TransportSecuritySettings | None:
    if settings is None or settings.mcp_access_mode == "local":
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=settings.mcp_allowed_host_list,
        allowed_origins=settings.mcp_allowed_origin_list,
    )


def create_polaris_mcp_server(settings: Settings | None = None) -> FastMCP:
    """Create a stateless MCP server containing only the reviewed tool allowlist."""
    server = FastMCP(
        name="POLARIS Read-Only Gateway",
        instructions=(
            "Read-only scientific planning gateway. It cannot start jobs, access arbitrary "
            "files, dispatch agents, upload artifacts, or execute laboratory hardware."
        ),
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        transport_security=_transport_security(settings),
    )

    @server.tool(
        name="polaris_capabilities",
        description="Describe the safe MCP allowlist and its enforced execution boundaries.",
        annotations=READ_ONLY_ANNOTATIONS,
    )
    def polaris_capabilities() -> dict[str, Any]:
        return {
            "service": "POLARIS",
            "interface": "read_only_mcp_gateway",
            "tools": list(TOOL_NAMES),
            "safety": {
                "read_only": True,
                "job_start": False,
                "hardware_execution": False,
                "arbitrary_agent_dispatch": False,
                "filesystem_access": False,
                "drive_access": False,
                "jupyter_access": False,
            },
            "experimental_scope": (
                "Planning-gate and deterministic OT-2 draft preflight only; no simulator or "
                "physical execution."
            ),
        }

    @server.tool(
        name="literature_search_local",
        description=(
            "Search the local literature knowledge graph and return claim-level provenance. "
            "No network search is performed."
        ),
        annotations=READ_ONLY_ANNOTATIONS,
    )
    def literature_search_local(query: str, limit: int = 5) -> dict[str, Any]:
        return _literature_results(query, limit, include_evidence=False)

    @server.tool(
        name="literature_evidence_local",
        description=(
            "Search the local literature knowledge graph and include retained evidence excerpts. "
            "Reported associations must not be presented as causal proof."
        ),
        annotations=READ_ONLY_ANNOTATIONS,
    )
    def literature_evidence_local(query: str, limit: int = 5) -> dict[str, Any]:
        return _literature_results(query, limit, include_evidence=True)

    @server.tool(
        name="experiment_validate_draft",
        description=(
            "Apply POLARIS release-gate checks to a draft experimental plan. This tool never "
            "generates a worklist, starts a job, contacts an instrument, or executes a protocol."
        ),
        annotations=READ_ONLY_ANNOTATIONS,
    )
    def experiment_validate_draft(
        plan: str,
        constraints: dict[str, Any] | None = None,
        protocol_source: str | None = None,
        worklist_csv: str | None = None,
    ) -> dict[str, Any]:
        from app.services.experiment_defaults import merge_constraints
        from app.services.experiment_validation import experiment_automation_blockers
        from app.tools.opentrons_draft import validate_opentrons_draft

        normalized_plan = str(plan or "").strip()
        if len(normalized_plan) > 50_000:
            return {
                "draft_gate_passed": False,
                "automation_ready": False,
                "blocker_count": 1,
                "blockers": ["The draft exceeds the 50,000-character validation limit."],
                "protocol_validation": None,
                "execution_performed": False,
                "hardware_contacted": False,
                "simulator_contacted": False,
            }
        try:
            normalized_constraints = merge_constraints(constraints or {})
        except (TypeError, ValueError, OverflowError):
            draft_blockers = [
                "The supplied constraints are not valid POLARIS experiment constraints."
            ]
        else:
            draft_blockers = experiment_automation_blockers(
                normalized_constraints,
                normalized_plan,
            )

        protocol_validation: dict[str, Any] | None = None
        release_blockers = list(draft_blockers)
        if protocol_source is None:
            release_blockers.append(
                "No OT-2 protocol draft was supplied for deterministic preflight."
            )
        elif len(protocol_source) > 100_000 or (
            worklist_csv is not None and len(worklist_csv) > 500_000
        ):
            release_blockers.append("The protocol or worklist exceeds the MCP validation limit.")
        else:
            protocol_validation = validate_opentrons_draft(protocol_source, worklist_csv).as_dict()
            if protocol_validation["status"] != "syntax_validated_draft":
                release_blockers.append("The OT-2 draft did not pass deterministic preflight.")

        release_blockers.extend(
            (
                "An official OT-2 simulator run has not passed.",
                "Human review and approval have not been recorded.",
            )
        )
        return {
            "draft_gate_passed": not draft_blockers,
            "draft_blockers": draft_blockers,
            # This read-only adapter is deliberately not a release authority.
            "automation_ready": False,
            "blocker_count": len(release_blockers),
            "blockers": release_blockers,
            "protocol_validation": protocol_validation,
            "validation_scope": "planning_gate_and_deterministic_ot2_preflight",
            "execution_performed": False,
            "hardware_contacted": False,
            "simulator_contacted": False,
            "human_approval_recorded": False,
        }

    return server


class _MCPAccessControl:
    """Enforce an explicit local-only or bearer-authenticated MCP boundary."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def _reject(self, scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
        response = JSONResponse({"detail": detail}, status_code=status)
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if self.settings.mcp_access_mode == "local":
            client = scope.get("client")
            host = str(client[0]) if client else ""
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = host.casefold() == "localhost"
            if not loopback:
                await self._reject(
                    scope,
                    receive,
                    send,
                    403,
                    "MCP local mode accepts loopback clients only",
                )
                return
        else:
            if self.settings.auth_disabled or not self.settings.supabase_jwt_secret:
                await self._reject(
                    scope,
                    receive,
                    send,
                    503,
                    "MCP authentication is not configured",
                )
                return
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            authorization = headers.get(b"authorization", b"").decode("latin-1")
            scheme, _, token = authorization.partition(" ")
            if scheme.casefold() != "bearer" or not token.strip():
                await self._reject(scope, receive, send, 401, "Missing bearer token")
                return
            try:
                verify_supabase_jwt(token.strip(), self.settings)
            except Exception as exc:
                status_code = int(getattr(exc, "status_code", 401))
                detail = str(getattr(exc, "detail", "Invalid bearer token"))
                await self._reject(scope, receive, send, status_code, detail)
                return

        await self.app(scope, receive, send)


def create_mcp_http_app(server: FastMCP, settings: Settings) -> ASGIApp:
    """Build the Streamable HTTP ASGI app with POLARIS access control."""
    return _MCPAccessControl(server.streamable_http_app(), settings)


def mount_mcp_gateway(app: Any, settings: Settings) -> FastMCP | None:
    """Mount the opt-in gateway and return its server for lifespan management."""
    if not settings.mcp_enabled:
        return None
    server = create_polaris_mcp_server(settings)
    app.state.polaris_mcp_server = server
    app.mount("/mcp", create_mcp_http_app(server, settings), name="mcp")
    return server
