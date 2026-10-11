from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session

from app.core.config import Settings
from app.services.literature_kg_retrieval import LiteratureKGClaim


EXPECTED_TOOLS = {
    "polaris_capabilities",
    "literature_search_local",
    "literature_evidence_local",
    "experiment_validate_draft",
}


def _structured(result):
    assert result.isError is False
    assert isinstance(result.structuredContent, dict)
    return result.structuredContent


def test_protocol_initialize_lists_only_the_read_only_allowlist():
    from app.mcp.gateway import create_polaris_mcp_server

    async def scenario():
        server = create_polaris_mcp_server()
        async with create_connected_server_and_client_session(server) as session:
            listed = await session.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == EXPECTED_TOOLS
            for tool in tools.values():
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is True
                assert tool.annotations.destructiveHint is False
                assert tool.annotations.idempotentHint is True
                assert tool.annotations.openWorldHint is False

            result = _structured(await session.call_tool("polaris_capabilities", {}))
            assert set(result["tools"]) == EXPECTED_TOOLS
            assert result["safety"]["hardware_execution"] is False
            assert result["safety"]["filesystem_access"] is False
            assert result["safety"]["job_start"] is False

    anyio.run(scenario)


def test_protocol_call_validates_draft_without_execution():
    from app.mcp.gateway import create_polaris_mcp_server

    async def scenario():
        server = create_polaris_mcp_server()
        async with create_connected_server_and_client_session(server) as session:
            return _structured(
                await session.call_tool(
                    "experiment_validate_draft",
                    {
                        "plan": "Mix a candidate using a Tecan.",
                        "constraints": {
                            "techniques": [],
                            "equipment": [],
                            "liquid_handling": {"materials": [], "instruments": []},
                        },
                    },
                )
            )

    result = anyio.run(scenario)

    assert result["automation_ready"] is False
    assert result["draft_gate_passed"] is False
    assert result["execution_performed"] is False
    assert result["hardware_contacted"] is False
    assert result["simulator_contacted"] is False
    assert result["blocker_count"] >= 3


def test_protocol_preflight_never_claims_simulation_or_automation_readiness():
    from app.mcp.gateway import create_polaris_mcp_server

    async def scenario():
        server = create_polaris_mcp_server()
        async with create_connected_server_and_client_session(server) as session:
            return _structured(
                await session.call_tool(
                    "experiment_validate_draft",
                    {
                        "plan": (
                            "Randomize substrates in triplicate (n=3), include negative and "
                            "process-matched controls, and characterize by XRD."
                        ),
                        "constraints": {
                            "techniques": ["XRD"],
                            "equipment": [],
                            "liquid_handling": {
                                "materials": ["stock A"],
                                "instruments": ["manual pipettes"],
                            },
                        },
                        "protocol_source": "import os\nmetadata = {'apiLevel': '2.13'}\ndef run(protocol):\n    pass\n",
                        "worklist_csv": "Well,stock_A_uL\nA1,50\n",
                    },
                )
            )

    result = anyio.run(scenario)
    assert result["draft_gate_passed"] is True
    assert result["protocol_validation"]["status"] == "invalid_draft"
    assert any("Disallowed protocol import" in item for item in result["protocol_validation"]["errors"])
    assert result["automation_ready"] is False
    assert result["simulator_contacted"] is False
    assert result["hardware_contacted"] is False
    assert result["human_approval_recorded"] is False


def test_protocol_literature_evidence_preserves_provenance_without_paths(monkeypatch):
    from app.mcp import gateway

    claim = LiteratureKGClaim(
        claim_id="claim:1",
        claim_text="A reported association.",
        predicate="associated_with",
        causal_status="reported_association",
        support_status="evidence_linked",
        doi="10.1000/test",
        paper_title="A paper",
        subjects=("material",),
        objects=("stability",),
        evidence_excerpt="The retained evidence excerpt.",
        score=1.25,
    )
    monkeypatch.setattr(gateway, "retrieve_literature_kg_claims", lambda *_a, **_k: [claim])
    server = gateway.create_polaris_mcp_server()

    async def scenario():
        async with create_connected_server_and_client_session(server) as session:
            return _structured(
                await session.call_tool(
                    "literature_evidence_local",
                    {"query": "perovskite stability", "limit": 4},
                )
            )

    result = anyio.run(scenario)

    assert result["source"] == "local_literature_knowledge_graph"
    assert result["results"][0]["doi"] == "10.1000/test"
    assert result["results"][0]["evidence_excerpt"] == "The retained evidence excerpt."
    serialized = str(result).lower()
    assert "graph_root" not in serialized
    assert "filesystem" not in serialized


def test_gateway_mount_is_opt_in_and_uses_the_mcp_path():
    from fastapi import FastAPI

    from app.mcp.gateway import mount_mcp_gateway

    disabled = FastAPI()
    assert mount_mcp_gateway(disabled, Settings(mcp_enabled=False)) is None
    assert not any(getattr(route, "path", None) == "/mcp" for route in disabled.routes)

    enabled = FastAPI()
    server = mount_mcp_gateway(
        enabled,
        Settings(mcp_enabled=True, mcp_access_mode="local", auth_disabled=True),
    )
    assert server is enabled.state.polaris_mcp_server
    assert any(getattr(route, "path", None) == "/mcp" for route in enabled.routes)


def test_streamable_http_initialize_list_and_call_on_mounted_path():
    from fastapi import FastAPI

    from app.mcp.gateway import create_mcp_http_app, create_polaris_mcp_server

    async def scenario():
        settings = Settings(mcp_enabled=True, mcp_access_mode="local", auth_disabled=True)
        server = create_polaris_mcp_server(settings)

        @asynccontextmanager
        async def lifespan(_app):
            async with server.session_manager.run():
                yield

        app = FastAPI(lifespan=lifespan)
        app.mount("/mcp", create_mcp_http_app(server, settings))

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app,
                    client=("127.0.0.1", 50000),
                ),
                base_url="http://127.0.0.1:8080",
                follow_redirects=True,
            ) as http_client:
                async with streamable_http_client(
                    "http://127.0.0.1:8080/mcp",
                    http_client=http_client,
                ) as (read_stream, write_stream, _session_id):
                    async with ClientSession(read_stream, write_stream) as session:
                        initialized = await session.initialize()
                        listed = await session.list_tools()
                        called = _structured(
                            await session.call_tool("polaris_capabilities", {})
                        )

        return initialized, listed, called

    initialized, listed, called = anyio.run(scenario)
    assert initialized.serverInfo.name == "POLARIS Read-Only Gateway"
    assert {tool.name for tool in listed.tools} == EXPECTED_TOOLS
    assert called["safety"]["hardware_execution"] is False


def test_local_access_mode_rejects_non_loopback_clients():
    from app.mcp.gateway import create_mcp_http_app, create_polaris_mcp_server

    settings = Settings(mcp_enabled=True, mcp_access_mode="local", auth_disabled=True)
    transport = create_mcp_http_app(create_polaris_mcp_server(), settings)
    with TestClient(transport, client=("203.0.113.7", 50000)) as client:
        response = client.post("/", json={})
    assert response.status_code == 403
    assert response.json()["detail"] == "MCP local mode accepts loopback clients only"


def test_authenticated_access_mode_fails_closed_when_auth_is_disabled():
    from app.mcp.gateway import create_mcp_http_app, create_polaris_mcp_server

    settings = Settings(
        mcp_enabled=True,
        mcp_access_mode="authenticated",
        auth_disabled=True,
    )
    transport = create_mcp_http_app(create_polaris_mcp_server(), settings)
    with TestClient(transport, client=("127.0.0.1", 50000)) as client:
        response = client.post("/", json={})
    assert response.status_code == 503
    assert response.json()["detail"] == "MCP authentication is not configured"
