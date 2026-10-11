from __future__ import annotations

from app.api.v1.mcp import mcp_status
from app.core.auth import AuthUser
from app.core.config import Settings


def test_mcp_status_reports_mounted_local_allowlist_without_secrets():
    result = mcp_status(
        user=AuthUser(id="test-user"),
        settings=Settings(
            mcp_enabled=True,
            mcp_access_mode="local",
            auth_disabled=True,
            supabase_jwt_secret="must-not-leak",
        ),
    )

    assert result["enabled"] is True
    assert result["available"] is True
    assert result["endpoint"] == "/mcp"
    assert result["access_mode"] == "local"
    assert result["hardware_execution_supported"] is False
    assert "polaris_capabilities" in result["tools"]
    assert "must-not-leak" not in str(result)


def test_mcp_status_fails_closed_for_misconfigured_cloud_auth():
    result = mcp_status(
        user=AuthUser(id="test-user"),
        settings=Settings(
            mcp_enabled=True,
            mcp_access_mode="authenticated",
            auth_disabled=True,
            supabase_jwt_secret="",
        ),
    )

    assert result["enabled"] is True
    assert result["available"] is False
    assert result["reason"] == "authentication_not_configured"


def test_mcp_status_requires_cloud_host_allowlist():
    result = mcp_status(
        user=AuthUser(id="test-user"),
        settings=Settings(
            mcp_enabled=True,
            mcp_access_mode="authenticated",
            auth_disabled=False,
            supabase_jwt_secret="configured",
            mcp_allowed_hosts="",
        ),
    )

    assert result["available"] is False
    assert result["reason"] == "allowed_hosts_not_configured"
