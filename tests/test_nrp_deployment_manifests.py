from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
NRP = ROOT / "deploy" / "nrp"


def _documents():
    documents = []
    for path in sorted(NRP.glob("*.yaml")):
        documents.extend(doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc)
    return documents


def test_nrp_base_has_api_storage_service_and_ingress():
    documents = _documents()
    kinds = {document["kind"] for document in documents}

    assert {"Deployment", "PersistentVolumeClaim", "Service", "Ingress", "NetworkPolicy"} <= kinds


def test_nrp_api_is_non_root_bounded_and_health_checked():
    deployment = next(doc for doc in _documents() if doc["kind"] == "Deployment")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert deployment["spec"]["replicas"] == 1
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["resources"]["requests"]
    assert container["resources"]["limits"]
    assert container["readinessProbe"]["httpGet"]["path"] == "/api/v1/health"
    assert container["livenessProbe"]["httpGet"]["path"] == "/api/v1/health"


def test_nrp_does_not_disable_auth_or_claim_hardware_access():
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in NRP.glob("*.yaml"))

    assert 'AUTH_DISABLED: "true"' not in rendered
    assert "OT2_HOST" not in rendered
    assert "robot.local" not in rendered
    assert "password:" not in rendered.lower()


def test_container_runs_as_unprivileged_user_and_has_dockerignore():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER polaris" in dockerfile
    assert (ROOT / ".dockerignore").exists()
