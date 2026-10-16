from pathlib import Path

import pytest

from app.tools.literature_agent_service import (
    STAGES,
    LiteratureAgentConfig,
    LiteratureAgentService,
)
from app.tools.agent_contract import AgentTask
from app.tools.polaris_orchestrator import PolarisOrchestrator


def test_health_and_artifact_contract():
    service = LiteratureAgentService(LiteratureAgentConfig.load())
    health = service.health()
    assert health["service"] == "LiteratureAgent"
    assert health["domain_profile"] == "halide_pv"
    assert set(health["domain_profiles"]) == {
        "general_science",
        "oxide_perovskite",
        "halide_pv",
    }
    assert "extract_batch" in health["stages"]
    assert health["path_checks"]["knowledge_graph_builder"] is True
    assert "all_records_csv" in service.artifact_manifest()


def test_commands_preserve_pass_contract():
    service = LiteratureAgentService(LiteratureAgentConfig.load())
    extract = service.build_command("extract_batch", {"max_papers": 2, "search_query": "perovskite stability"})
    assert "--disable_google_drive" in extract
    assert "--inline_vision" in extract
    assert extract[extract.index("--domain_profile") + 1] == "halide_pv"
    assert extract[extract.index("--family_gating") + 1] == "strict"
    assert "perovskite stability" in extract
    integrate = service.build_command("integrate_and_model", {})
    assert "--skip_literature_agent" in integrate


def test_every_stage_command_includes_domain_and_kg_builder(tmp_path):
    service = LiteratureAgentService(
        LiteratureAgentConfig(job_dir=str(tmp_path / "jobs"))
    )
    requests = {
        "evidence_packet": {"query": "phase stability"},
    }

    for stage in STAGES:
        command = service.build_command(stage, requests.get(stage, {}))
        assert command[command.index("--domain_profile") + 1] == "halide_pv"
        assert command[command.index("--knowledge_graph_builder") + 1] == service.config.knowledge_graph_builder


def test_hypothesis_retrieval_uses_fast_evidence_profile(monkeypatch):
    service = LiteratureAgentService(LiteratureAgentConfig.load())
    captured = {}

    def fake_start_stage(stage, request, wait=False):
        captured.update(stage=stage, request=request, wait=wait)
        return {"job_id": "test-job", "status": "running"}

    monkeypatch.setattr(service, "start_stage", fake_start_stage)
    service.start_hypothesis_support_retrieval("RP and DJ spacer phase stability", max_papers=3)

    assert captured["stage"] == "extract_batch"
    assert captured["request"]["extraction_profile"] == "hypothesis_support"
    assert captured["request"]["disable_google_drive"] is True
    command = service.build_command(captured["stage"], captured["request"])
    assert command[command.index("--extraction_profile") + 1] == "hypothesis_support"


def test_default_config_uses_portable_bundled_paths():
    config = LiteratureAgentConfig.load()
    backend_root = Path(__file__).resolve().parents[1]
    stack_root = backend_root.parent

    assert Path(config.controller).exists()
    assert Path(config.ontology_path).exists()
    assert Path(config.knowledge_graph_builder).exists()
    assert Path(config.model_script).exists()
    assert Path(config.controller).is_relative_to(backend_root)
    assert Path(config.knowledge_graph_builder).is_relative_to(backend_root)
    assert Path(config.work_dir).is_relative_to(stack_root / "runtime")
    config_text = (backend_root / "config" / "literature_agent.json").read_text(encoding="utf-8").lower()
    assert "users\\jorda" not in config_text
    assert "e:\\" not in config_text


def test_config_domain_profile_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("LITERATURE_AGENT_DOMAIN_PROFILE", "general_science")

    config = LiteratureAgentConfig.load()

    assert config.domain_profile == "general_science"


def test_config_kg_builder_can_be_overridden_by_environment(monkeypatch):
    custom_builder = Path(__file__).resolve()
    monkeypatch.setenv("LITERATURE_AGENT_KG_BUILDER", str(custom_builder))

    config = LiteratureAgentConfig.load()

    assert config.knowledge_graph_builder == str(custom_builder)


@pytest.mark.parametrize("domain_profile", ["general_science", "oxide_perovskite"])
def test_non_halide_extract_forces_non_pv_gating(tmp_path, domain_profile):
    service = LiteratureAgentService(
        LiteratureAgentConfig(domain_profile=domain_profile, job_dir=str(tmp_path / "jobs"))
    )

    command = service.build_command(
        "extract_batch",
        {
            "domain_profile": domain_profile,
            "extraction_profile": "general_literature",
            "family_gating": "strict",
        },
    )

    assert command[command.index("--domain_profile") + 1] == domain_profile
    assert command[command.index("--family_gating") + 1] == "off"


@pytest.mark.parametrize("domain_profile", ["general_science", "oxide_perovskite"])
def test_non_halide_rejects_pce_extraction_profile(tmp_path, domain_profile):
    service = LiteratureAgentService(
        LiteratureAgentConfig(domain_profile=domain_profile, job_dir=str(tmp_path / "jobs"))
    )

    with pytest.raises(ValueError, match="pce_stability_modeling.*halide_pv"):
        service.build_command(
            "extract_batch",
            {"extraction_profile": "pce_stability_modeling"},
        )


@pytest.mark.parametrize("domain_profile", ["general_science", "oxide_perovskite"])
def test_non_halide_rejects_integration_and_modeling(tmp_path, domain_profile):
    service = LiteratureAgentService(
        LiteratureAgentConfig(domain_profile=domain_profile, job_dir=str(tmp_path / "jobs"))
    )

    with pytest.raises(ValueError, match="integrate_and_model.*halide_pv"):
        service.build_command("integrate_and_model", {})


def test_request_domain_override_is_validated(tmp_path):
    service = LiteratureAgentService(
        LiteratureAgentConfig(job_dir=str(tmp_path / "jobs"))
    )

    with pytest.raises(ValueError, match="Unsupported domain_profile"):
        service.build_command("knowledge_graph", {"domain_profile": "unknown"})


def test_request_ontology_override_is_resolved_and_passed_to_controller(tmp_path):
    service = LiteratureAgentService(
        LiteratureAgentConfig(job_dir=str(tmp_path / "jobs"))
    )

    command = service.build_command(
        "knowledge_graph",
        {
            "domain_profile": "general_science",
            "ontology_path": "literature_agent/config/domain_packs/general_science.json",
        },
    )

    ontology_path = Path(command[command.index("--ontology_path") + 1])
    assert ontology_path == (
        Path(__file__).resolve().parents[1]
        / "literature_agent"
        / "config"
        / "domain_packs"
        / "general_science.json"
    ).resolve()


def test_request_ontology_override_cannot_escape_backend(tmp_path):
    outside_ontology = tmp_path / "outside.json"
    outside_ontology.write_text("{}", encoding="utf-8")
    service = LiteratureAgentService(
        LiteratureAgentConfig(job_dir=str(tmp_path / "jobs"))
    )

    with pytest.raises(ValueError, match="within the POLARIS backend"):
        service.build_command(
            "knowledge_graph",
            {"ontology_path": str(outside_ontology)},
        )


def test_orchestrator_evidence_contract(monkeypatch):
    orchestrator = PolarisOrchestrator()
    service = orchestrator.agents["literature_agent"]
    monkeypatch.setattr(
        service,
        "write_evidence_packet",
        lambda query, limit=5, max_chars=16000: {
            "query": query,
            "papers": [],
            "relationships": [],
            "provenance": {"paper_count": 0},
            "formatted_context": "No matching evidence.",
        },
    )
    result = orchestrator.dispatch(
        AgentTask("literature_agent", "evidence_packet", {"query": "perovskite stability", "limit": 1})
    )
    assert result.status == "completed"
    assert "papers" in result.data
    assert "relationships" in result.data
    assert "provenance" in result.data
