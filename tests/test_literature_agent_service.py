from pathlib import Path

from app.tools.literature_agent_service import LiteratureAgentConfig, LiteratureAgentService
from app.tools.agent_contract import AgentTask
from app.tools.polaris_orchestrator import PolarisOrchestrator


def test_health_and_artifact_contract():
    service = LiteratureAgentService(LiteratureAgentConfig.load())
    health = service.health()
    assert health["service"] == "LiteratureAgent"
    assert "extract_batch" in health["stages"]
    assert "all_records_csv" in service.artifact_manifest()


def test_commands_preserve_pass_contract():
    service = LiteratureAgentService(LiteratureAgentConfig.load())
    extract = service.build_command("extract_batch", {"max_papers": 2, "search_query": "perovskite stability"})
    assert "--disable_google_drive" in extract
    assert "--inline_vision" in extract
    assert "perovskite stability" in extract
    integrate = service.build_command("integrate_and_model", {})
    assert "--skip_literature_agent" in integrate


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
    assert Path(config.model_script).exists()
    assert Path(config.controller).is_relative_to(backend_root)
    assert Path(config.work_dir).is_relative_to(stack_root / "runtime")
    config_text = (backend_root / "config" / "literature_agent.json").read_text(encoding="utf-8").lower()
    assert "users\\jorda" not in config_text
    assert "e:\\" not in config_text


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
