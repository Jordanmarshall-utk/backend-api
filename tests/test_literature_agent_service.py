from pathlib import Path

from app.tools.literature_agent_service import LiteratureAgentConfig, LiteratureAgentService
from app.tools.hypothesis_quality import (
    build_quality_flags,
    guard_hypothesis_output,
    prepare_retrieval_query,
)
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


def test_retry_guidance_is_not_used_as_a_scientific_query():
    prepared = prepare_retrieval_query(
        "what are 2D perovskites good for?\n\n"
        "Retry requirement: Novel compounds, variables, and mechanisms are allowed."
    )
    assert prepared["retrieval_query"] == "what are 2D perovskites good for?"
    assert "removed_retry_instruction_section" in prepared["query_transformations"]


def test_selected_continuation_drives_retrieval():
    prepared = prepare_retrieval_query(
        "Broad clarified question\n\nAssumptions:\n- performance is undefined\n"
        "Selected continuation: Test phenethylammonium surface passivation of FAPbI3.\n\n##"
    )
    assert prepared["retrieval_query"] == "Test phenethylammonium surface passivation of FAPbI3."
    assert "Broad clarified question" not in prepared["retrieval_query"]


def test_selected_continuation_retrieval_strips_polaris_disclosure_text():
    prepared = prepare_retrieval_query(
        "Identify diammonium Dion-Jacobson spacers for stable 2D/3D perovskites.\n"
        "Selected continuation: Exploratory hypothesis (not directly supported by the current local packet): "
        "Utilizing N-methyl-2-aminoethanol (NMEA+) as a spacer could alter interfacial dipoles. "
        "## Any implied prior support was omitted because no DOI or paper_slug was supplied."
    )

    assert "N-methyl-2-aminoethanol" in prepared["retrieval_query"]
    assert "interfacial dipoles" in prepared["retrieval_query"]
    assert "exploratory hypothesis" not in prepared["retrieval_query"].lower()
    assert "current local packet" not in prepared["retrieval_query"].lower()
    assert "implied prior support" not in prepared["retrieval_query"].lower()
    assert "paper_slug" not in prepared["retrieval_query"].lower()


def test_unverified_continuation_retrieval_keeps_question_not_warning_label():
    prepared = prepare_retrieval_query(
        "Unverified continuation (no direct supporting evidence retrieved): Which measurement can "
        "distinguish the proposed NMEA+ interfacial-dipole mechanism from competing explanations?"
    )

    assert prepared["retrieval_query"].startswith("Which measurement")
    assert "NMEA+" in prepared["retrieval_query"]
    assert "unverified continuation" not in prepared["retrieval_query"].lower()
    assert "supporting evidence retrieved" not in prepared["retrieval_query"].lower()


def test_continuation_essay_retrieves_only_the_selected_question():
    prepared = prepare_retrieval_query(
        "How can we continue exploring this hypothesis?\n"
        "How does spacer identity alter humidity stability?\n\n"
        "**Evidence anchor:** [E1] unrelated substrate result\n\n"
        "A generated essay with old citations follows."
    )
    assert prepared["retrieval_query"] == "How does spacer identity alter humidity stability?"


def test_final_hypothesis_retrieval_excludes_predictions_and_tests():
    prepared = prepare_retrieval_query(
        "How can we continue exploring this hypothesis?\n"
        "Hypothesis:\n3-APB+ may alter a 2D/3D interface.\n\n"
        "Predictions:\nMobility will improve by 20%.\n\n"
        "Tests:\nUse spectroscopy."
    )
    assert prepared["retrieval_query"] == "3-APB+ may alter a 2D/3D interface."


def test_acronym_first_identity_and_method_checks():
    flags = build_quality_flags(
        "Use 3-APB+ (3-aminophenylbutylammonium) as a Dion-Jacobson spacer. "
        "UV-Vis will measure pi-pi stacking, while TAS and TRPL quantify mobility and trap-state density. "
        "Four-point conductivity will directly measure interfacial electronic coupling.",
        strong_evidence_count=0,
    )
    codes = {item["code"] for item in flags}
    assert "unverified_identity_3-apb" in codes
    assert "dj_spacer_identity_unresolved" in codes
    assert "uvvis_stacking_overreach" in codes
    assert "spectroscopy_transport_overreach" in codes
    assert "conductivity_coupling_overreach" in codes
    assert "unverified_identity_tas" not in codes


def test_synthesis_guard_removes_arbitrary_targets_and_adds_identity_boundary():
    guarded = guard_hypothesis_output(
        "Hypothesis:\n3-APB+ as a Dion-Jacobson spacer will increase by at least 20% "
        "and improve by approximately 15%.\n##"
    )
    assert "20%" not in guarded
    assert "15%" not in guarded
    assert "Chemical identity boundary:" in guarded
    assert "##" not in guarded


def test_quality_flags_preserve_novelty_but_reject_overclaims():
    flags = build_quality_flags(
        "Methylammonium stearate (MSG) should improve crystallinity by at least 20%. "
        "XRD and SEM will directly measure stronger interfacial bonding.",
        strong_evidence_count=0,
    )
    codes = {item["code"] for item in flags}
    assert "no_direct_support" in codes
    assert "msg_identity_conflict" in codes
    assert "unsupported_quantitative_target" in codes
    assert "xrd_bonding_overreach" in codes
    assert "sem_mechanism_overreach" in codes


def test_hypothesis_retrieval_uses_fast_evidence_profile(monkeypatch, tmp_path):
    config = LiteratureAgentConfig.load()
    config.work_dir = str(tmp_path / "outputs")
    config.job_dir = str(tmp_path / "jobs")
    service = LiteratureAgentService(config)
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


def test_long_scientific_request_gets_one_compact_local_search_fallback(tmp_path):
    config = LiteratureAgentConfig.load()
    config.work_dir = str(tmp_path / "outputs")
    config.job_dir = str(tmp_path / "jobs")
    summaries = Path(config.work_dir) / "paper_summaries_text"
    summaries.mkdir(parents=True)
    (summaries / "diammonium_dj_summary.txt").write_text(
        "Diammonium-induced Dion-Jacobson 2D/3D perovskites\n"
        "A diammonium spacer formed a Dion-Jacobson 2D/3D perovskite and improved phase stability.",
        encoding="utf-8",
    )
    service = LiteratureAgentService(config)

    results = service.search(
        "Identify novel or underexplored diammonium DJ spacer cations for stable FAPbI3, CsPbI3, "
        "and MAPbI3 mixed 2D/3D systems. Evaluate geometry, hydrophobicity, dipole, defect "
        "passivation, band alignment, commercial availability, solubility, risks, and novelty.",
        limit=3,
    )

    assert results
    assert results[0]["strong_relevance_passed"] is True
    assert results[0]["retrieval_query_strategy"] == "compact_fallback"
    assert "Dion-Jacobson" in results[0]["retrieval_query_used"]
    assert "diammonium" in results[0]["retrieval_query_used"]


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
