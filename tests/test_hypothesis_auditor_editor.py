import json

from app.agents.hypothesis_auditor_agent import (
    HypothesisAudit,
    HypothesisAuditorAgent,
    deterministic_hypothesis_audit,
)
from app.agents.hypothesis_editor_agent import HypothesisReportEditorAgent
from app.services.hypothesis_quality_control import run_hypothesis_quality_cycle
from app.tools.hypothesis_scientific_guard import (
    guard_hypothesis_analysis,
    guard_hypothesis_synthesis,
)


class FakeMemory:
    def __init__(self):
        self.values = {}
        self.events = []

    def set_var(self, key, value):
        self.values[key] = value

    def get_var(self, key, default=None):
        return self.values.get(key, default)

    def log_event(self, event_type, payload):
        self.events.append((event_type, payload))


def test_deterministic_auditor_preserves_novelty_but_flags_scientific_risks():
    report = """
    The novel 3-APB+ spacer forms a DJ phase. XRD d-spacing will provide direct
    evidence for stronger bonding, and AFM will assess defect density. We predict
    a 50% reduction. The proposal is testable, but no direct evidence exists.
    """
    audit = deterministic_hypothesis_audit(report)

    categories = {finding.category for finding in audit.findings}
    assert audit.verdict == "block"
    assert "chemical_identity" in categories
    assert "unsupported_quantification" in categories
    assert "measurement_validity" in categories
    assert any("novel proposal" in strength for strength in audit.strengths)


def test_cha_identity_and_replacement_numeric_targets_remain_unresolved():
    generated = """
    CHA+(PbI4)2 is cyclohexylammonium, C6H14N+, whose cyclohexyl ring improves
    mobility by approximately 20%. A bandgap shift is estimated to be around
    0.1 eV.
    """
    guarded = guard_hypothesis_synthesis(generated)

    assert "cyclohexylammonium" not in guarded.lower()
    assert "C6H14N" not in guarded
    assert "20%" not in guarded
    assert "0.1 eV" not in guarded
    assert "identity must remain unresolved" in guarded
    assert "stoichiometry unresolved" in guarded


def test_analysis_guard_rejects_identity_assignment_against_uncertain_context():
    report = "The chemical identity of CHA+ (cyclohexylammonium, C6H14N+) is defined."
    context = "The exact identity of CHA+ is uncertain. A 25% effect is predicted."
    guarded = guard_hypothesis_analysis(report, context)

    assert "cyclohexylammonium" not in guarded.lower()
    assert "remains unresolved" in guarded
    assert "Definition-required" in guarded


def test_llm_audit_json_is_merged_with_deterministic_findings():
    def llm(_prompt):
        return """```json
        {"verdict":"revise","score":0.7,"findings":[{"severity":"warning",
        "category":"evidence_calibration","claim":"supported","issue":"Analogy is presented as direct support.","required_change":"Label it as analogy."}],
        "strengths":["The proposal is falsifiable."],"summary":"Revise evidence labels.",
        "improvement_proposals":["Add a provenance-label check to the prompt."]}
        ```"""

    audit = HypothesisAuditorAgent(llm).audit(
        "A 25% improvement is predicted, although no direct evidence exists."
    )

    assert audit.source == "deterministic+humean_evidence_llm"
    assert {item.category for item in audit.findings} == {
        "unsupported_quantification",
        "evidence_calibration",
    }
    assert audit.improvement_proposals


def test_editor_runs_once_and_applies_the_final_scientific_guard():
    audit = HypothesisAudit(
        verdict="revise",
        findings=[
            {
                "severity": "warning",
                "category": "unsupported_quantification",
                "issue": "Unsupported target",
                "required_change": "Remove it",
            }
        ],
    )
    calls = []

    def editor_llm(prompt):
        calls.append(prompt)
        return """**Report on Hypothesis Evaluation**

The novel CHA+(PbI4)2 proposal remains testable, but its identity and charge
balance require clarification. A 50% reduction is expected.

**Conclusion** The claim is an untested prediction rather than validation."""

    result = HypothesisReportEditorAgent(editor_llm).revise(
        "Original report with a 50% target.", audit, "CHA+(PbI4)2 hypothesis"
    )

    assert len(calls) == 1
    assert result.changed
    assert "50%" not in result.report
    assert "charge" in result.report.lower()


def test_pass_verdict_does_not_call_editor():
    calls = []

    def editor_llm(_prompt):
        calls.append(True)
        return "should not run"

    result = HypothesisReportEditorAgent(editor_llm).revise(
        "The proposal is speculative but testable.", HypothesisAudit(verdict="pass")
    )

    assert not calls
    assert result.source in {"none", "deterministic_guard"}


def test_post_edit_audit_rejects_tof_sims_rate_proof_and_design_praise():
    report = """
    The tests are only partially aligned and the identity must remain unresolved.
    ToF-SIMS is suitable for measuring the migration rate and provides direct evidence.
    Nevertheless, the proposed tests are well-designed.
    """
    audit = deterministic_hypothesis_audit(report)

    categories = {finding.category for finding in audit.findings}
    assert audit.verdict == "revise"
    assert "measurement_validity" in categories
    assert "internal_consistency" in categories


def test_auditor_checks_transport_errors_in_hypothesis_context():
    context = """
    The treatment suppresses ion migration by reducing the activation energy for ion migration.
    Time-resolved photoluminescence will quantify the ion migration rate. The mechanism is
    expected to translate into improved PCE.
    """
    audit = deterministic_hypothesis_audit(
        "The analysis describes the proposal as speculative but testable.", context
    )

    categories = {finding.category for finding in audit.findings}
    assert audit.verdict == "block"
    assert "mechanistic_consistency" in categories
    assert "measurement_validity" in categories
    assert "evidence_calibration" in categories


def test_auditor_accepts_guarded_transport_and_measurement_boundaries():
    from app.tools.hypothesis_scientific_guard import guard_hypothesis_synthesis

    context = guard_hypothesis_synthesis(
        """
        The spacer suppresses ion migration by reducing the activation energy required for
        ion migration. Time-resolved photoluminescence decay will directly quantify the ion
        migration rate. XRD and SEM will establish a lower grain-boundary density. The resulting
        reduction in migration is expected to translate into improved PCE.
        """
    )
    audit = deterministic_hypothesis_audit(
        "The proposal remains speculative but testable, with no direct validation.", context
    )

    assert audit.verdict == "pass", [
        (finding.category, finding.issue) for finding in audit.findings
    ]


def test_post_edit_status_cannot_pass_with_flawed_hypothesis_context(tmp_path):
    def editor_llm(_prompt):
        return "The report treats the proposed mechanism as speculative and requires validation."

    _, trace = run_hypothesis_quality_cycle(
        "The initial analysis overstates the method.",
        hypothesis_context=(
            "Ion migration is suppressed by reducing the activation energy for ion migration. "
            "TRPL directly measures the ion migration rate."
        ),
        editor_llm=editor_llm,
        trace_root=tmp_path,
    )

    assert trace["scope"] == "hypothesis_and_analysis_document"
    assert trace["post_edit_deterministic_audit"]["verdict"] != "pass"
    assert trace["post_edit_deterministic_audit"]["findings"]


def test_quality_cycle_persists_trace_and_approval_only_proposal(tmp_path):
    memory = FakeMemory()

    def auditor_llm(_prompt):
        return json.dumps(
            {
                "verdict": "revise",
                "score": 0.6,
                "findings": [
                    {
                        "severity": "warning",
                        "category": "evidence_calibration",
                        "claim": "validated",
                        "issue": "The conclusion overstates the evidence.",
                        "required_change": "Call it speculative but testable.",
                    }
                ],
                "strengths": ["The hypothesis is testable."],
                "summary": "One revision is needed.",
                "improvement_proposals": [
                    "Add a stronger evidence-calibration instruction to the analysis prompt."
                ],
            }
        )

    def editor_llm(_prompt):
        return """**Report on Hypothesis Evaluation**

The proposed mechanism is novel and testable. The available evidence is
indirect, so the mechanism remains an untested prediction.

**Conclusion** Controlled experiments are required to falsify the proposal."""

    final_report, trace = run_hypothesis_quality_cycle(
        "The literature validates the mechanism, although no direct evidence exists.",
        auditor_llm=auditor_llm,
        editor_llm=editor_llm,
        memory=memory,
        trace_root=tmp_path,
    )

    assert "**Automated quality control:**" in final_report
    assert trace["max_edit_passes"] == 2
    assert trace["source_changes_applied"] is False
    assert list((tmp_path / "audits").glob("*.json"))
    proposal_files = list((tmp_path / "proposals").glob("*.json"))
    assert proposal_files
    proposal = json.loads(proposal_files[0].read_text(encoding="utf-8"))
    assert proposal["status"] == "awaiting_human_approval"
    assert proposal["applied"] is False
    assert "last_hypothesis_quality_trace" in memory.values
    assert memory.events[0][0] == "hypothesis_live_quality"
