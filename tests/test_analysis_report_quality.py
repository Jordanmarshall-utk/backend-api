from app.agents.report_auditor_agent import deterministic_report_audit
from app.agents.report_editor_agent import ReportEditorAgent
from app.services.report_quality_control import run_report_quality_cycle


class Memory:
    def __init__(self):
        self.values = {}
        self.events = []

    def get_var(self, key, default=None):
        return self.values.get(key, default)

    def set_var(self, key, value):
        self.values[key] = value

    def log_event(self, event_type, payload, mode=None):
        self.events.append((event_type, payload, mode))


def test_deterministic_audit_flags_overclaim_and_unanchored_values():
    context = "Measured peak center: 742.1 nm. Model R-squared: 0.98."
    report = """## Results
The measured peak center was 755.0 nm and the hypothesis is confirmed.

## Interpretation
Published studies demonstrate the same mechanism.
"""
    audit = deterministic_report_audit(
        report,
        context,
        stage="analysis",
        literature_grounded=False,
    )
    categories = {item.category for item in audit.findings}
    assert "calibration.overclaim" in categories
    assert "evidence.ungrounded_literature" in categories
    assert "evidence.unanchored_quantity" in categories


def test_editor_rejects_a_new_scientific_quantity():
    context = "Measured peak center: 742.1 nm."
    report = """## Results
The measured peak center was 742.1 nm.

## Interpretation
The current result is consistent with the proposed trend.
"""
    audit = deterministic_report_audit(report, context, stage="analysis")

    def editor(_prompt):
        return report.replace("742.1 nm", "800.0 nm") + "\n\n## Limitations and uncertainty\nMore data are needed."

    edit = ReportEditorAgent(editor).revise(
        report,
        audit,
        context,
        stage="analysis",
    )
    assert "800.0 nm" not in edit.report
    assert edit.integrity_rejections
    assert edit.source == "deterministic_integrity_fallback"


def test_quality_cycle_persists_trace_without_applying_source_changes(tmp_path):
    memory = Memory()
    context = "Measured peak center: 742.1 nm. The fitted model output was R-squared 0.98."
    report = """## Results
The measured peak center was 742.1 nm and the fitted model output was R-squared 0.98.

## Interpretation
The result suggests an association but does not establish a mechanism.
"""
    final_report, trace = run_report_quality_cycle(
        report,
        context,
        stage="analysis",
        memory=memory,
        trace_root=tmp_path,
    )
    assert "## Limitations and uncertainty" in final_report
    assert trace["source_changes_applied"] is False
    assert trace["max_edit_passes"] == 1
    assert memory.values["last_analysis_quality_trace"]["trace_path"] == trace["trace_path"]
    assert memory.events[0][0] == "pipeline_report_quality"
