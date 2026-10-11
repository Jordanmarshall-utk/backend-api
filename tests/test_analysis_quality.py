from app.agents.analysis_agent import AnalysisAgent
from app.services.analysis_service import (
    _guard_analysis_report,
    _hypothesis_fingerprint,
    _summarize_curve_fitting,
    get_analysis_session,
)


class AnalysisMemory:
    def __init__(self, values=None, components=None):
        self.values = values or {}
        self.components = components or {}

    def get_var(self, key, default=None):
        return self.values.get(key, default)

    def set_var(self, key, value):
        self.values[key] = value

    def view_component(self, key):
        return self.components.get(key)


def test_synthetic_analysis_cannot_confirm_physical_hypothesis():
    raw = """## Goal / Hypothesis Evaluation
The hypothesis appears to be confirmed and the data supports the hypothesis.

## Results Analysis
The fitted trend confirms the expected relationship. This trend aligns well with the hypothesis and suggests that the electronic properties are being systematically tuned by altering the composition.

## Literature Context
Published studies report the same mechanism.

## Recommended Compositions / Next Experiments
Validate the candidate.
"""
    guarded = _guard_analysis_report(
        raw,
        synthetic_data=True,
        literature_grounded=False,
    )
    lower = guarded.lower()
    assert "synthetic polaris demo data" in lower
    assert "does not test, support, reject, or confirm" in lower
    assert "no traceable literatureagent evidence packet" in lower
    assert "published studies report" not in lower
    assert "appears to be confirmed" not in lower
    assert "aligns well with the hypothesis" not in lower
    assert "electronic properties are being systematically tuned" not in lower


def test_real_grounded_analysis_preserves_literature_section():
    raw = """## Goal / Hypothesis Evaluation
The hypothesis is supported by current data.

## Literature Grounding
[E1, DOI: 10.1000/example] reports a related trend.
"""
    guarded = _guard_analysis_report(
        raw,
        synthetic_data=False,
        literature_grounded=True,
    )
    assert "10.1000/example" in guarded
    assert "Synthetic POLARIS demo data" not in guarded


def test_curve_summary_includes_full_peak_range_for_small_dataset():
    results = {
        "wells": {
            "A1": {
                "fitting_results": {
                    "success": True,
                    "quality_metrics": {"r_squared": 0.99},
                    "quality_peaks": [{"center_nm": 450.2}],
                }
            },
            "B3": {
                "fitting_results": {
                    "success": True,
                    "quality_metrics": {"r_squared": 0.98},
                    "quality_peaks": [{"center_nm": 850.0}],
                }
            },
        }
    }
    summary = _summarize_curve_fitting(results)
    assert "450.20-850.00 nm" in summary
    assert "A1" in summary and "B3" in summary


def test_list_curve_summary_does_not_truncate_later_wells():
    wells = []
    for index in range(15):
        wells.append(
            {
                "well_name": f"W{index + 1}",
                "read": 1,
                "fit": {
                    "success": True,
                    "r2": 0.99,
                    "peaks": [{"center": 450.0 + index * (400.0 / 14.0)}],
                },
            }
        )
    summary = _summarize_curve_fitting({"wells": wells})
    assert "450.00-850.00 nm" in summary
    assert "W15" in summary


def test_analysis_session_hides_outputs_without_a_finalized_hypothesis():
    memory = AnalysisMemory(
        values={
            "analysis_full_report": "Old report from another track.",
            "analysis_recommendations": ["s", "ation:", "orphaned fragment"],
        },
        components={
            "clarified_question": "A research question is not a finalized hypothesis.",
            "socratic_pass": "Pre-hypothesis probing context.",
        },
    )
    session = get_analysis_session(memory)
    assert session["has_hypothesis"] is False
    assert session["analysis_full_report"] is None
    assert session["analysis_recommendations"] == []
    assert session["stale_analysis_available"] is True
    assert session["hypothesis_context"].startswith("No finalized hypothesis")
    assert "Pre-hypothesis probing" not in session["hypothesis_context"]


def test_analysis_session_returns_only_report_for_current_hypothesis():
    hypothesis = "Increasing spacer rigidity will reduce the measured defect proxy."
    memory = AnalysisMemory(
        values={
            "last_hypothesis": hypothesis,
            "analysis_full_report": "Current report.",
            "analysis_recommendations": ["Measure the defect proxy in matched controls."],
            "analysis_hypothesis_fingerprint": _hypothesis_fingerprint(hypothesis),
        }
    )
    session = get_analysis_session(memory)
    assert session["analysis_matches_hypothesis"] is True
    assert session["analysis_full_report"] == "Current report."
    assert session["analysis_recommendations"] == [
        "Measure the defect proxy in matched controls."
    ]


def test_analysis_parser_extracts_complete_recommendation_items_only():
    report = """## Results Analysis
The model generates recommendations but does not establish a mechanism.

## Recommended Compositions / Next Experiments
Based on the acquisition function, evaluate the following candidates:

1. **PEA2PbI4 = 0.95, FAPbI3 = 0.05** | Predicted peak = 850.37 nm
2. **PEA2PbI4 = 0.88, FAPbI3 = 0.12** | Predicted peak = 850.10 nm
3. Repeat matched controls using measured data.

## Impact Assessment
This synthetic run validates software behavior only.
"""
    agent = AnalysisAgent.__new__(AnalysisAgent)
    parsed = agent._parse_analysis_response(report)
    recommendations = parsed["experiment_recommendations"]
    assert len(recommendations) == 3
    assert recommendations[0].startswith("**PEA2PbI4 = 0.95")
    assert recommendations[-1] == "Repeat matched controls using measured data."
    assert parsed["impact_assessment"] == "This synthetic run validates software behavior only."
