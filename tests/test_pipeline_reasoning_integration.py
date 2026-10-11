from app.agents.reasoning.reasoning_service import ReasoningService
from app.services.reasoning_integration import (
    build_pipeline_task_packet,
    plan_pipeline_reasoning,
    prepare_pipeline_reasoning,
)


class Memory:
    def __init__(self):
        self.values = {
            "reasoning_mode": "Active",
            "research_goal": "Determine whether interface treatment changes carrier behavior.",
            "last_hypothesis": "The treatment changes interface recombination.",
            "experimental_constraints": {"plate_format": "96-well"},
            "reasoning_assumptions": [],
            "reasoning_contradictions": [],
            "uploaded_files": [],
        }

    def get_var(self, name, default=None):
        return self.values.get(name, default)

    def set_var(self, name, value):
        self.values[name] = value

    def log_event(self, *args, **kwargs):
        return None


def test_curve_fitting_routes_through_deterministic_reasoning_gate():
    memory = Memory()
    packet = build_pipeline_task_packet(
        memory,
        "Curve Fitting Agent",
        "curve_fitting",
        payload={"action": "run", "data_file": "spectra.csv"},
    )
    assert packet.objective.startswith("Calculate curve fits")
    assert ReasoningService.needs_reasoning(packet) is False


def test_active_pipeline_plan_is_stored_and_exposed_as_context():
    memory = Memory()
    plan = {
        "primary_school": "humean",
        "supporting_schools": ["kantian"],
        "assignments": {
            "humean": "Separate measurements from causal inference.",
            "kantian": "Preserve experimental constraints.",
        },
        "rationale": "The analysis requires evidence calibration and constraint checks.",
        "confidence": 0.9,
        "reasoning_required": True,
        "task_summary": "placeholder",
    }
    service = ReasoningService(memory=memory, crew_runner=lambda _: plan)
    decision = plan_pipeline_reasoning(
        memory,
        "Analysis Agent",
        "analysis",
        payload={"action": "run"},
        service=service,
    )
    assert decision.effective_policy == "humean_evidence"
    assert "Primary school: humean" in decision.context
    assert memory.values["reasoning_plan_analysis"]["primary_school"] == "humean"
    assert memory.values["reasoning_context_analysis"] == decision.context


def test_active_fallback_cannot_influence_pipeline_output():
    memory = Memory()

    def fail(_):
        raise RuntimeError("council unavailable")

    decision = plan_pipeline_reasoning(
        memory,
        "Analysis Agent",
        "analysis",
        payload={"action": "run"},
        requested_policy="auto",
        service=ReasoningService(memory=memory, crew_runner=fail),
    )
    assert decision.plan.fallback_used is True
    assert decision.effective_policy == "auto"
    assert decision.context == ""


def test_pipeline_packet_contains_compact_scientific_evidence():
    memory = Memory()
    memory.values["curve_fitting_results"] = {"A01": {"peak_nm": 742.1, "r2": 0.98}}
    packet = build_pipeline_task_packet(
        memory,
        "Analysis Agent",
        "analysis",
        payload={"action": "run"},
    )
    joined = " ".join(packet.evidence)
    assert "Current hypothesis" in joined
    assert "Curve-fitting results" in joined
    assert "742.1" in joined


def test_nonexecution_requests_do_not_invoke_council():
    memory = Memory()
    called = False

    def runner(_):
        nonlocal called
        called = True
        return {}

    body, decision = prepare_pipeline_reasoning(
        memory,
        "Experiment Agent",
        payload={"action": "save_session"},
        service=ReasoningService(memory=memory, crew_runner=runner),
    )
    assert body == {"action": "save_session"}
    assert decision is None
    assert called is False


def test_curve_fitting_execution_stays_on_deterministic_gate():
    memory = Memory()
    called = False

    def runner(_):
        nonlocal called
        called = True
        return {}

    body, decision = prepare_pipeline_reasoning(
        memory,
        "Curve Fitting Agent",
        payload={"action": "run", "data_file": "spectra.csv"},
        service=ReasoningService(memory=memory, crew_runner=runner),
    )
    assert called is False
    assert decision is not None
    assert decision.plan.reasoning_required is False
    assert decision.effective_policy == "deterministic_gate"
    assert body["reasoning_context"] == ""
