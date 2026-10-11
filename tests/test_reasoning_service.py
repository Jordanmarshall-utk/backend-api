from app.agents.reasoning.reasoning_service import ReasoningService
from app.agents.reasoning.schemas import ReasoningPlan, TaskPacket


class FakeMemory:
    def __init__(self, mode="Shadow"):
        self.values = {"reasoning_mode": mode}
        self.events = []

    def get_var(self, name, default=None):
        return self.values.get(name, default)

    def set_var(self, name, value):
        self.values[name] = value

    def log_event(self, event_type, payload, mode):
        self.events.append((event_type, payload, mode))


def packet():
    return TaskPacket(
        domain_agent="Hypothesis Agent",
        objective="Does the evidence support this stabilization claim?",
        stage="hypothesis_generation",
    )


def valid_plan():
    return ReasoningPlan(
        primary_school="humean",
        supporting_schools=["socratic"],
        assignments={"humean": "Assess evidence", "socratic": "Test assumptions"},
        rationale="Evidence-centered task",
        confidence=0.9,
    )


def test_successful_plan_is_validated_and_stored():
    memory = FakeMemory()
    plan = ReasoningService(memory, crew_runner=lambda _: valid_plan()).plan(packet())
    assert plan.primary_school == "humean"
    assert plan.crew_success is True
    assert memory.values["last_reasoning_plan"]["primary_school"] == "humean"


def test_malformed_result_uses_safe_fallback():
    plan = ReasoningService(crew_runner=lambda _: {}).plan(packet(), mode="Shadow")
    assert plan.fallback_used is True
    assert plan.crew_success is False


def test_exception_uses_safe_fallback():
    def fail(_):
        raise RuntimeError("offline")

    plan = ReasoningService(crew_runner=fail).plan(packet(), mode="Active")
    assert plan.fallback_used is True
    assert "offline" in (plan.error or "")


def test_legacy_mode_bypasses_crewai():
    called = False

    def runner(_):
        nonlocal called
        called = True
        return valid_plan()

    plan = ReasoningService(crew_runner=runner).plan(packet(), mode="Legacy")
    assert called is False
    assert plan.mode == "Legacy"
    assert plan.reasoning_required is False
