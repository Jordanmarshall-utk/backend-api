import pytest

from app.agents.reasoning.reasoning_service import ReasoningService
from app.agents.reasoning.schemas import TaskPacket
from app.services.reasoning_integration import build_hypothesis_task_packet


@pytest.mark.parametrize(
    ("objective", "constraints", "contradictions", "expected"),
    [
        ("Synthesize whether experimental evidence supports the stabilization claim.", [], [], "humean"),
        ("PL and XRD imply conflicting phases; interpret the discrepancy.", [], ["PL vs XRD"], "hegelian"),
        ("Determine the next three experiments needed to separate mechanisms.", [], [], "cartesian"),
        ("Determine whether this experiment satisfies workflow constraints.", ["96-well limit"], [], "kantian"),
        ("Identify transferable principles shared by successful additives.", [], [], "platonic"),
    ],
)
def test_deterministic_fallback_benchmark(objective, constraints, contradictions, expected):
    packet = TaskPacket(
        domain_agent="Hypothesis Agent",
        objective=objective,
        stage="hypothesis_generation",
        constraints=constraints,
        contradictions=contradictions,
    )
    plan = ReasoningService().deterministic_plan(packet, mode="Shadow", error="mocked")
    assert plan.primary_school == expected
    assert len(plan.supporting_schools) <= 2


def test_crewai_plan_is_grounded_to_current_task_packet():
    objective = "Resolve conflicting PL and photoKPFM evidence for carrier behavior."
    packet = TaskPacket(
        domain_agent="Hypothesis Agent",
        objective=objective,
        stage="hypothesis_generation",
    )
    stale_plan = {
        "primary_school": "socratic",
        "supporting_schools": ["humean"],
        "assignments": {},
        "rationale": "Question assumptions and inspect evidence.",
        "confidence": 0.8,
        "task_summary": "Review the TaskPacket and bids.",
        "reasoning_sequence": ["Analyze unrelated alkyl-chain stability."],
        "unresolved_questions": ["How does alkyl chain length stabilize 2D perovskites?"],
        "evidence_requirements": ["Collect alkyl-chain phase-stability data."],
    }
    plan = ReasoningService(crew_runner=lambda _: stale_plan).plan(packet, mode="Active")
    assert plan.task_summary == objective
    assert plan.reasoning_sequence == ["socratic", "humean"]
    assert set(plan.assignments) == {"socratic", "humean"}
    assert plan.unresolved_questions == []
    assert plan.evidence_requirements == []


def test_initial_hypothesis_packet_excludes_prior_hypothesis():
    class Memory:
        values = {
            "stage": "initial",
            "last_hypothesis": "An unrelated prior alkyl-chain hypothesis.",
            "uploaded_files": [],
            "experimental_constraints": {},
            "reasoning_assumptions": [],
            "reasoning_contradictions": [],
            "research_goal": "",
            "hypothesis_round_count": 0,
            "experimental_mode": False,
        }

        def get_var(self, name, default=None):
            return self.values.get(name, default)

        def get_negative_hypotheses(self, limit=1):
            return []

    packet = build_hypothesis_task_packet(Memory(), "A new question", "hypothesis_generation")
    assert not any("Prior hypothesis exists" in item for item in packet.assumptions)
