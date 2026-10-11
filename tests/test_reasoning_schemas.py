import pytest
from pydantic import ValidationError

from app.agents.reasoning.schemas import ReasoningBid, ReasoningPlan, TaskPacket


def test_task_packet_uses_safe_defaults():
    first = TaskPacket(domain_agent="Hypothesis Agent", objective="Test", stage="hypothesis")
    second = TaskPacket(domain_agent="Analysis Agent", objective="Test", stage="analysis")
    first.evidence.append("E1")
    assert second.evidence == []
    assert second.session_context == {}


def test_reasoning_bid_rejects_invalid_school_and_score():
    with pytest.raises(ValidationError):
        ReasoningBid(
            school="stoic",
            applicability=1.2,
            confidence=0.5,
            rationale="No",
            proposed_subtask="No",
        )


def test_reasoning_plan_rejects_duplicate_primary_support():
    with pytest.raises(ValidationError):
        ReasoningPlan(
            primary_school="humean",
            supporting_schools=["humean"],
            rationale="Evidence",
            confidence=0.8,
        )


def test_reasoning_plan_confidence_is_bounded():
    with pytest.raises(ValidationError):
        ReasoningPlan(
            primary_school="socratic",
            rationale="Questions",
            confidence=-0.1,
        )
