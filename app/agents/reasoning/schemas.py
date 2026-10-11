"""Serializable contracts for the POLARIS Reasoning Council."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Philosophy = Literal[
    "socratic",
    "cartesian",
    "humean",
    "kantian",
    "aristotelian",
    "hegelian",
    "platonic",
]
ReasoningMode = Literal["Legacy", "Shadow", "Active"]
SelectionSource = Literal["crewai", "deterministic_fallback", "legacy", "gate"]


class TaskPacket(BaseModel):
    domain_agent: str
    objective: str
    stage: str
    evidence: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    requested_output: str = ""
    session_context: Dict[str, Any] = Field(default_factory=dict)


class ReasoningBid(BaseModel):
    school: Philosophy
    applicability: float = Field(ge=0.0, le=1.0)
    necessity: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    proposed_subtask: str
    questions_to_resolve: List[str] = Field(default_factory=list)
    constraints_to_preserve: List[str] = Field(default_factory=list)


class ReasoningContribution(BaseModel):
    school: Philosophy
    assignment: str
    analysis: str
    key_findings: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    evidence_needed: List[str] = Field(default_factory=list)


class ReasoningPlan(BaseModel):
    primary_school: Philosophy
    supporting_schools: List[Philosophy] = Field(default_factory=list, max_length=2)
    assignments: Dict[str, str] = Field(default_factory=dict)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_required: bool = True
    task_summary: str = ""
    reasoning_sequence: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    evidence_requirements: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    mode: ReasoningMode = "Shadow"
    selection_source: SelectionSource = "crewai"
    crew_success: bool = True
    fallback_used: bool = False
    error: Optional[str] = None

    @field_validator("supporting_schools")
    @classmethod
    def _unique_supporting_schools(cls, schools: List[Philosophy]) -> List[Philosophy]:
        if len(set(schools)) != len(schools):
            raise ValueError("supporting_schools must be unique")
        return schools

    @model_validator(mode="after")
    def _validate_selection(self) -> "ReasoningPlan":
        if self.primary_school in self.supporting_schools:
            raise ValueError("primary_school cannot also be a supporting school")
        selected = {self.primary_school, *self.supporting_schools}
        invalid_assignments = set(self.assignments) - selected
        if invalid_assignments:
            raise ValueError("assignments may only reference selected schools")
        return self
